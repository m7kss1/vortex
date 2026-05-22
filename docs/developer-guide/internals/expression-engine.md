# Linear Expression Engine (LEE)

LEE is Vortex's expression evaluation engine. An expression is compiled once into a flat
`Vec<Opcode>` and executed by a threaded-dispatch loop — one indirect call per opcode, no
call stack growth, no tree traversal at runtime.

> TODO(m7kss1): heuristic to choose recursive tree-walker for small/simple expressions?

---

## Opcode set

Ten opcodes cover all expression shapes currently supported:

| Opcode | Operands | Semantics |
|---|---|---|
| `LoadScope` | `dst` | `regs[dst] = scope` — the top-level batch array (zero-copy Arc clone) |
| `LoadConst` | `dst, scalar` | `regs[dst] = ConstantArray::new(scalar, scope.len())` — literal broadcast to batch length |
| `LoadCapture` | `dst, array: ArrayRef` | `regs[dst] = array` — pre-extracted sub-array embedded in the opcode at compile time |
| `Call` | `dst, scalar_fn, args: [RegId]` | `regs[dst] = scalar_fn.execute(regs[args])` — generic scalar function invocation |
| `AllocBool` | `dst, init: bool` | `regs[dst] = BitBufferMut::new(scope.len(), init)` — fresh bit-buffer for in-place AND/OR reduction |
| `AndInto` | `dst, src` | `regs[dst].bits &= regs[src].as_bool_bits()` word-by-word, then free `src` |
| `OrInto` | `dst, src` | `regs[dst].bits \|= regs[src].as_bool_bits()` word-by-word, then free `src` |
| `NotInto` | `reg` | flip every bit in `regs[reg].bits` in place |
| `CaseMerge` | `dst, value, cond` | `regs[dst][i] = regs[value][i] if regs[cond][i] else regs[dst][i]` |
| `Return` | `src` | terminate; return `regs[src]` as `ArrayRef` or `Mask` |

---

## Registers

Each execution allocates a flat `Vec<OutputRegister>` of length `program.num_regs`. No
heap allocation happens between opcodes beyond what individual handlers produce.

```rust
enum OutputRegister {
    Empty,                // uninitialised — default
    View(ArrayRef),       // any Array, owned via Arc — assignment is a clone, no data copy
    Bool(BitBufferMut),   // mutable bit-packed scratch buffer for AND/OR reduction
}
```

`View` is the general slot — produced by `LoadScope`, `LoadConst`, `LoadCapture`, `Call`.
`Bool` is the fast-path slot — allocated by `AllocBool`, mutated in place by `AndInto` /
`OrInto` / `NotInto`. At `Return` a `Bool` register is frozen into a `BoolArray` (for
`execute_program`) or handed directly to `Mask::from_buffer` (for `execute_mask_program`,
skipping the freeze step entirely).

`Bool` is **strictly non-nullable**. Nullable bool inputs fall back to `Call` to preserve
Kleene semantics. A future `NullableBool { values, validity }` variant will cover that path.

Register ids are allocated in `CompileCtx::alloc_reg()` with a free-list — dead registers
are recycled across AND/OR leaves and CaseWhen branches, keeping `num_regs` small.

---

## Dispatch loop

The executor maintains a static handler table indexed by `OpTag` discriminant:

```rust
static HANDLERS: [Handler; OpTag::COUNT] = [
    h_load_scope,    // 0
    h_load_const,    // 1
    h_call,          // 2
    h_alloc_bool,    // 3
    h_and_into,      // 4
    h_or_into,       // 5
    h_not_into,      // 6
    h_return,        // 7
    h_case_merge,    // 8
    h_load_capture,  // 9
];

loop {
    let opcode = &program.opcodes[state.pc];
    let ctrl = HANDLERS[opcode.tag() as usize](&mut state)?;
    match ctrl { Next => state.pc += 1, Return => break }
}
```

One indexed indirect call per opcode. No match on opcode variants, no call stack growth.
When Rust gains stable tail-call support (`become` / `musttail`), each handler can
tail-call `HANDLERS[next_op.tag()](state)` directly — identical to PostgreSQL's
`EEOP_NEXT` computed-goto macro, expected gain ~10–20% on opcode-bound workloads.

---

## Before LEE — what was replaced

Before this module, expression evaluation in `vortex-layout` used `ArrayRef::apply()`,
which recursively builds a lazy `ScalarFnArray` tree, fires encoding-aware optimizer rules
at each node, and returns that tree without evaluating it:

```rust
pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
    if expr.is::<Root>()    { return Ok(self); }
    if expr.is::<Literal>() { return Ok(ConstantArray::new(scalar, self.len())); }

    let children: Vec<_> = expr.children().iter()
        .map(|e| self.clone().apply(e))   // ← recursive call per child
        .try_collect()?;

    let node = ScalarFnArray::try_new(expr.scalar_fn(), children, self.len())?;
    node.optimize()    // ← DictPushDown, StructGetItem, RunEnd rules fire here
}
```

Call sites then either evaluated immediately (filters) or returned the lazy tree to
the caller (projections):

```rust
// flat/reader.rs — filter, dense branch: build tree, evaluate immediately
let array = array.apply(&expr)?;
let array_mask = array.execute::<Mask>(&mut ctx)?;
mask.bitand(&array_mask)

// flat/reader.rs — filter, sparse branch: build tree, filter the lazy tree, evaluate
let array = array.apply(&expr)?;
let array = array.filter(mask.clone())?;   // forces eval of ScalarFnArray
let array_mask = array.execute::<Mask>(&mut ctx)?;
mask.intersect_by_rank(&array_mask)

// flat/reader.rs — projection: build tree, return it unevaluated
if !mask.all_true() { array = array.filter(mask)?; }
array = array.apply(&expr)?;
Ok(array)    // ScalarFnArray returned; caller evaluates with execute::<T>() later
```

`execute::<T>()` on a `ScalarFnArray` is itself recursive — each node collects its
child `ArrayRef`s and calls `scalar_fn.execute(&args, ctx)`, which calls
`arg.execute::<T>(ctx)?` on each child in turn. Stack depth equals expression depth.

### Why this was slow for filters

**Per-batch cost**: `apply()` + `optimize()` ran on every batch. For 100 000 row-groups
and an 11-node expression: 100 000 × recursive tree build + 100 000 × recursive evaluate.

**N−1 intermediate `BoolArray`s**: for an N-conjunct AND, `execute()` evaluates each
conjunct to a `BoolArray`, then AND combines two arrays into a third — N−1 extra
allocations per batch that immediately go to the allocator:

```
AND(gt(col, 8766), lt(col, 9131), lt(qty, 24))
  → BoolArray_1 + BoolArray_2 → BoolArray_3 (AND)
  → BoolArray_3 + BoolArray_4 → BoolArray_5 (AND)   ← 4 allocs for 3 conjuncts
```

LEE eliminates both costs: `apply()` + `optimize()` run once at compile time; AND chains
become `AllocBool + N×AndInto` — a single `BitBufferMut` updated in place, zero
intermediate allocs.

---

## Migration — all call sites updated

| Call site | Before | After |
|---|---|---|
| `FlatReader::filter_evaluation` dense path | `apply` → `execute::<Mask>` → `bitand` | `compile_expr` + `execute_mask_program` (mask folded in) |
| `FlatReader::filter_evaluation` sparse path | `apply` → `filter(lazy)` → `execute::<Mask>` | `execute_expr` → `filter(concrete)` → `execute::<Mask>` |
| `FlatReader::projection_evaluation` | `apply` (lazy, returned to caller) | `execute_expr` (eager, concrete `ArrayRef`) |
| `DictReader` values eval + projection | `apply` (lazy) | `execute_expr` (eager) |
| `RowIdxLayoutReader` mask + projection | `apply` + `execute::<Mask>` / lazy return | `execute_expr_mask` / `execute_expr` |
| `PartitionedExpr` mask + projection | `apply` + `execute::<Mask>` / lazy return | `execute_expr_mask` / `execute_expr` |

`ArrayRef::apply` is now `#[doc(hidden)]` and kept only for zone-map `substitute_row_count`
which needs to inspect the lazy tree before evaluation.

---

## API

```rust
use vortex_array::lee::{ArrayRefLeeExt as _, ProgramCacheSessionExt as _};

// High-level — compile + execute in one call, cache managed automatically
let result: ArrayRef = scope.execute_expr(&expr, &session)?;
let mask:   Mask     = scope.execute_expr_mask(&expr, &session)?;
let mask:   Mask     = scope.execute_expr_mask_with_input(&expr, &input_mask, &session)?;

// Low-level — compile once, reuse program across batches
let program: Arc<ExprProgram> = session.compile_expr(&expr, &scope)?;
for batch in batches {
    let mut ctx = session.create_execution_ctx();
    let mask = execute_mask_program(&program, &batch, &input_mask, &mut ctx)?;
}
```

---

## Compile pipeline

`compile(expr, &scope)` runs in two steps:

**Step 1 — encoding-aware tree (`scope.apply(expr)`).**
The same `apply()` call the old code ran on every batch. Now it runs once. Optimizer
rules fire at each node and may rewrite the tree:

- `StructGetItemRule` — resolves `get_item("col", struct)` to the concrete child array.
  At runtime there is no `get_item` call; the column lands in a `LoadCapture` opcode.
- `DictionaryScalarFnValuesPushDownRule` — rewrites `gt(dict_col_M, lit_M(v))` to
  `gt(dict_values_N, lit_N(v))`. N distinct values instead of M rows.
- `DictionaryScalarFnCodesPullUpRule` — wraps the N-row result with `take(_, codes_M)`
  to expand back to M rows.

```
gt(dict_col_M, lit_M(8766))              ← M comparisons
    ↓ after pushdown rules
take(gt(dict_values_N, lit_N(8766)), codes_M)   ← N comparisons + gather
```

For a date column with 365 distinct values in a 100 000-row batch: 274× fewer comparisons
per predicate. The old path had the same optimization — but discarded and rebuilt the
tree on every batch.

**Step 2 — lower tree to opcodes (`lower_tree`).**
Walks the optimised `ArrayRef` tree post-order and emits flat opcodes:

| Tree node | Opcode(s) |
|---|---|
| The scope array itself (`ptr_eq`) | `LoadScope` (shared, deduped) |
| `ConstantArray` with `len == scope.len()` | `LoadConst` — handler recreates at runtime |
| `ConstantArray` with `len != scope.len()` | `LoadCapture` — exact length preserved (e.g. after dict pushdown `lit_N` has length N, not M) |
| Non-nullable bool AND chain | `AllocBool(true)` + N×`AndInto` |
| Non-nullable bool OR chain | `AllocBool(false)` + N×`OrInto` |
| Non-nullable bool NOT | `AllocBool(false)` + `OrInto` + `NotInto` |
| CaseWhen | lower ELSE → reverse-order `CaseMerge` per pair |
| `ScalarFnArray` (generic) | recurse children → `Call` |
| Any other `ArrayRef` (optimizer extract: dict values, codes, struct field) | `LoadCapture` (pointer-deduped) |

`LoadCapture` embeds a live `Arc<dyn Array>` in the opcode. The handler does one `Arc`
clone at runtime — no data copy. If the same array appears in two conjuncts (e.g.
`dict_values` referenced by two date comparisons), `lower_tree` deduplicates by pointer
and reuses the same register.

Programs that emit any `LoadCapture` are marked `cacheable = false` — their opcodes
contain batch-specific pointers. They are recompiled per batch (same cost as the old
`apply()` path, but with the flat executor instead of recursive `execute()`). Programs
without `LoadCapture` are `cacheable = true` and stored in `ProgramCache`.

---

## Worked example: TPC-H Q6 predicate

```sql
l_shipdate >= 8766 AND l_shipdate < 9131
AND l_discount >= 0.05 AND l_discount <= 0.07
AND l_quantity < 24
```

After `compile()` with a plain struct scope (no dict encoding), the five non-nullable
comparisons flatten into twelve opcodes:

```
 0: LoadScope  { dst: r0 }          // r0 = View(lineitem_struct_array)
 1: Call       { dst: r1, fn: get_item("l_shipdate"), args: [r0] }
 2: Call       { dst: r2, fn: gte(_, lit(8766)), args: [r1] }
 3: AllocBool  { dst: r3, init: true }   // r3 = Bool(all-1s BitBufferMut)
 4: AndInto    { dst: r3, src: r2 }      // r3 &= r2, r2 freed
 5: Call       { dst: r2, fn: lt(_, lit(9131)), args: [r1] }   // r2 reused
 6: AndInto    { dst: r3, src: r2 }
    … three more Call + AndInto pairs for discount ×2 and quantity …
12: Return     { src: r3 }               // Bool → BoolArray or Mask
```

Only `r3` (the `Bool` scratch buffer) persists across the full reduction. All `View`
registers are freed after each `AndInto`. Compare to the old recursive path: 4
intermediate `BoolArray` allocs for 3 conjuncts; here: 0.

---

## Caching

`ProgramCache` is a `SessionVar` — one instance per `VortexSession`, never evicted.

Cache key: `(ExactExpr, DType, ArrayId)`.

- `ExactExpr` — pointer equality on the `Arc<dyn ExpressionKind>`. A scan that reuses
  the same parsed `Expression` object across all batches hits the cache in O(1).
- `DType` — separates schemas with the same expression but different field types.
- `ArrayId` (top-level encoding id) — a `DictArray` and a `PrimitiveArray` with the same
  dtype produce different opcode streams; they get separate entries.

`OnceLock<Arc<ExprProgram>>` per entry ensures at-most-once compilation under concurrent
access.

Non-cacheable programs (`cacheable = false`, i.e. those with `LoadCapture`) are compiled
on every batch and never inserted into the cache — no stale entries accumulate.

---

## Filter-mask fast path

`execute_mask_program` avoids the `BoolArray → Mask` round-trip for AND-reduction programs:

1. `AllocBool { init: true }` — seeds the `Bool` register from `input_mask` bits.
2. Each `AndInto` — reduces the scratch buffer in place.
3. `Return` — hands `Bool(BitBufferMut)` directly to `Mask::from_buffer`, no freeze.
4. `input_mask` was already folded in at step 1, so no final `bitand` is needed.

For non-AND programs (projections, OR-chains) the result is a `View(ArrayRef)`, converted
via `array.execute::<Mask>(ctx)` and intersected with `input_mask`.

---

## Extending LEE

See [`vortex-array/src/lee/README.md`](../../../vortex-array/src/lee/README.md) for the
step-by-step recipe for adding a new fusion opcode.

---

## Migration notes

`ArrayRef::apply` is `#[doc(hidden)]`. Use `ArrayRefLeeExt` methods instead. The three
remaining internal callers that use `apply()` for zone-map `substitute_row_count` will be
migrated once row-count substitution is moved to the `Expression` level.

---

## Future work

### 1. Cacheable programs for dict/struct columns

Currently programs that touch dict-encoded or struct columns contain `LoadCapture` opcodes
with batch-specific pointers and are marked `cacheable = false` — recompiled per batch.

Fix: replace `LoadCapture` with runtime extraction opcodes: `Call(get_dict_values, reg)` /
`Call(get_dict_codes, reg)` / `Call(get_item("col"), scope_reg)`. This requires adding
`GetDictValues` / `GetDictCodes` scalar functions. The resulting program is fully
runtime-derived and safely cacheable.

### 2. `NullableBool` register variant

Add `OutputRegister::NullableBool { values: BitBufferMut, validity: BitBufferMut }` and
Kleene-aware `AndInto` / `OrInto` / `NotInto` variants. Currently nullable bool inputs
fall back to `Call`, which allocates an intermediate `BoolArray` per conjunct.

### 3. Fused opcodes

- `LoadColumn { dst, field }` — fuse `LoadScope + Call(get_item)` for struct field access.
- `CompareConst { dst, src, op, scalar }` — fuse `LoadConst + Call(cmp)`.
- `IsNull` / `IsNotNull` — direct validity-bitmap copies, no kernel invocation.
- `FilterTake { dst, src, mask }` — fuse the `array.filter(mask)?` step currently done
  outside the program in the sparse-density filter branch of `FlatReader`.

### 4. Cache eviction

`ProgramCache` is currently unbounded. LRU with a configurable cap (e.g. 1024 entries)
is sufficient — compilation is cheap relative to a typical scan.

### 5. Heuristic for tiny expressions

For depth ≤ 2 expressions on small batches, compile overhead ≈ evaluation cost. A
heuristic could bypass compilation and run a direct interpreter. This is the `TODO(m7kss1)`
at the top of this doc.

### 6. Tail-call dispatch

When `become` / `musttail` lands in stable Rust, each handler can tail-call its successor
directly — identical to PostgreSQL's `EEOP_NEXT`. Expected gain: ~10–20% on
opcode-bound workloads.

### 7. Retire `ArrayRef::apply`

Three internal callers still use the lazy tree for zone-map `substitute_row_count`.
Once row-count substitution moves to the `Expression` level, `expression.rs` can be
deleted and `ScalarFnArray` becomes `pub(crate)`.

### 8. Public-API snapshot regeneration

Run `./scripts/public-api.sh` before merge. Expected diffs: `+lee::*`, `+ArrayRefLeeExt`,
`-apply()` (now `doc(hidden)`), `-cranelift`, `-expr_v2_jit`.
