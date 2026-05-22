# Linear Expression Engine (LEE)

LEE is the single expression evaluation engine for Vortex. Every expression evaluation — filter
pushdown, projection, pruning — goes through one path: `compile()` → `ExprProgram` →
`execute_program` / `execute_mask_program`.

## Component map

| File | Responsibility |
|---|---|
| `mod.rs` | Public exports: `compile`, `execute_program`, `execute_mask_program`, `ExprProgram`, `ProgramCache`, `ArrayRefLeeExt` |
| `compile.rs` | `compile(expr, &ArrayRef)` → `ExprProgram`. Phase A+B (apply + optimize) then Phase C (lower_tree → opcodes). |
| `execute.rs` | Threaded-dispatch executor. `run_program` → handler-table loop; `execute_program` / `execute_mask_program` as public entry points. |
| `program.rs` | `ExprProgram` struct: opcode list + register count + result register + scope dtype + cacheable flag. |
| `opcode.rs` | `Opcode` enum (10 variants) and `OpTag` discriminant used to index the handler table. |
| `register.rs` | `OutputRegister` enum (`Empty`, `View(ArrayRef)`, `Bool(BitBufferMut)`) and `RegId` type alias. |
| `cache.rs` | `ProgramCache` as a `SessionVar`: `DashMap<CacheKey, Arc<OnceLock<Arc<ExprProgram>>>>`. Non-cacheable programs bypass the cache. |
| `ext.rs` | `ArrayRefLeeExt` extension trait: `execute_expr`, `execute_expr_mask`, `execute_expr_mask_with_input`. |

## Compile pipeline

```
Expression  +  scope ArrayRef
        │
        ▼
compile(expr, scope)           ← public entry (compile.rs)
  │
  ├── Phase A: scope.apply(expr)
  │     ├── Root → scope itself (Arc clone)
  │     ├── Literal → ConstantArray::new(scalar, scope.len())
  │     └── other → ScalarFnArray::try_new(fn, children) → optimize()
  │           └── ArrayOptimizer fires encoding-specific rules at each node:
  │                 StructGetItemRule     (resolves get_item to a concrete column)
  │                 DictValuesPushDown    (pushes fn into dict values)
  │                 DictCodesPullUp       (wraps result with take(_, codes))
  │                 RunEnd, Chunked, …
  │
  ├── Phase C: lower_tree(optimised ArrayRef tree)
  │     ├── scope array itself (ptr_eq) → LoadScope opcode  [register shared]
  │     ├── ConstantArray, len == scope.len() → LoadConst   [cacheable]
  │     ├── ConstantArray, len != scope.len() → LoadCapture [non-cacheable]
  │     ├── And chain (non-nullable) → AllocBool + N×AndInto  [flattened]
  │     ├── Or chain (non-nullable) → AllocBool + N×OrInto    [flattened]
  │     ├── Not (non-nullable Bool) → NotInto in place
  │     ├── CaseWhen → CaseMerge opcodes (reverse WHEN order)
  │     ├── ScalarFnArray (generic) → recurse children → Call opcode
  │     └── any other ArrayRef (sub-array extracted by optimizer)
  │           → LoadCapture  [non-cacheable; deduplicated by pointer]
  │
  └── append Return opcode
        │
        ▼
   ExprProgram { opcodes, num_regs, result_reg, scope_dtype, cacheable }
```

`cacheable = true` when no `LoadCapture` opcode was emitted (e.g. scope is a plain
`PrimitiveArray` or the expression uses only `root()` directly). `cacheable = false` when
the optimizer extracted sub-arrays — the program is valid only for the scope it was built
from, and `ProgramCache` returns it without storing it.

## Opcode set

| Opcode | Semantics |
|---|---|
| `LoadScope { dst }` | `regs[dst] = scope` (zero-copy view) |
| `LoadConst { dst, scalar }` | `regs[dst] = ConstantArray::new(scalar, scope.len())` |
| `LoadCapture { dst, array }` | `regs[dst] = array` — pre-extracted sub-array from scope (Phase A/B) |
| `Call { dst, scalar_fn, args }` | `regs[dst] = scalar_fn.execute(args)` — generic fallback |
| `AllocBool { dst, init }` | Allocate a `BitBufferMut` filled with `init` (AND identity = true, OR identity = false) |
| `AndInto { dst, src }` | `regs[dst].bool_buf &= regs[src].as_bool_bits()` in place, word by word |
| `OrInto { dst, src }` | `regs[dst].bool_buf \|= regs[src].as_bool_bits()` in place, word by word |
| `NotInto { reg }` | Flip every bit in `regs[reg].bool_buf` in place |
| `CaseMerge { dst, value, cond }` | `regs[dst][i] = regs[value][i] if regs[cond][i] else regs[dst][i]` |
| `Return { src }` | Terminate; return `regs[src]` |

## Register model

Each register is an `OutputRegister`:

- `Empty` — uninitialized (default). Handlers write here before any other handler reads it.
- `View(ArrayRef)` — holds an owned `ArrayRef`. Set by `LoadScope`, `LoadConst`, `LoadCapture`, `Call`.
- `Bool(BitBufferMut)` — a mutable bit-packed scratch buffer. Set by `AllocBool`, mutated by
  `AndInto`, `OrInto`, `NotInto`.

Register IDs are allocated by `CompileCtx::alloc_reg()` during compilation. A free-list
(`CompileCtx::free_list`) allows temporary registers to be reused within sub-expressions,
keeping `num_regs` small.

## Threaded dispatch loop

```
static HANDLERS: [Handler; OpTag::COUNT] = [
    h_load_scope,   // 0
    h_load_const,   // 1
    h_call,         // 2
    h_alloc_bool,   // 3
    h_and_into,     // 4
    h_or_into,      // 5
    h_not_into,     // 6
    h_return,       // 7
    h_case_merge,   // 8
];

loop {
    let opcode = &program.opcodes[state.pc];
    let ctrl = HANDLERS[opcode.tag() as usize](&mut state)?;
    match ctrl {
        Next   => state.pc += 1,
        Return => break,
    }
}
```

Each handler reads its inputs from `state.regs` and writes its output in-place. On stable Rust
this is one indirect call per opcode. When `become`/`musttail` lands, each handler can
tail-call its successor directly — identical to PostgreSQL's computed-goto `EEOP_NEXT` macro.

## Filter-mask fast path

`execute_mask_program` avoids the `BoolArray → Mask` round-trip for AND-reduction programs:

1. `AllocBool { init: true }` is seeded from `input_mask` (the `mask_seeded` flag is set).
2. Each `AndInto` reduces the scratch buffer in place — no intermediate `BoolArray` is allocated.
3. At `Return`, the handler returns the `Bool(BitBufferMut)` directly as a `Mask::from_buffer`.
4. Because the input mask was already folded in at step 1, no final `bitand` is needed.

For non-AND programs (projections, OR-chains, `Call`-heavy paths) the result is a `View(ArrayRef)`,
which is converted via `array.execute::<Mask>(ctx)` and then intersected with `input_mask`.

## Session-scoped program cache

`ProgramCache` (in `cache.rs`) is a `SessionVar` — one instance per `VortexSession`. The cache
key is `(ExactExpr, DType, ArrayId)`:

- `ExactExpr` provides pointer-equality hashing on expression trees for O(1) lookups when the
  same `Expression` object is reused across batches.
- `DType` ensures programs are not shared across schema changes.
- `ArrayId` (the top-level encoding id) distinguishes arrays with the same dtype but different
  physical encodings, which will produce different programs once encoding-aware Phase B rules
  are active.

`OnceLock<Arc<ExprProgram>>` per cache entry ensures the program is compiled at most once even
under concurrent access.

Access the cache via the `ProgramCacheSessionExt` trait:
```rust
use vortex_array::lee::ProgramCacheSessionExt as _;
let program = session.compile_expr(&expr, &scope)?;
```

Or use the higher-level `ArrayRefLeeExt` trait which handles both compilation and execution:
```rust
use vortex_array::lee::ArrayRefLeeExt as _;
let result = scope.execute_expr(&expr, &session)?;
let mask   = scope.execute_expr_mask(&expr, &session)?;
let mask   = scope.execute_expr_mask_with_input(&expr, &input_mask, &session)?;
```

## How to add a new fusion opcode

1. **Define the opcode.** Add a variant to `Opcode` in `opcode.rs` and increment `OpTag::COUNT`.
   Add the corresponding `OpTag` variant and a match arm in `Opcode::tag()`.

2. **Recognise the pattern in `compile.rs`.** In the `lower()` function, add a branch that
   detects the target expression shape (e.g., a specific `ScalarFnRef` type or expression
   combinator) and emits the new opcode instead of falling back to `Call`.

3. **Write the handler in `execute.rs`.** Add `h_my_opcode` with signature
   `fn(&mut ExecState<'_>) -> VortexResult<HandlerCtrl>`. Add it to the `HANDLERS` array at
   the index matching the new `OpTag` discriminant.

4. **Add a test.** Compile an expression that triggers the new opcode, execute it against a
   known scope, and assert the result matches the legacy `apply() + execute::<T>()` path.

## Testing strategy

- Unit tests in `execute.rs` compare LEE results against the legacy `apply+execute` path for
  multiple expression shapes (bool AND, comparison, CaseWhen, nested).
- Benchmarks in `vortex-array/benches/expr/linear_filter.rs` pair `legacy_*` vs `linear_*`
  functions for direct apples-to-apples comparison.
- IAI-Callgrind benchmarks in `vortex-array/benches/iai_lee.rs` track instruction counts for
  regression detection.
