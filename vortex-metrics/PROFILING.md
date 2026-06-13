# Vortex Profiling

## TLDR

Base profile, no root:

```bash
cargo build -p vortex-tui --features profile --bin vx
vx profile query ./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex \
  --sql "select count(*) from data" \
  --json /tmp/vortex-profile.json
```

With the eBPF kernel layers (each opt-in, all need root):

```bash
cargo build -p vortex-tui --features profile-ebpf --bin vx
sudo -E target/debug/vx profile query ./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex \
  --sql "select count(*) from data where l_quantity > 20" \
  --syscalls --bio --locks --net \
  --json /tmp/vortex-profile-ebpf.json
```

Off-CPU and per-encoding PMU additionally need a `profile-pmu` build (it compiles
the context markers the kernel attributes to):

```bash
cargo build -p vortex-tui --features profile-pmu --bin vx
sudo -E target/debug/vx profile query <file> --sql <SQL> --offcpu --pmu --json /tmp/r.json
```

Compare two reports:

```bash
vx profile diff /tmp/before.json /tmp/after.json
```

The opt-in eBPF layers (compose freely):

| Flag | Build | Adds |
| --- | --- | --- |
| `--syscalls` | `profile-ebpf` | read-syscall count, size histogram, **latency** |
| `--bio` | `profile-ebpf` | block-layer device reads, bytes, service latency |
| `--locks` | `profile-ebpf` | blocking-futex (lock) wait count + latency |
| `--net` | `profile-ebpf` | TCP retransmits + connection setup |
| `--offcpu` | `profile-pmu` | off-CPU (blocked) time per phase + runqueue latency |
| `--pmu` | `profile-pmu` | per-encoding hardware counters (samples) |

## What It Measures

```text
                 no root                         root
Vortex code  ---------------->  vortex-metrics  --------> JSON report
 rows, bytes, decode time       deterministic
 pruning, fallbacks             counters

Linux kernel ---------------->  vortex-ebpf    --------> same JSON report
 read/block/futex/tcp/sched     optional, root
 syscall+device+lock+net+       --syscalls --bio
 offcpu+PMU                     --locks --net --offcpu --pmu
```

In-process metrics are the main signal:

- `scan.*`
- `decode.*`
- `filter.*`
- `pruning.*`
- `io.*`
- `pushdown_fallback.*`
- `memory.*`
- `cold.*`

eBPF metrics are optional, one group per flag:

- `--syscalls`: `io.read_syscalls`, `io.read_syscall_bytes`,
  `io.read_size_p50_bytes`, `io.read_size_p99_bytes`, `io.tiny_reads`,
  `io.tiny_read_threshold_bytes`, `io.read_size_hist.*`,
  `io.read_latency_p50_us`, `io.read_latency_p99_us`
- `--bio`: `bio.device_reads`, `bio.device_read_bytes`,
  `bio.read_latency_p50_ms`, `bio.read_latency_p99_ms`
- `--locks`: `lock.futex_waits`, `lock.futex_wait_p50_us`,
  `lock.futex_wait_p99_us`
- `--net`: `net.tcp_retransmits`, `net.tcp_connections`,
  `net.connect_latency_p50_ms`, `net.connect_latency_p99_ms`
- `--offcpu`: `offcpu.<phase>.ms`, `offcpu.<phase>.count`, `offcpu.total_ms`,
  `sched.runqueue_latency_p50_us`, `sched.runqueue_latency_p99_us`
- `--pmu`: `pmu.<enc>.*_samples`

The report also has a `notes` object next to `metrics`. It holds short text
that explains a number when the number alone is not clear (for now only the
pushdown fallback). `notes` is text, not numbers, so the diff gate ignores it.

## Metrics Reference

All keys below live inside the `metrics` object of the JSON report. The `Source`
column says where the number comes from:

- `in-process` — always present, no root, no eBPF.
- `--syscalls`, `--bio`, `--locks`, `--net` — present only with that eBPF layer
  (needs root and a `profile-ebpf` build). Missing in the base profile.
- `--offcpu`, `--pmu` — present only with that eBPF layer (needs root and a
  `profile-pmu` build, whose context markers the kernel attributes to). Missing
  otherwise.

Scope: `--syscalls`/`--locks` are pid-scoped (this `vx` process). `--bio`/`--net`
are **system-wide** for the run (the kernel serves block/TCP work asynchronously
with no reliable originating pid) — run on an otherwise-idle host. `--offcpu`/
`--pmu` are scoped in-kernel to the threads with a Vortex phase in flight.

`<enc>` is one encoding id (`vortex.pco`, `vortex.cast`, ...). `<lo>` is a byte
value. The keys with `<...>` repeat once per encoding / bucket / pair.

### scan

| Key | Source | Meaning |
| --- | --- | --- |
| `scan.rows_out` | in-process | rows the scan returned to the engine |
| `scan.splits` | in-process | number of scan splits (row-range tasks) |
| `scan.split_duration_p50_ms` | in-process | median split wall time, ms |
| `scan.split_duration_p99_ms` | in-process | p99 split wall time, ms |
| `scan.split_peak_concurrent` | in-process | max splits running at the same time |

### pruning and filter

| Key | Source | Meaning |
| --- | --- | --- |
| `pruning.rows_in` | in-process | rows seen by statistics pruning |
| `pruning.rows_kept` | in-process | rows that survived pruning |
| `pruning.pruned_ratio` | in-process | `1 - kept/in`. Higher is better (more skipped) |
| `pruning.conjunct.<n>.{rows_in,rows_kept,ms}` | in-process | same, per predicate conjunct |
| `filter.rows_in` | in-process | rows seen by the row filter |
| `filter.rows_kept` | in-process | rows that passed the filter |
| `filter.selectivity` | in-process | `kept/in` |
| `filter.conjunct.<n>.{rows_in,rows_kept,ms,selectivity}` | in-process | same, per conjunct |

### io and metadata

| Key | Source | Meaning |
| --- | --- | --- |
| `io.segment_requests` | in-process | logical segment reads the scan asked for |
| `io.logical_segment_bytes` | in-process | bytes of those logical segments |
| `io.physical_reads` | in-process | physical reads after coalescing |
| `io.physical_read_bytes` | in-process | bytes actually read from the store |
| `io.read_amplification` | in-process | `physical_bytes / logical_bytes` |
| `io.coalescing_factor_avg` | in-process | `segment_requests / physical_reads` |
| `io.segment_size_p50_bytes` | in-process | median logical segment size |
| `io.segment_size_p99_bytes` | in-process | p99 logical segment size |
| `metadata.footer_reads` | in-process | footer read count |
| `metadata.footer_bytes` | in-process | footer bytes read |

### decode

| Key | Source | Meaning |
| --- | --- | --- |
| `decode.total_calls` | in-process | decode calls over all encodings |
| `decode.total_ms` | in-process | decode time over all encodings, ms |
| `decode.<enc>.calls` | in-process | decode calls for this encoding |
| `decode.<enc>.ms` | in-process | decode time for this encoding, ms |
| `decode.<enc>.rows` | in-process | rows this encoding produced |
| `decode.<enc>.bytes` | in-process | own-buffer bytes this encoding produced |
| `decode.<enc>.rows_per_sec` | in-process | `rows / decode_seconds` |
| `decode.<enc>.mb_per_sec` | in-process | `bytes / decode_seconds` |

### pushdown fallback

| Key | Source | Meaning |
| --- | --- | --- |
| `pushdown_fallback.total` | in-process | total compute-on-encoded misses |
| `pushdown_fallback.<parent>=><child>.count` | in-process | how many times `parent` had to canonicalize `child` instead of pushing compute into it. Lower is better |

### memory and cold cache

| Key | Source | Meaning |
| --- | --- | --- |
| `memory.rss_peak_bytes` | in-process | peak RSS, bytes (`/proc/self/status` VmHWM) |
| `memory.rss_peak_delta_bytes` | in-process | peak RSS minus RSS before the query |
| `cold.storage_read_bytes` | in-process | bytes pulled from storage (`/proc/self/io`) |
| `cold.page_cache_hit_ratio` | in-process | `1 - storage_read/logical_read`. Cache dependent, **not reproducible** |
| `cold.major_faults` | in-process | major page faults during the query |

`cold.*` depends on page cache state. The same query gives different `cold.*`
on a warm run and a cold run. `vx` prints a warning to stderr when the cache was
warm. For a cold run drop caches first:

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### read syscalls (eBPF only)

These keys exist **only** with `--syscalls`. The base profile has no kernel
view, so it cannot produce them.

| Key | Source | Meaning |
| --- | --- | --- |
| `io.read_syscalls` | `--syscalls` | `read` + `pread64` syscalls during the query |
| `io.read_syscall_bytes` | `--syscalls` | bytes returned by those syscalls |
| `io.read_size_p50_bytes` | `--syscalls` | median read size (log2 bucket lower bound) |
| `io.read_size_p99_bytes` | `--syscalls` | p99 read size (log2 bucket lower bound) |
| `io.tiny_reads` | `--syscalls` | reads below the tiny threshold |
| `io.tiny_read_threshold_bytes` | `--syscalls` | what "tiny" means (4096 = 4 KiB) |
| `io.read_size_hist.<lo>` | `--syscalls` | read count in log2 bucket `[<lo>, 2*<lo>)`, only non-empty buckets |
| `io.read_latency_p50_us` | `--syscalls` | median bare-syscall read latency, µs |
| `io.read_latency_p99_us` | `--syscalls` | p99 bare-syscall read latency, µs |

`io.read_size_hist.*` is the full size breakdown. Use it to find where the small
reads sit. Example: a high `io.read_size_hist.32` means many reads in `[32, 64)`
bytes — metadata, not data. `io.read_latency_*` is the kernel-side service time
of the `read`/`pread64` itself (page-cache hit vs disk), separate from the
in-process time the scan spends queued on the blocking pool.

### block-layer device reads (eBPF only)

These keys exist **only** with `--bio`. They are the read traffic that reached the
block device *past* the page cache — the honest cold-cache signal. **System-wide**
for the run (the block layer has no reliable originating pid): run on an idle
host. A warm-cache run shows `bio.device_reads` near zero; a cold run shows the
real device IO.

| Key | Source | Meaning |
| --- | --- | --- |
| `bio.device_reads` | `--bio` | block-device read requests completed during the run |
| `bio.device_read_bytes` | `--bio` | bytes those reads moved (`nr_sector × 512`) |
| `bio.read_latency_p50_ms` | `--bio` | median device read service latency, ms |
| `bio.read_latency_p99_ms` | `--bio` | p99 device read service latency, ms |

### lock contention (eBPF only)

These keys exist **only** with `--locks`. They time blocking futex waits
(`FUTEX_WAIT`/`FUTEX_WAIT_BITSET`) — the kernel side of contended
`parking_lot`/`std` mutexes, the metrics registry lock, segment-cache locks, etc.
Pid-scoped to this `vx` process.

| Key | Source | Meaning |
| --- | --- | --- |
| `lock.futex_waits` | `--locks` | blocking-futex waits during the query |
| `lock.futex_wait_p50_us` | `--locks` | median wait time, µs |
| `lock.futex_wait_p99_us` | `--locks` | p99 wait time, µs |

### network (eBPF only)

These keys exist **only** with `--net`. For remote (object-store / S3) reads they
expose the TCP behaviour the in-process view cannot see (it lives inside
`object_store`/`reqwest`). **System-wide** for the run. On a local-file query they
are all zero.

| Key | Source | Meaning |
| --- | --- | --- |
| `net.tcp_retransmits` | `--net` | TCP retransmits during the run (network trouble) |
| `net.tcp_connections` | `--net` | new TCP connections established |
| `net.connect_latency_p50_ms` | `--net` | median `SYN_SENT`→`ESTABLISHED` time, ms |
| `net.connect_latency_p99_ms` | `--net` | p99 connection setup time, ms |

### off-CPU and scheduler (eBPF only)

These keys exist **only** with `--offcpu` (`profile-pmu` build, root). Off-CPU is
the time a thread spent **blocked** (descheduled) while a Vortex phase was in
flight, attributed to that phase via the context markers — the answer to "is this
phase compute-bound or waiting?". Runqueue latency is the wakeup→run scheduling
delay of those threads (CPU oversubscription). `<phase>` is the encoding for a
decode (`vortex.pco`), else the phase and instance (e.g. `filter.conjunct[0]`).

| Key | Source | Meaning |
| --- | --- | --- |
| `offcpu.<phase>.ms` | `--offcpu` | off-CPU (blocked) time attributed to `<phase>`, ms |
| `offcpu.<phase>.count` | `--offcpu` | off-CPU episodes for `<phase>` |
| `offcpu.total_ms` | `--offcpu` | off-CPU time over all phases, ms |
| `sched.runqueue_latency_p50_us` | `--offcpu` | median runqueue (wakeup→run) latency, µs |
| `sched.runqueue_latency_p99_us` | `--offcpu` | p99 runqueue latency, µs |

### PMU (eBPF only)

These keys exist **only** with `--pmu` (`profile-pmu` build, root). They are
samples, not exact counts. Do not use them as a hard regression gate. Each
hardware counter is best-effort: counters unavailable on the host (common on
virtualized / cloud instances, where `perf_event_open` is restricted) are skipped
with a warning and their keys are absent.

| Key | Source | Meaning |
| --- | --- | --- |
| `pmu.<enc>.cycle_samples` | `--pmu` | CPU-cycle samples attributed to this encoding's decode |
| `pmu.<enc>.instruction_samples` | `--pmu` | retired-instruction samples |
| `pmu.<enc>.cache_miss_samples` | `--pmu` | cache-miss samples |
| `pmu.<enc>.branch_miss_samples` | `--pmu` | branch-misprediction samples (FSST/ALP hot spots) |
| `pmu.<enc>.llc_load_miss_samples` | `--pmu` | last-level-cache load-miss samples (wide bit-packing) |
| `pmu.<enc>.stalled_cycle_samples` | `--pmu` | backend-stalled-cycle samples (memory-bound decode) |

## Diff Gate

`vx profile diff` compares two reports and fails (exit 1) when a hard counter
moved the wrong way past the tolerance (default 5%). Gated counters:

| Counter pattern | Bad direction | Why |
| --- | --- | --- |
| `decode.total_calls`, `decode.<enc>.calls` | increase | more decode work |
| `pushdown_fallback.total`, `pushdown_fallback.<parent>=><child>.count` | increase | more compute-on-encoded misses |
| `io.read_syscalls`, `io.read_syscall_bytes`, `io.tiny_reads` | increase | more / smaller IO (eBPF only) |
| `io.read_amplification`, `io.physical_reads`, `io.physical_read_bytes`, `io.segment_requests` | increase | more IO |
| `metadata.footer_reads`, `metadata.footer_bytes` | increase | more metadata IO |
| `pruning.pruned_ratio` | decrease | less skipped |
| `scan.rows_out`, `filter.rows_kept` | any change | correctness; must stay equal |

The per-encoding `decode.<enc>.calls` and per-pair
`pushdown_fallback.<parent>=><child>.count` are gated by pattern. The grand
total can stay flat while work shifts onto one encoding; the per-key gate still
catches it. `cold.*` and `pmu.*` are not gated — they are not reproducible.

The kernel-layer signals added by `--bio`, `--locks`, `--net`, `--offcpu` and the
`io.read_latency_*` / `sched.*` keys are **not gated** either: they are latencies,
system-wide counts, or scheduler/blocking timings that depend on host load and
cache state, not deterministic per-query counters. Use them as diagnostics, not
as a regression gate. The gated counters stay the deterministic in-process ones
(plus the `--syscalls` read counts/sizes).

## How `--syscalls` Works

```text
vx profile query --syscalls
        |
        v
vortex-ebpf host loader
        |
        v
attach tracepoints:
  syscalls:sys_enter_read
  syscalls:sys_enter_pread64
        |
        v
BPF maps:
  READ_STATS = count, bytes
  READ_HIST  = log2(size) buckets
        |
        v
render metrics into JSON
```

The tracepoints are scoped to the current `vx` process pid. They count read
syscalls made while the query runs.

## Requirements

For base profile:

- normal user is fine
- no BPF
- no root

For `--syscalls` / `--bio` / `--locks` / `--net`:

- Linux
- root or enough `CAP_BPF` / `CAP_PERFMON`
- a `profile-ebpf` build
- `bpf-linker` installed:

```bash
cargo install bpf-linker
```

For `--offcpu` / `--pmu`:

- the above, plus a `profile-pmu` build (`--features profile-pmu`) so the
  context markers exist for the kernel to attribute to
- `--pmu` additionally needs the host to expose hardware PMU counters via
  `perf_event_open`; on many virtualized/cloud instances these are restricted, in
  which case the unavailable counters are skipped with a warning

For CI/check without BPF toolchain:

```bash
VORTEX_EBPF_SKIP_BPF=1 cargo check -p vortex-tui --features profile-ebpf --no-default-features
```

This builds a stub object. It is only for compile check. It cannot collect
syscall stats.

## Real Workload Example

A TPC-H Q6-style aggregate touches four columns under several predicates — good
for seeing pushdown, statistics pruning, and lazy compute-on-encoded at once. Run
with every layer (a `profile-pmu` build, root, cold cache):

```bash
FILE=./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex
SQL="select sum(l_extendedprice * l_discount) as revenue
     from data
     where l_shipdate >= date '1994-01-01' and l_shipdate < date '1995-01-01'
       and l_discount between 0.05 and 0.07
       and l_quantity < 24"

cargo build -p vortex-tui --features profile-pmu --bin vx
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

sudo -E target/debug/vx profile query "$FILE" --sql "$SQL" \
  --syscalls --bio --locks --net --offcpu --pmu \
  --json /tmp/q6.json
```

`metrics` excerpt (selected keys from the full report; `pmu.*` is absent because
this cloud host restricts `perf_event_open`):

```json
{
  "scan.rows_out": 11618,
  "scan.splits": 2,
  "scan.split_peak_concurrent": 2,

  "pruning.pruned_ratio": 0.0,
  "pruning.rows_in": 1801716,
  "pruning.rows_kept": 1801716,

  "filter.rows_in": 722164,
  "filter.rows_kept": 133210,
  "filter.selectivity": 0.1845,

  "decode.total_calls": 339,
  "decode.vortex.pco.calls": 44,
  "decode.vortex.pco.bytes": 11321212,
  "decode.vortex.pco.mb_per_sec": 15.9328,
  "decode.vortex.decimal_byte_parts.calls": 33,
  "decode.fastlanes.bitpacked.calls": 11,
  "decode.vortex.filter.calls": 66,
  "decode.vortex.filter.bytes": 0,
  "decode.vortex.between.calls": 22,
  "decode.vortex.between.bytes": 0,
  "decode.vortex.binary.calls": 77,
  "decode.vortex.binary.bytes": 0,

  "pushdown_fallback.total": 22,
  "pushdown_fallback.vortex.filter=>fastlanes.bitpacked.count": 11,
  "pushdown_fallback.vortex.filter=>vortex.pco.count": 11,

  "io.physical_reads": 12,
  "io.read_amplification": 1.1329,
  "io.coalescing_factor_avg": 2.0833,
  "io.tiny_reads": 138,
  "io.read_latency_p50_us": 2.048,
  "io.read_latency_p99_us": 1073741.824,

  "bio.device_reads": 19,
  "bio.device_read_bytes": 3883008,
  "bio.read_latency_p99_ms": 2.0972,

  "lock.futex_waits": 773,
  "lock.futex_wait_p99_us": 33554.432,

  "offcpu.vortex.pco.ms": 0.6625,
  "offcpu.total_ms": 0.7296,
  "sched.runqueue_latency_p99_us": 131.072,

  "net.tcp_retransmits": 0,
  "net.tcp_connections": 0
}
```

### What the report shows

- **Compute-on-encoded (laziness).** The compute nodes `vortex.filter`,
  `vortex.between`, `vortex.binary` ran with `bytes: 0` — they evaluated over
  *encoded* children without materializing their own buffers. Only the stored
  column encodings (`vortex.pco`, `vortex.decimal_byte_parts`,
  `fastlanes.bitpacked`) produced real bytes. So most of the predicate work
  happened without canonicalizing columns.
- **Pushdown misses (where laziness leaked).** `pushdown_fallback.total` = 22:
  the row filter could not push compute into `fastlanes.bitpacked` (11×) or
  `vortex.pco` (11×), so those were canonicalized first (the `notes` object spells
  this out). Lower is better — this is the concrete signal for "which encoding
  defeated pushdown", and the per-pair gate in `vx profile diff` catches
  regressions here.
- **Pruning was ineffective.** `pruning.pruned_ratio` = 0.0 — zone-map pruning
  skipped no row range (this SF-0.1 data is not clustered by `l_shipdate`), so the
  row filter did all the selection: `filter.selectivity` 0.18 cut 722k → 133k, and
  the query ends at `scan.rows_out` 11 618 (~2 %, the expected Q6 selectivity).
- **IO.** `io.read_amplification` 1.13 with coalescing factor 2.08 — tight. 138
  reads are tiny (< 4 KiB) metadata. `io.read_latency_p50_us` 2 µs is the common
  (cache-served) read; the 1 s p99 is an outlier the size histogram isolates from
  the bulk.
- **Kernel cross-check (cold run).** `bio.device_reads` 19 / 3.8 MB is what
  actually reached the disk past the page cache. `lock.futex_waits` 773 (p99
  33 ms) is the two concurrent splits contending shared locks. `offcpu.total_ms`
  0.73, almost all under `vortex.pco`, confirms decode is **compute-bound** — it
  barely blocks. `net.*` is zero (local file). `pmu.*` is absent: this instance
  does not expose hardware counters (they appear on a PMU-enabled host).

Numbers are workload, cache, and host dependent.

## Notes

- Use profile diff for deterministic counters.
- Do not use PMU (or the other kernel-layer latency/system-wide signals) as a hard
  regression gate.
- All eBPF layers require root because they load eBPF programs.
- `--bio` and `--net` are system-wide for the run; use an idle host.
- `--offcpu` and `--pmu` need a `profile-pmu` build (the context markers).
- If a layer says the object is stubbed, rebuild without `VORTEX_EBPF_SKIP_BPF`.
