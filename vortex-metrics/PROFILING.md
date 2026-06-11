# Vortex Profiling

## TLDR

Base profile, no root:

```bash
cargo build -p vortex-tui --features profile --bin vx
vx profile query ./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex \
  --sql "select count(*) from data" \
  --json /tmp/vortex-profile.json
```

With eBPF read syscall stats:

```bash
cargo build -p vortex-tui --features profile-ebpf --bin vx
sudo -E target/debug/vx profile query ./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex \
  --sql "select count(*) from data" \
  --syscalls \
  --json /tmp/vortex-profile-ebpf.json
```

Compare two reports:

```bash
vx profile diff /tmp/before.json /tmp/after.json
```

PMU is not the default path right now. Use `--syscalls` first.

## What It Measures

```text
                 no root                         root
Vortex code  ---------------->  vortex-metrics  --------> JSON report
 rows, bytes, decode time       deterministic
 pruning, fallbacks             counters

Linux kernel ---------------->  vortex-ebpf    --------> same JSON report
 read/pread syscalls            optional
 read size histogram            --syscalls
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

eBPF syscall metrics are optional:

- `io.read_syscalls`
- `io.read_syscall_bytes`
- `io.read_size_p50_bytes`
- `io.read_size_p99_bytes`
- `io.tiny_reads`
- `io.tiny_read_threshold_bytes`
- `io.read_size_hist.*`

The report also has a `notes` object next to `metrics`. It holds short text
that explains a number when the number alone is not clear (for now only the
pushdown fallback). `notes` is text, not numbers, so the diff gate ignores it.

## Metrics Reference

All keys below live inside the `metrics` object of the JSON report. The `Source`
column says where the number comes from:

- `in-process` — always present, no root, no eBPF.
- `--syscalls` — present only with the eBPF read-syscall layer (needs root and a
  `profile-ebpf` build). Missing in the base profile.
- `--pmu` — present only with the eBPF PMU layer (needs root and a `profile-pmu`
  build). Missing otherwise.

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

`io.read_size_hist.*` is the full size breakdown. Use it to find where the small
reads sit. Example: a high `io.read_size_hist.32` means many reads in `[32, 64)`
bytes — metadata, not data.

### PMU (eBPF only)

These keys exist **only** with `--pmu` (`profile-pmu` build, root). They are
samples, not exact counts. Do not use them as a hard regression gate.

| Key | Source | Meaning |
| --- | --- | --- |
| `pmu.<enc>.cycle_samples` | `--pmu` | CPU-cycle samples attributed to this encoding's decode |
| `pmu.<enc>.instruction_samples` | `--pmu` | retired-instruction samples |
| `pmu.<enc>.cache_miss_samples` | `--pmu` | cache-miss samples |

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

For `--syscalls`:

- Linux
- root or enough `CAP_BPF` / `CAP_PERFMON`
- `bpf-linker` installed:

```bash
cargo install bpf-linker
```

For CI/check without BPF toolchain:

```bash
VORTEX_EBPF_SKIP_BPF=1 cargo check -p vortex-tui --features profile-ebpf --no-default-features
```

This builds a stub object. It is only for compile check. It cannot collect
syscall stats.

## Real Workload Example

```bash
FILE=./vortex-bench/data/tpch/0.1/vortex-compact/lineitem_0.vortex
SQL="select count(*) from data where l_quantity > 20"

cargo build -p vortex-tui --features profile-ebpf --bin vx

sudo -E target/debug/vx profile query "$FILE" \
  --sql "$SQL" \
  --syscalls \
  --json /tmp/lineitem-syscalls.json
```

Inspect:

```bash
jq ".metrics | {
  rows: .[\"scan.rows_out\"],
  reads: .[\"io.read_syscalls\"],
  read_bytes: .[\"io.read_syscall_bytes\"],
  p50: .[\"io.read_size_p50_bytes\"],
  p99: .[\"io.read_size_p99_bytes\"],
  tiny: .[\"io.tiny_reads\"]
}" /tmp/lineitem-syscalls.json
```

Expected shape:

```json
{
  "rows": 12345,
  "reads": 42,
  "read_bytes": 8388608,
  "p50": 65536,
  "p99": 1048576,
  "tiny": 3
}
```

Numbers are workload and cache dependent.

## Notes

- Use profile diff for deterministic counters.
- Do not use PMU as a hard regression gate.
- `--syscalls` requires root because it loads eBPF programs.
- If `--syscalls` says the object is stubbed, rebuild without `VORTEX_EBPF_SKIP_BPF`.
