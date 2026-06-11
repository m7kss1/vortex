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
