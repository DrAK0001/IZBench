# izbench.py

Portable, dependency-free benchmark script focused on **speed-only** measurements:

- CPU single-thread + multi-process (int/float), reported as **Mops/s**
- RAM throughput **write/read** (MB/s)
- Disk throughput **write/read** (MB/s) using a temporary file
- Basic system info (OS, CPU name/cores, RAM total, disk size/model best-effort)
- Optional **countdown** based on the previous run duration for the same host/CPU/config

> Designed for quick comparisons across machines/VMs/containers without external tools.

---

## Purpose

`izbench.py` is meant for:
- quick sanity checks (“is this box slow?”)
- comparing hardware/VM flavors with the same settings
- collecting a small JSON blob for later parsing (CI, monitoring, ad-hoc inventory)

It is **not** a scientific benchmark and does not try to:
- model real workloads,
- control CPU frequency scaling,
- pin CPU affinity,
- avoid OS caches,
- measure latency distributions, etc.

---

## Requirements

- Python **3.8+** (should work on Linux/Windows/macOS/*BSD; also Termux/Android in many cases)
- No third-party Python packages.

---

## Install

Just copy the file:

```bash
git clone https://github.com/DrAK0001/IZBench.git
cd IZBench
chmod +x izbench.py
```

Or run directly:

```bash
python3 izbench.py
```

---

## Usage

### Default run

```bash
python3 izbench.py
```

### JSON only (machine-readable)

```bash
python3 izbench.py --json-only
```

### Tune CPU operations and workers

```bash
python3 izbench.py --cpu-ops 100000000 --cpu-workers 8
```

### Tune RAM test size and passes

```bash
python3 izbench.py --mem-size-mb 256 --mem-passes 6
```

### Disk test directory or disable disk

```bash
python3 izbench.py --disk-dir /var/tmp
python3 izbench.py --no-disk
```

### Save results to a file

```bash
python3 izbench.py --json-only > izbench_$(hostname)_$(date +%Y%m%d_%H%M%S).json
```

---

## CLI Options

- `--cpu-ops N`  
  Total operations per CPU test (default: `50_000_000`)

- `--cpu-workers N`  
  Number of processes for multi-process CPU test (default: CPU count)

- `--mem-size-mb N`  
  RAM buffer size (default: `64` MB)

- `--mem-passes N`  
  How many passes over the RAM buffer (default: `4`)

- `--disk-size-mb N`  
  Disk test file size (default: `128` MB)

- `--disk-dir PATH`  
  Directory for disk temp file (default: system temp dir)

- `--no-disk`  
  Disable disk test completely

- `--json-only`  
  Print only JSON (no human summary and **no countdown**)

---

## Output

The script prints:
1) a human-readable summary (unless `--json-only`)
2) a JSON document to stdout (always)

### JSON structure (high level)

```json
{
  "meta": {
    "benchmark": "...",
    "started_ts": 0,
    "finished_ts": 0,
    "duration_s": 0,
    "hostname": "...",
    "system_info": { "...": "..." },
    "params": { "...": "..." }
  },
  "cpu": {
    "single": { "int_mops": 0, "float_mops": 0 },
    "multi": { "int_mops": 0, "float_mops": 0, "workers": 0 }
  },
  "mem": { "write_mb_per_s": 0, "read_mb_per_s": 0 },
  "disk": { "write_mb_per_s": 0, "read_mb_per_s": 0 }
}
```

`disk` is omitted when `--no-disk` is used.

---

## Last-run storage & countdown

To estimate remaining time, the script stores the **total runtime** of the last run for a given config key.

- **File:** `~/.izbench_lastrun.json`
- **Stored value:** `duration_s` (seconds)
- **Key:** built from:
  - OS
  - CPU name
  - CPU cores
  - `--cpu-ops`
  - `--mem-size-mb`
  - `--mem-passes`
  - `--disk-size-mb`
  - disk enabled/disabled

On the next run with the same key, a simple countdown is printed (unless `--json-only`).

> Only the *duration* is persisted. Full benchmark results are **not** stored unless you redirect stdout.

---

## Disk test behavior

- Creates a temporary file named like `izbench_*.bin`
- Location:
  - `--disk-dir PATH` if set
  - otherwise the system temp directory (e.g. `/tmp` on Linux)
- The file is deleted after the test (best effort).

Note: results can be influenced by filesystem caches, storage write-back policy, and underlying virtualized storage.

---

## Notes / Portability

- CPU multi-thread test uses `multiprocessing.Pool` (processes, not threads).
- CPU model name/disk model are “best effort” and may show as `"unknown"` depending on OS/tools available.
- On Windows, `wmic` is used for some fields; on newer systems this may be missing or deprecated.

---

## License

- MIT


