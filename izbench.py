
#!/usr/bin/env python3
"""
izbench.py - simple portable benchmark:
- CPU single-thread and multi-thread (int/float, Mops/s)
- RAM read/write throughput (MB/s)
- Disk throughput (MB/s)
- basic system info (CPU, RAM, disk, OS)
- optional countdown based on last run for this host/CPU/config
"""

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
import multiprocessing as mp
import threading


LASTRUN_FILE = Path.home() / ".izbench_lastrun.json"


def now():
    return time.perf_counter()


# ---------------------- LAST RUN STORAGE ---------------------- #

def load_last_runs():
    try:
        if LASTRUN_FILE.is_file():
            with open(LASTRUN_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_last_runs(data):
    try:
        with open(LASTRUN_FILE, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception:
        pass


def make_last_key(sysinfo, args):
    # key identifies host + CPU + benchmark parameters
    return "|".join(
        [
            str(sysinfo.get("os", "")),
            str(sysinfo.get("cpu_name", "")),
            str(sysinfo.get("cpu_cores", "")),
            str(args.cpu_ops),
            str(args.mem_size_mb),
            str(args.mem_passes),
            str(args.disk_size_mb),
            "disk_on" if not args.no_disk else "disk_off",
        ]
    )


def start_countdown(predicted_seconds):
    if not predicted_seconds or predicted_seconds <= 0:
        return None, None

    stop_event = threading.Event()

    def worker():
        start_t = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_t
            remaining = predicted_seconds - elapsed
            if remaining < 0:
                remaining = 0
            # crude single-line countdown
            print(
                "\rEstimated time left: ~{:4d} s".format(int(remaining)),
                end="",
                flush=True,
            )
            if remaining <= 0:
                break
            time.sleep(1.0)
        # clear line
        print("\r" + " " * 40 + "\r", end="", flush=True)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return stop_event, t


# ---------------------- SYSTEM INFO ---------------------- #

def get_system_info():
    os_name = platform.system()
    release = platform.release()
    machine = platform.machine()

    info = {
        "os": os_name,
        "release": release,
        "machine": machine,
        "cpu_name": "unknown",
        "cpu_cores": os.cpu_count() or 1,
        "memory_total_mb": None,
        "disk_model": "unknown",
        "disk_size_gb": None,
    }

    # CPU name
    try:
        if os_name == "Linux":
            # works also on Android/Termux
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if (
                        "model name" in line
                        or "Hardware" in line
                        or "Processor" in line
                    ):
                        info["cpu_name"] = line.split(":", 1)[1].strip()
                        break
        elif os_name == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) >= 2:
                info["cpu_name"] = lines[1]
        elif os_name == "Darwin":
            info["cpu_name"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        elif "BSD" in os_name:
            info["cpu_name"] = subprocess.check_output(
                ["sysctl", "-n", "hw.model"],
                text=True,
            ).strip()
    except Exception:
        pass

    # Memory total (MB)
    try:
        if os_name == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kB = int(re.findall(r"\d+", line)[0])
                        info["memory_total_mb"] = kB // 1024
                        break
        elif os_name == "Windows":
            out = subprocess.check_output(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            nums = re.findall(r"\d+", out)
            if nums:
                bytes_ = int(nums[0])
                info["memory_total_mb"] = bytes_ // (1024 * 1024)
        elif os_name == "Darwin":
            bytes_ = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                ).strip()
            )
            info["memory_total_mb"] = bytes_ // (1024 * 1024)
        elif "BSD" in os_name:
            bytes_ = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physmem"],
                    text=True,
                ).strip()
            )
            info["memory_total_mb"] = bytes_ // (1024 * 1024)
    except Exception:
        pass

    # Disk size (fallback) from filesystem
    try:
        stat = shutil.disk_usage("/")
        info["disk_size_gb"] = round(stat.total / (1024 ** 3), 1)
    except Exception:
        pass

    # Disk model (best effort)
    try:
        if os_name == "Linux":
            try:
                out = subprocess.check_output(
                    ["lsblk", "-ndo", "MODEL,SIZE"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                if lines:
                    parts = lines[0].split()
                    if len(parts) >= 2:
                        model = " ".join(parts[:-1])
                        info["disk_model"] = model
            except Exception:
                pass

        elif os_name == "Windows":
            out = subprocess.check_output(
                ["wmic", "diskdrive", "get", "Model,Size"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) >= 2:
                line = lines[1]
                m = re.search(r"(\d+)", line)
                if m:
                    size_bytes = int(m.group(1))
                    info["disk_size_gb"] = size_bytes // (1024 ** 3)
                    model = line[: m.start()].strip()
                    if model:
                        info["disk_model"] = model

        elif os_name == "Darwin":
            # disk model is not critical here, filesystem size is enough
            pass

        elif "BSD" in os_name:
            pass
    except Exception:
        pass

    return info


# ---------------------- CPU single-thread ---------------------- #

def run_cpu_int_single(ops: int):
    acc = 0
    a = 1664525
    c = 1013904223
    mask = 0xFFFFFFFF

    start = now()
    for _ in range(ops):
        acc = (acc * a + c) & mask
    elapsed = now() - start

    mops = ops / elapsed / 1e6 if elapsed > 0 else 0.0
    return {
        "mops": mops,
        "seconds": elapsed,
    }


def run_cpu_float_single(ops: int):
    x = 1.0
    mul = 1.0000001
    add = 1.0

    start = now()
    for _ in range(ops):
        x = x * mul + add
        if x > 1e20:
            x *= 1e-20
    elapsed = now() - start

    mops = ops / elapsed / 1e6 if elapsed > 0 else 0.0
    return {
        "mops": mops,
        "seconds": elapsed,
    }


# ---------------------- CPU multi-thread (multiprocessing) ---------------------- #

def _cpu_int_worker(ops: int):
    acc = 0
    a = 1664525
    c = 1013904223
    mask = 0xFFFFFFFF
    for _ in range(ops):
        acc = (acc * a + c) & mask
    return acc


def _cpu_float_worker(ops: int):
    x = 1.0
    mul = 1.0000001
    add = 1.0
    for _ in range(ops):
        x = x * mul + add
        if x > 1e20:
            x *= 1e-20
    return x


def run_cpu_int_multi(total_ops: int, workers: int):
    if workers < 1:
        workers = 1
    ops_per_worker = total_ops // workers
    if ops_per_worker < 1:
        ops_per_worker = 1
        workers = min(total_ops, workers)

    total_ops_effective = ops_per_worker * workers

    start = now()
    with mp.Pool(processes=workers) as pool:
        pool.map(_cpu_int_worker, [ops_per_worker] * workers)
    elapsed = now() - start

    mops = total_ops_effective / elapsed / 1e6 if elapsed > 0 else 0.0
    return {
        "mops": mops,
        "seconds": elapsed,
        "workers": workers,
        "total_ops": total_ops_effective,
    }


def run_cpu_float_multi(total_ops: int, workers: int):
    if workers < 1:
        workers = 1
    ops_per_worker = total_ops // workers
    if ops_per_worker < 1:
        ops_per_worker = 1
        workers = min(total_ops, workers)

    total_ops_effective = ops_per_worker * workers

    start = now()
    with mp.Pool(processes=workers) as pool:
        pool.map(_cpu_float_worker, [ops_per_worker] * workers)
    elapsed = now() - start

    mops = total_ops_effective / elapsed / 1e6 if elapsed > 0 else 0.0
    return {
        "mops": mops,
        "seconds": elapsed,
        "workers": workers,
        "total_ops": total_ops_effective,
    }


# ---------------------- RAM (separate read/write) ---------------------- #

def run_mem(size_mb: int, passes: int):
    size_bytes = size_mb * 1024 * 1024
    buf = bytearray(size_bytes)
    total_bytes = size_bytes * passes

    # write test
    start = now()
    for _ in range(passes):
        # stride 64 bytes - easy to port to C
        for i in range(0, size_bytes, 64):
            buf[i] = 0xAA
    write_elapsed = now() - start
    write_mb_per_s = (
        total_bytes / write_elapsed / (1024 * 1024) if write_elapsed > 0 else 0.0
    )

    # read test
    start = now()
    acc = 0
    for _ in range(passes):
        for i in range(0, size_bytes, 64):
            acc += buf[i]
    read_elapsed = now() - start
    read_mb_per_s = (
        total_bytes / read_elapsed / (1024 * 1024) if read_elapsed > 0 else 0.0
    )

    return {
        "write_mb_per_s": write_mb_per_s,
        "read_mb_per_s": read_mb_per_s,
        "write_seconds": write_elapsed,
        "read_seconds": read_elapsed,
    }


# ---------------------- Disk ---------------------- #

def run_disk(size_mb: int, directory=None):
    if directory is None:
        directory = tempfile.gettempdir()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    size_bytes = size_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    chunks = size_bytes // chunk_size
    data = b"\xAA" * chunk_size

    fd, path_str = tempfile.mkstemp(prefix="izbench_", suffix=".bin", dir=directory)
    os.close(fd)
    path = Path(path_str)

    write_seconds = None
    read_seconds = None

    try:
        # write
        start = now()
        with open(path, "wb", buffering=0) as f:
            for _ in range(chunks):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        write_seconds = now() - start

        # read
        start = now()
        total_read = 0
        with open(path, "rb", buffering=0) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
        read_seconds = now() - start
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    write_mb_per_s = size_mb / write_seconds if write_seconds and write_seconds > 0 else 0.0
    read_mb_per_s = size_mb / read_seconds if read_seconds and read_seconds > 0 else 0.0

    return {
        "write_mb_per_s": write_mb_per_s,
        "read_mb_per_s": read_mb_per_s,
        "write_seconds": write_seconds,
        "read_seconds": read_seconds,
    }


# ---------------------- CLI ---------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        description="Simple portable benchmark (CPU single/multi, RAM R/W, disk) - speed only."
    )
    p.add_argument(
        "--cpu-ops",
        type=int,
        default=50_000_000,
        help="Total number of operations for CPU tests (default: 50M).",
    )
    p.add_argument(
        "--cpu-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of processes for multi-thread CPU test (default: CPU count).",
    )
    p.add_argument(
        "--mem-size-mb",
        type=int,
        default=64,
        help="Memory block size for RAM test (default: 64 MB).",
    )
    p.add_argument(
        "--mem-passes",
        type=int,
        default=4,
        help="Number of passes over RAM block (default: 4).",
    )
    p.add_argument(
        "--disk-size-mb",
        type=int,
        default=128,
        help="File size for disk test (default: 128 MB).",
    )
    p.add_argument(
        "--disk-dir",
        type=str,
        default=None,
        help="Directory for disk test (default: system temp).",
    )
    p.add_argument(
        "--no-disk",
        action="store_true",
        help="Disable disk test.",
    )
    p.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON (no human-readable summary and no countdown).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    sysinfo = get_system_info()

    # last-run based countdown
    last_runs = load_last_runs()
    key = make_last_key(sysinfo, args)
    predicted = last_runs.get(key)
    stop_event = None
    countdown_thread = None

    if (predicted is not None) and (not args.json_only):
        stop_event, countdown_thread = start_countdown(predicted)

    started = time.time()

    # CPU single-thread
    cpu_int_single = run_cpu_int_single(args.cpu_ops)
    cpu_float_single = run_cpu_float_single(args.cpu_ops)

    # CPU multi-thread
    cpu_int_multi = run_cpu_int_multi(args.cpu_ops, args.cpu_workers)
    cpu_float_multi = run_cpu_float_multi(args.cpu_ops, args.cpu_workers)

    # RAM
    mem = run_mem(args.mem_size_mb, args.mem_passes)

    # Disk
    disk = None
    if not args.no_disk:
        disk = run_disk(args.disk_size_mb, args.disk_dir)

    finished = time.time()
    total_duration = finished - started

    # stop countdown
    if stop_event is not None:
        stop_event.set()
    if countdown_thread is not None:
        countdown_thread.join()

    # store last run for this config
    last_runs[key] = total_duration
    save_last_runs(last_runs)

    results = {
        "meta": {
            "benchmark": "izbench_python_ascii_rw_countdown",
            "started_ts": started,
            "finished_ts": finished,
            "duration_s": total_duration,
            "hostname": socket.gethostname(),
            "system_info": sysinfo,
            "params": {
                "cpu_ops": args.cpu_ops,
                "cpu_workers": args.cpu_workers,
                "mem_size_mb": args.mem_size_mb,
                "mem_passes": args.mem_passes,
                "disk_size_mb": args.disk_size_mb,
                "disk_dir": args.disk_dir,
                "disk_enabled": not args.no_disk,
            },
        },
        "cpu": {
            "single": {
                "int_mops": cpu_int_single["mops"],
                "float_mops": cpu_float_single["mops"],
            },
            "multi": {
                "int_mops": cpu_int_multi["mops"],
                "float_mops": cpu_float_multi["mops"],
                "workers": cpu_int_multi["workers"],
            },
        },
        "mem": {
            "write_mb_per_s": mem["write_mb_per_s"],
            "read_mb_per_s": mem["read_mb_per_s"],
        },
    }

    if disk is not None:
        results["disk"] = {
            "write_mb_per_s": disk["write_mb_per_s"],
            "read_mb_per_s": disk["read_mb_per_s"],
        }

    if not args.json_only:
        print("== izbench (CPU single/multi, RAM R/W, disk) ==")
        print(f"Host      : {results['meta']['hostname']}")
        print(
            f"OS        : {sysinfo['os']} {sysinfo['release']} ({sysinfo['machine']})"
        )
        print(
            f"CPU       : {sysinfo['cpu_name']} ({sysinfo['cpu_cores']} cores)"
        )
        print(f"RAM total : {sysinfo['memory_total_mb']} MB")
        print(f"Disk      : {sysinfo['disk_model']} ({sysinfo['disk_size_gb']} GB)")
        print()
        print(f"CPU ops   : {args.cpu_ops:,}")
        print(f"Workers   : {results['cpu']['multi']['workers']}")
        print(
            f"CPU int 1T: {results['cpu']['single']['int_mops']:.2f} Mops/s"
        )
        print(
            f"CPU flt 1T: {results['cpu']['single']['float_mops']:.2f} Mops/s"
        )
        print(
            f"CPU int MT: {results['cpu']['multi']['int_mops']:.2f} Mops/s"
        )
        print(
            f"CPU flt MT: {results['cpu']['multi']['float_mops']:.2f} Mops/s"
        )
        print(
            f"RAM W     : {results['mem']['write_mb_per_s']:.2f} MB/s"
        )
        print(
            f"RAM R     : {results['mem']['read_mb_per_s']:.2f} MB/s"
        )
        if "disk" in results:
            print(
                f"Disk W    : {results['disk']['write_mb_per_s']:.2f} MB/s"
            )
            print(
                f"Disk R    : {results['disk']['read_mb_per_s']:.2f} MB/s"
            )
        print()
        print(f"Total time: {total_duration:.2f} s")
        if predicted is not None:
            print(f"Prev run  : {predicted:.2f} s (same config key)")

    print(json.dumps(results, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
