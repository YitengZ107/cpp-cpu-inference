"""Record calibrated repeated C++ and CPU PyTorch measurements on identical data."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.benchmark import Timer

from model import DigitMLP, read_matrix, write_matrix


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True, help="inference_bench executable")
    parser.add_argument("--artifacts", type=Path, default=Path("examples"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 1024])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--target-ms", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if any(b < 1 for b in args.batches) or any(t < 1 or t > 256 for t in args.threads) or args.repeats < 3 or args.warmup < 1 or not 1 <= args.target_ms <= 10000:
        parser.error("invalid positive batch/thread/warmup/repetition/target settings")
    # Scope provenance to implementation files so output files cannot mark their own run dirty.
    source_paths = ["CMakeLists.txt", "CMakePresets.json", "requirements.txt", "include", "src", "cmake", "python", "scripts", "tests", ".github"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *source_paths], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    torch.set_num_interop_threads(1)
    model = DigitMLP().eval()
    model.load_state_dict(torch.load(args.artifacts / "model.pt", map_location="cpu", weights_only=True))
    source = read_matrix(args.artifacts / "inputs.bin")
    if source.shape[0] == 0:
        parser.error("input fixture is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(mode, batch, threads) for batch in dict.fromkeys(args.batches)
             for mode in ("baseline", "optimized", "pytorch")
             for threads in ([1] if mode == "baseline" else dict.fromkeys(args.threads))]
    random.Random(42).shuffle(tasks) # Reduce systematic bias from running all of one mode first.
    rows = []
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        for mode, batch, threads in tasks:
            values = np.tile(source, (math.ceil(batch / len(source)), 1))[:batch].copy()
            if mode == "pytorch":
                torch.set_num_threads(threads)
                tensor = torch.from_numpy(values)
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        model(tensor)
                    timer = Timer(stmt="model(inputs)", globals={"model": model, "inputs": tensor}, num_threads=threads)
                    estimate = timer.blocked_autorange(min_run_time=0.1)
                    iterations = max(1, math.ceil(args.target_ms / 1000 / estimate.median))
                    samples = [timer.timeit(number=iterations).median for _ in range(args.repeats)]
                    checksum = float(model(tensor).double().sum().item()) * args.repeats
                result = {"mode": mode, "batch": batch, "threads": threads, "iterations": iterations,
                          "warmup": args.warmup, "seconds_per_call": samples, "checksum": checksum}
            else:
                fixture = Path(temporary) / "batch.bin"
                write_matrix(fixture, values)
                result = json.loads(subprocess.check_output([str(args.exe.resolve()), str(args.artifacts / "model.bin"),
                    str(fixture), mode, str(threads), str(args.warmup), str(args.repeats), str(args.target_ms)], text=True))
            samples = result["seconds_per_call"]
            result.update(median_us=float(np.median(samples) * 1e6),
                          p25_us=float(np.quantile(samples, 0.25) * 1e6),
                          p75_us=float(np.quantile(samples, 0.75) * 1e6),
                          samples_per_second=float(batch / np.median(samples)))
            rows.append(result)
            print(f"{mode:9} batch={batch:4} threads={threads}: {result['median_us']:.2f} us", flush=True)
    rows.sort(key=lambda row: (row["batch"], row["mode"], row["threads"]))
    for row in rows:
        baseline = next(r for r in rows if r["batch"] == row["batch"] and r["mode"] == "baseline")
        row["speedup_vs_baseline"] = baseline["median_us"] / row["median_us"]
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(), "os": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
        "torch_build": torch.__config__.show(), "interop_threads": 1,
        "source_commit": commit, "source_dirty": dirty,
        "source_paths_checked": source_paths,
        "model_sha256": hashlib.sha256((args.artifacts / "model.bin").read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256((args.artifacts / "inputs.bin").read_bytes()).hexdigest(),
        "executable_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        "warmup_calls": args.warmup, "repeats": args.repeats, "target_ms_per_repeat": args.target_ms,
        "order_seed": 42, "affinity": "not pinned", "power_mode": "not controlled",
        "scope": "hot in-process CPU inference; excludes IO, process launch, model loading, and pool creation; includes C++ validation and dispatch"}
    args.output.write_text(json.dumps({"environment": metadata, "results": rows}, indent=2) + "\n")
    fields = ["mode", "batch", "threads", "median_us", "p25_us", "p75_us", "samples_per_second", "speedup_vs_baseline", "iterations", "warmup"]
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
