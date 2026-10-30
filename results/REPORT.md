# Measured results

## Environment and provenance

- CPU: **11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz**, 8 logical processors (4 physical cores reported by Windows CIM).
- OS: `Windows-11-10.0.26200-SP0`; architecture `AMD64`.
- C++: **GNU 16.2.0**, w64devkit 2.11.0, C++17 Release.
- Library flags: `-O3 -DNDEBUG -Wall -Wextra -Wpedantic -ffp-contract=off`. Native CPU tuning and fast-math are off.
- Python 3.14.4; CPU PyTorch 2.15.0+cpu; NumPy 2.5.4.
- PyTorch uses its packaged MKL/oneDNN/OpenMP backend; its full build configuration is in the JSON.
- Recorded at `2026-10-30T20:03:36.960016+00:00`.
- Source: [`3d568ca`](https://github.com/YitengZ107/cpp-cpu-inference/commit/3d568ca6bd6a22c068569c9edb9d85ede536f85c); checked implementation paths were clean at measurement start.
- Model/input/executable SHA-256 and the checked source paths are recorded in [benchmark.json](benchmark.json).

This was a normal Windows desktop session. CPU affinity, power mode, thermals,
and competing desktop activity were not controlled. Results describe this
machine and run; they are not a universal performance claim.

## Numerical correctness

Five CPU training epochs on all 60,000 MNIST training images reached **96.51%**
accuracy on the full 10,000-image test set. Seed: 42; Adam learning rate: 0.001;
training batch: 128; training threads: 2; pixels scaled by 1/255.

Baseline, optimized serial, and optimized four-thread C++ each passed the same
1,028-input / 10,280-logit fixture (1,024 MNIST inputs plus four numerical edges):

- Maximum absolute error: **1.71661376953125e-5**.
- Mean absolute error: **7.807559801214864e-7**.
- Maximum relative error with denominator floor 1e-6: **5.504442871746481e-4**.
- Prediction agreement with PyTorch: **100%**.
- Passing rule: `abs(cpp-ref) <= 1e-4 + 1e-4*abs(ref)` for every logit.

The larger relative error occurs near small reference scores; the combined
absolute/relative criterion passes. Small float32 rounding differences are
expected from different dot-product accumulation implementations.

Raw error reports: [baseline](correctness_baseline.json),
[optimized serial](correctness_optimized.json), [four threads](correctness_parallel.json).
The [36-input checked-in demo](correctness_demo.json) also passes.

## Serial comparison

All numbers below are **median microseconds per whole batch**, not per image.
Each configuration has 9 measurements after 10 explicit warm-ups and calibration.
Each timed block targets at least 100 ms. Percentiles and every individual timing
are in the [CSV](benchmark.csv) and [JSON](benchmark.json).

| Batch | C++ baseline, 1 thread | C++ optimized, 1 thread | PyTorch, 1 thread | C++ speedup over baseline |
| --- | --- | --- | --- | --- |
| 1 | 62.76 | 8.41 | 31.96 | 7.46× |
| 32 | 2,139.89 | 323.03 | 77.11 | 6.62× |
| 128 | 7,467.96 | 1,117.15 | 227.07 | 6.68× |
| 1024 | 77,528.70 | 9,277.12 | 1,318.46 | 8.36× |

The contiguous weight traversal and buffer reuse improve this C++ baseline at
every measured batch size. Serial C++ wins the single-image comparison in this
run; PyTorch is faster for batches 32, 128, and 1,024. PyTorch uses optimized
matrix operations while the C++ implementation processes samples independently.
That is an architectural explanation to investigate further, not a profiling result.

## Concurrency

| Batch | Optimized C++, 1 thread | Optimized C++, 2 threads | Optimized C++, 4 threads |
| --- | --- | --- | --- |
| 1 | 8.41 | 13.37 | 54.69 |
| 32 | 323.03 | 391.75 | 488.94 |
| 128 | 1,117.15 | 1,907.73 | 1,633.38 |
| 1024 | 9,277.12 | 13,162.25 | 12,140.96 |

The worker pool demonstrates correct concurrent batch processing but does **not**
improve measured throughput on this setup. Dispatch, scheduling, memory access,
and desktop load are possible contributors; their individual costs were not
profiled. One thread is the measured default choice for this implementation.

| Batch | PyTorch, 1 thread | PyTorch, 2 threads | PyTorch, 4 threads |
| --- | --- | --- | --- |
| 1 | 31.96 | 34.08 | 59.74 |
| 32 | 77.11 | 67.43 | 77.82 |
| 128 | 227.07 | 161.81 | 159.89 |
| 1024 | 1,318.46 | 1,184.68 | 898.70 |

More threads are not uniformly better, including for PyTorch. All tested counts
are retained rather than selecting only the fastest configuration.

## Measurement boundaries

Both paths use identical float32 weights and the same input rows for each batch.
The recorded run uses the complete local `artifacts/` fixture; batches up to 1,024
draw MNIST rows without tiling. Model loading, file IO, process startup, input
preparation, and worker creation are outside the timed region.

C++ timings include dimension/finite-value checks, output writes, baseline
allocations, and optimized pool dispatch. Optimized output/scratch capacity is
reused after warm-up. PyTorch is eager CPU inference under `inference_mode`, with
operator dispatch and output allocations timed. PyTorch Timer performs additional
internal warm-up/calibration. Its intra-op count is explicit and inter-op count
is 1. C++ uses `steady_clock` and calibrated blocks. Checksums are consumed outside
the timed blocks. Configuration order uses shuffle seed 42.

These compare the implementations' inference API costs. They do not measure
kernel-only latency, loading latency, cold-cache inference, or training speed.

## Reproduce

From the repository root, use the selected Python environment:

```sh
python scripts/dev.py test
python python/train_export.py --epochs 5 --threads 2 --seed 42 --samples 1024
python python/verify.py --exe build/inference_cli --threads 4
python python/benchmark.py --exe build/inference_bench --artifacts artifacts --batches 1 32 128 1024 --threads 1 2 4 --warmup 10 --repeats 9 --target-ms 100
```

On Windows, append `.exe` to the C++ paths. Full setup and VS Code tasks are in
the [README](../README.md). Re-training may vary across PyTorch versions, CPU
backends, and platforms; the checked-in model and SHA-256 values identify this run.
The benchmark rejects Debug and unspecified build configurations.

CI checks Windows/MSVC, Linux/GCC, PyTorch parity, and Linux address/undefined-
behavior sanitizers. See [actual GitHub Actions runs](https://github.com/YitengZ107/cpp-cpu-inference/actions).
