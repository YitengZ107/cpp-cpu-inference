# C++ CPU inference engine

A C++17 MNIST classifier for studying numerical correctness, memory traversal,
allocation, and concurrency.

```text
784 float32 pixels → Dense(784,64) → ReLU → Dense(64,10) → 10 logits
```

The C++ executable uses the standard library and contiguous `std::vector<float>`
buffers. It has no PyTorch, Python, or BLAS dependency.

- **96.51% MNIST test accuracy** after five CPU training epochs, using 60,000
  training images, 10,000 test images, and seed 42.
- All C++ modes agree with PyTorch on every prediction across **1,028 inputs /
  10,280 logits**. Maximum absolute logit error: **1.7166e-5**.
- Optimized C++ improves on the project's baseline. **PyTorch wins the larger
  batches on the measured machine.** Parallel C++ is slower than serial here.

See [measured results](results/REPORT.md), [raw timings](results/benchmark.json),
[CSV](results/benchmark.csv), and [training provenance](examples/training.json).

## Use VS Code

The configured workstation already has its compiler, Python environment, and
local CMake preset selected. Machine-specific paths are ignored by Git.

1. Open this folder in VS Code; select the project's Python interpreter.
2. Press **Ctrl+Shift+B** to build.
3. Use **Terminal → Run Task → Test C++**, then **Verify PyTorch parity**.
4. **Benchmark CPU** runs repeated measurements. **Train and export MNIST** trains
   a fresh reference and downloads MNIST if needed.

The recommended extensions are C++, Python, and CMake Tools.

## Build a fresh checkout

Use a modern C++17 compiler and Python 3.12–3.14. `requirements.txt` includes
CMake and Ninja. Windows users can put a recent portable
[w64devkit](https://github.com/skeeto/w64devkit) `bin` folder on PATH, or use a
Visual Studio Developer PowerShell with C++ build tools installed. Bypass any
broken or outdated MinGW installation earlier in PATH.

Windows:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/dev.py test
.\build\inference_cli.exe infer examples/model.bin examples/inputs.bin build/scores.bin optimized 4
.\.venv\Scripts\python.exe python/verify.py --exe build/inference_cli.exe --artifacts examples --threads 4
```

Linux with GCC or Clang:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/dev.py test
./build/inference_cli infer examples/model.bin examples/inputs.bin build/scores.bin optimized 4
python python/verify.py --exe build/inference_cli --artifacts examples --threads 4
```

The pinned CPU PyTorch dependencies target Windows/Linux. On macOS, install
compatible `torch` and `numpy` builds from PyPI instead; macOS is not in CI.

For just the C++ runtime, Python packages are unnecessary:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The helper uses Ninja and Release. Visual Studio's multi-configuration generator
puts executables under `build/Release/`; the VS Code tasks use the Ninja layout.

## Train, export, and verify

The checked-in `examples/` folder contains trained weights, **32 MNIST images and
four numerical edge inputs**. This demo needs no dataset download. Preprocess
pixels as `float32(pixel) / 255` in row-major image order. Outputs are raw logits;
the predicted digit is `argmax`, with no softmax required.

```sh
python python/train_export.py --epochs 5 --threads 2 --seed 42 --samples 1024
python python/verify.py --exe build/inference_cli --mode baseline --output results/correctness_baseline.json
python python/verify.py --exe build/inference_cli --threads 1 --output results/correctness_optimized.json
python python/verify.py --exe build/inference_cli --threads 4 --output results/correctness_parallel.json
python python/export_demo.py
```

On Windows, append `.exe` to the C++ executable path. Training writes to ignored
`artifacts/`: a PyTorch state dictionary, portable weights, inputs, expected scores,
and metadata/checksums. The complete fixture is the first 1,024 MNIST test images
plus zero, one, negative-one, and alternating inputs. Downloaded MNIST archives
are checksum-verified. Raw dataset files stay out of Git.

Verification checks every exported score against a live PyTorch reference, then
every C++ score against that reference. It reports max/mean absolute error,
relative error with a denominator floor, and prediction agreement. A score passes
when `abs(cpp-ref) <= atol + rtol*abs(ref)`, using `atol=rtol=1e-4`. This is not a
claim that maximum relative error alone is below `rtol`.

## Memory access and ownership

Both C++ paths share weights stored as `[input][output]` float32 buffers.

| Path | Dense traversal | Allocation and scheduling |
| --- | --- | --- |
| Baseline | Output outer loop; strided weight access | Fresh output each call, hidden buffer each sample |
| Optimized serial | Input outer loop; contiguous independent output lanes | Reuses hidden/output capacity |
| Optimized parallel | Same kernel over disjoint row ranges | Persistent workers with separate scratch; caller participates |

The inner loop permits compiler auto-vectorization. There are no handwritten SIMD
instructions or fast-math flags. This compares against a straightforward baseline,
not an optimized GEMM library. There are 50,890 parameters, about 199 KiB.

`Model` owns validated layer buffers. `Engine` owns its model copy, scratch, output,
and worker lifetime. `run()` returns a reference valid until the next run or Engine
destruction. Copy scores if they must outlive the next call. Use one caller at a
time per Engine; separate Engines may run independently. Same-size steady-state
calls reuse storage; a larger batch can allocate. Empty batches return no scores.

Parallel row partitions preserve each dot product's accumulation order. The API
rejects invalid dimensions, non-finite inputs/parameters, and input aliasing its
own output. File IO checks headers, exact lengths, and size limits. See the
[binary format](docs/FORMAT.md).

## Reproduce performance measurements

```sh
python python/benchmark.py --exe build/inference_bench --artifacts artifacts \
  --batches 1 32 128 1024 --threads 1 2 4 --warmup 10 --repeats 9 --target-ms 100
```

Use one line in PowerShell and append `.exe`. Omit `--artifacts artifacts` to use
the smaller checked-in demo, tiled as needed; that is a different input fixture.

Each configuration receives ten explicit warm-ups, calibration, and nine timed
blocks targeting at least 100 ms each. C++ uses `steady_clock`; PyTorch uses
`torch.utils.benchmark.Timer` under `inference_mode`, including additional internal
warm-up/calibration. PyTorch intra-op threads are explicit and inter-op threads
are one. Configuration order is shuffled with seed 42. JSON preserves all raw
timings, median, 25th/75th percentiles, throughput, compiler flags, CPU, package
versions, source revision, and model/input/executable hashes.

Measurements cover hot in-process inference. Loading, file IO, process startup,
and worker construction are excluded. C++ validation, output handling, baseline
allocations, and parallel dispatch are included. PyTorch eager operator overhead
and output allocations are included. These are each implementation's public API
costs. Affinity, power mode, thermals, and desktop activity were not controlled.

Native CPU tuning is off by default. Test `-DINFERENCE_NATIVE=ON` separately and
save another results file. The benchmark rejects Debug/unspecified configurations.

## Tests and implementation stages

CTest runs C++ unit tests and Python standard-library CLI tests. Coverage includes
hand-computed results, negative/zero activations, randomized shapes/vector tails,
dimension mismatches, overflow, non-finite values, corrupt/truncated/trailing
binary data, repeated parallel jobs, uneven partitions, and batches smaller than
the worker count. Test checks stay active in Release.

CI builds Windows/MSVC and Linux/GCC, checks baseline/serial/parallel scores
against the checked-in PyTorch fixture, and runs Linux address/undefined-behavior
sanitizers. Git history separates scaffold, baseline, PyTorch reference,
correctness, memory optimization, parallel workers, benchmarks/CI, and evidence.
Use `git log --reverse --oneline` to inspect the stages.

```text
include/inference/    Public model, runtime, and IO interfaces
src/                 Baseline, optimized runtime, CLI, benchmark driver
python/              Training/export, verification, benchmark orchestration
tests/               C++ unit tests and malformed-file CLI tests
examples/            Runnable reference fixture and training provenance
results/             Numerical errors and raw benchmark evidence
scripts/dev.py       VS Code / command-line build and test helper
```

## Sources and next experiments

- [PyTorch benchmarking guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
  informs warm-up, repeated measurements, and thread control.
- [Torchvision's MNIST loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
  documents the dataset mirror and checksums.

Further experiments could test blocked batch matrix multiplication, kernel-only
timing, profiling the parallel path, and controlled affinity/power settings.
The current parallel implementation demonstrates correct concurrency; its
measured slowdown is included in the report.
