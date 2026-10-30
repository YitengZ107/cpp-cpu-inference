# C++ CPU inference engine

A learning project implementing a small digit classifier in C++17 with PyTorch
as the numerical reference: **784 inputs → 64 ReLU units → 10 logits**.

## Implementation stages

1. Project structure and VS Code tasks.
2. A straightforward dense-layer baseline with owned float buffers and unit tests.
3. Train a PyTorch model on MNIST and export portable weights and test inputs.
4. Compare every output logit against PyTorch with explicit error tolerances.
5. Improve memory traversal, reuse scratch buffers, and add parallel batches.
6. Record repeated benchmarks and document the hardware and software environment.

Each stage is committed separately as it is implemented and checked. Commit
timestamps reflect the actual work; this repository does not simulate work dates.

The C++ runtime will have no PyTorch dependency. Python is used for training,
export, verification, and benchmark orchestration. Raw MNIST data and local build
products stay out of Git.
