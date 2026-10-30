"""Train on real MNIST using CPU PyTorch, then export weights and parity fixtures."""
import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

from model import DigitMLP, export_model, write_matrix

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def mnist(directory, train):
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "train" if train else "t10k"
    arrays = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve("https://ossci-datasets.s3.amazonaws.com/mnist/" + name, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f"MNIST checksum mismatch: {path}; remove it and retry")
        raw = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", raw[:8])
        if "images" in kind:
            if magic != 2051 or struct.unpack(">II", raw[8:16]) != (28, 28) or len(raw) != 16 + count * 784:
                raise ValueError("invalid MNIST images")
            arrays.append(torch.from_numpy(np.frombuffer(raw, np.uint8, offset=16).copy().reshape(count, 784)).float() / 255.0)
        else:
            if magic != 2049 or len(raw) != 8 + count:
                raise ValueError("invalid MNIST labels")
            arrays.append(torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8).copy()).long())
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or not 1 <= args.samples <= 10000:
        parser.error("epochs/threads must be positive and samples must be in [1,10000]")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_x, train_y = mnist(args.data_dir, True)
    test_x, test_y = mnist(args.data_dir, False)
    model = DigitMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = torch.randperm(len(train_x))
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            correct = sum((model(x).argmax(1) == y).sum().item()
                          for x, y in zip(test_x.split(512), test_y.split(512)))
        row = {"epoch": epoch + 1, "train_loss": total_loss / len(train_x), "test_accuracy": correct / len(test_x)}
        history.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    export_model(model, args.output / "model.bin")
    # Include zero, one, negative, and alternating inputs as numerical edge cases.
    edges = torch.stack([torch.zeros(784), torch.ones(784), -torch.ones(784), torch.arange(784).remainder(2).float()])
    inputs = torch.cat((test_x[:args.samples], edges))
    with torch.inference_mode():
        expected = model(inputs)
    write_matrix(args.output / "inputs.bin", inputs.numpy())
    write_matrix(args.output / "expected.bin", expected.numpy())
    metadata = {"architecture": [784, 64, 10], "dataset": "MNIST", "preprocessing": "float32 pixel / 255",
                "seed": args.seed, "epochs": args.epochs, "training_samples": len(train_x), "test_samples": len(test_x),
                "exported_mnist_samples": args.samples, "edge_samples": 4, "threads": args.threads,
                "torch_version": torch.__version__, "history": history,
                "mnist_md5": FILES}
    metadata["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.bin")}
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
