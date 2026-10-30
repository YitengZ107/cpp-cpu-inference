"""Package a small, reproducible demo subset of trained artifacts for Git."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from model import read_matrix, write_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("examples"))
    args = parser.parse_args()
    metadata = json.loads((args.source / "training.json").read_text())
    count = min(32, metadata["exported_mnist_samples"])
    edge_start = metadata["exported_mnist_samples"]
    indices = list(range(count)) + list(range(edge_start, edge_start + metadata["edge_samples"]))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("model.bin", "model.pt"):
        shutil.copyfile(args.source / name, args.output / name)
    for name in ("inputs.bin", "expected.bin"):
        write_matrix(args.output / name, read_matrix(args.source / name)[indices])
    metadata["exported_mnist_samples"] = count
    metadata["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in args.output.glob("*.bin") if p.name != "cpp_scores.bin"}
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
