"""Compare ALL C++ logits against exported and live PyTorch reference scores."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from model import DigitMLP, read_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0 or not np.isfinite([args.atol, args.rtol]).all():
        parser.error("tolerances must be finite and nonnegative")
    torch.set_num_threads(1)
    model = DigitMLP().eval()
    model.load_state_dict(torch.load(args.artifacts / "model.pt", map_location="cpu", weights_only=True))
    inputs = read_matrix(args.artifacts / "inputs.bin")
    expected = read_matrix(args.artifacts / "expected.bin")
    with torch.inference_mode():
        reference = model(torch.from_numpy(inputs)).numpy()
    np.testing.assert_allclose(expected, reference, atol=args.atol, rtol=args.rtol)
    destination = args.artifacts / "cpp_scores.bin"
    subprocess.run([str(args.exe.resolve()), "infer", str(args.artifacts / "model.bin"),
                    str(args.artifacts / "inputs.bin"), str(destination), args.mode], check=True)
    actual = read_matrix(destination)
    np.testing.assert_allclose(actual, reference, atol=args.atol, rtol=args.rtol)
    error = np.abs(actual.astype(np.float64) - reference)
    report = {"mode": args.mode, "samples": len(inputs), "logits": actual.size, "max_absolute_error": float(error.max()),
              "mean_absolute_error": float(error.mean()),
              "max_relative_error_floor_1e-6": float((error / np.maximum(np.abs(reference), 1e-6)).max()),
              "prediction_agreement": float((actual.argmax(1) == reference.argmax(1)).mean()),
              "atol": args.atol, "rtol": args.rtol, "passed": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
