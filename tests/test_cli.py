"""File-format and CLI boundary tests, using only the Python standard library."""
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

EXE = Path(os.environ["INFERENCE_EXE"]).resolve()


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "model.bin"
        self.inputs = self.root / "inputs.bin"
        self.output = self.root / "output.bin"
        # PyTorch output-major order: first weights [1,2,-1,1], bias [-1,0].
        self.model.write_bytes(b"DMLP0001" + struct.pack("<III9f", 2, 2, 1, 1, 2, -1, 1, -1, 0, 3, -2, 0.5))
        self.inputs.write_bytes(b"DMLPMAT1" + struct.pack("<II6f", 3, 2, 1, 2, -1, -2, 0, 0))

    def run_cli(self, success=True):
        result = subprocess.run([str(EXE), "infer", str(self.model), str(self.inputs), str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)

    def test_round_trip(self):
        self.run_cli()
        self.assertEqual(struct.unpack("<3f", self.output.read_bytes()[16:]), (10.5, 0.5, 0.5))

    def test_empty_batch(self):
        self.inputs.write_bytes(b"DMLPMAT1" + struct.pack("<II", 0, 2))
        self.run_cli()
        self.assertEqual(len(self.output.read_bytes()), 16)

    def test_malformed_model(self):
        good = self.model.read_bytes()
        for bad in (b"", b"WRONG!!!" + good[8:], good[:-1], good + b"x",
                    b"DMLP0001" + struct.pack("<III", 0, 2, 1),
                    b"DMLP0001" + struct.pack("<III", 0xFFFFFFFF, 0xFFFFFFFF, 10),
                    good[:20] + struct.pack("<f", float("nan")) + good[24:]):
            with self.subTest(length=len(bad)):
                self.model.write_bytes(bad)
                self.run_cli(False)

    def test_malformed_inputs(self):
        good = self.inputs.read_bytes()
        for bad in (good[:-1], good + b"x", b"WRONG!!!" + good[8:],
                    b"DMLPMAT1" + struct.pack("<II3f", 1, 3, 0, 0, 0),
                    good[:16] + struct.pack("<f", float("inf")) + good[20:]):
            with self.subTest(length=len(bad)):
                self.inputs.write_bytes(bad)
                self.run_cli(False)

    def test_missing_file(self):
        self.model.unlink()
        self.run_cli(False)


if __name__ == "__main__":
    unittest.main()
