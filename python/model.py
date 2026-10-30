"""The reference architecture and endian-explicit interchange format."""
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn


class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs):
        return self.fc2(torch.relu(self.fc1(inputs)))


def export_model(model, path):
    with Path(path).open("wb") as stream:
        stream.write(b"DMLP0001" + struct.pack("<III", 784, 64, 10))
        for layer in (model.fc1, model.fc2):
            # PyTorch is output-major; C++ transposes once during model loading.
            for tensor in (layer.weight, layer.bias):
                stream.write(tensor.detach().cpu().numpy().astype("<f4").tobytes())


def write_matrix(path, values):
    values = np.asarray(values, dtype="<f4", order="C")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("matrix must be 2D and finite")
    with Path(path).open("wb") as stream:
        stream.write(b"DMLPMAT1" + struct.pack("<II", *values.shape))
        stream.write(values.tobytes())


def read_matrix(path):
    payload = Path(path).read_bytes()
    if len(payload) < 16 or payload[:8] != b"DMLPMAT1":
        raise ValueError("bad matrix header")
    rows, columns = struct.unpack("<II", payload[8:16])
    if columns == 0 or len(payload) != 16 + rows * columns * 4:
        raise ValueError("bad matrix dimensions or length")
    values = np.frombuffer(payload, dtype="<f4", offset=16).reshape(rows, columns).copy()
    if not np.isfinite(values).all():
        raise ValueError("non-finite matrix")
    return values
