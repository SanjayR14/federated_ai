"""
Federated Model Definition and Serialization / Quantization Utilities.
Defines the neural network architecture suitable for parameter aggregation (FedAvg)
and utilities for model update serialization and quantization.
"""

import io
import gzip
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple


class FederatedHealthNet(nn.Module):
    """
    Multi-layer Perceptron (MLP) for glycemic (HbA1c) regression.
    Designed for parameter-level Federated Averaging (FedAvg).
    """

    def __init__(self, input_dim: int = 5, hidden_1: int = 32, hidden_2: int = 16, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (for Flower / FedAvg)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy arrays into the PyTorch module."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict}
    net.load_state_dict(state_dict, strict=True)


def get_model_size_bytes(net: nn.Module) -> int:
    """Calculate the memory footprint of model parameters in bytes (FP32)."""
    return sum(p.numel() * p.element_size() for p in net.parameters())


def quantize_fp16(parameters: List[np.ndarray]) -> List[np.ndarray]:
    """
    Quantize parameter arrays to Float16 for 50% communication bandwidth reduction.
    """
    return [p.astype(np.float16) for p in parameters]


def dequantize_fp16(parameters: List[np.ndarray]) -> List[np.ndarray]:
    """Dequantize Float16 parameter arrays back to Float32 for model computation."""
    return [p.astype(np.float32) for p in parameters]


def serialize_parameters(parameters: List[np.ndarray]) -> bytes:
    """Serialize parameter list into bytes using an in-memory buffer."""
    buf = io.BytesIO()
    np.savez(buf, *parameters)
    return buf.getvalue()


def deserialize_parameters(payload: bytes) -> List[np.ndarray]:
    """Deserialize parameter bytes back to a list of NumPy arrays."""
    buf = io.BytesIO(payload)
    archive = np.load(buf)
    return [archive[f] for f in archive.files]


def compress_payload(payload: bytes) -> bytes:
    """Gzip compression for serialized model updates."""
    return gzip.compress(payload)


def decompress_payload(compressed: bytes) -> bytes:
    """Gzip decompression for serialized model updates."""
    return gzip.decompress(compressed)
