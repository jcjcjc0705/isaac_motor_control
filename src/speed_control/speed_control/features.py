"""Command feature construction and input/output scaling.

Training and inference both build their inputs with :func:`build_features` and
scale them with :class:`Scaler`, which is what keeps the two consistent.
"""

import json
from typing import Dict

import numpy as np
import torch

# Each raw command channel expands to [u, du].
FEATURE_MULTIPLIER = 2


def feature_dim(input_dim: int, history_window: int) -> int:
    """Width of one feature vector produced by :func:`build_features`."""
    return input_dim * FEATURE_MULTIPLIER * history_window


def build_features(u_raw: np.ndarray, history_window: int) -> np.ndarray:
    """Expand a command sequence into stacked-history features.

    Args:
        u_raw: raw commands, shape [T, input_dim].
        history_window: how many past steps to stack.

    Returns:
        Array of shape [T, feature_dim(input_dim, history_window)]. Lag ``i`` is
        shifted forward and zero-padded, so row ``t`` never sees a future step.
    """
    u_diff = np.zeros_like(u_raw)
    u_diff[1:] = u_raw[1:] - u_raw[:-1]
    u_features = np.concatenate([u_raw, u_diff], axis=1)

    lags = []
    for lag in range(history_window):
        shifted = np.roll(u_features, lag, axis=0)
        shifted[:lag] = 0.0
        lags.append(shifted)
    return np.concatenate(lags, axis=1)


class Scaler:
    """Per-channel standardisation shared by training and inference."""

    def __init__(self, u_mean, u_std, y_mean, y_std):
        self.u_mean = torch.as_tensor(u_mean, dtype=torch.float32)
        self.u_std = torch.as_tensor(u_std, dtype=torch.float32)
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.as_tensor(y_std, dtype=torch.float32)

    @classmethod
    def fit(cls, u_data: torch.Tensor, y_data: torch.Tensor) -> "Scaler":
        """Fit on tensors shaped [N, T, C]; flat channels get unit std."""
        u_flat = u_data.reshape(-1, u_data.shape[-1])
        y_flat = y_data.reshape(-1, y_data.shape[-1])

        u_std = u_flat.std(dim=0)
        y_std = y_flat.std(dim=0)
        u_std[u_std < 1e-6] = 1.0
        y_std[y_std < 1e-6] = 1.0

        return cls(u_flat.mean(dim=0), u_std, y_flat.mean(dim=0), y_std)

    @classmethod
    def load(cls, path: str) -> "Scaler":
        with open(path, "r") as handle:
            payload: Dict = json.load(handle)
        return cls(payload["u_mean"], payload["u_std"],
                   payload["y_mean"], payload["y_std"])

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump({
                "u_mean": self.u_mean.tolist(),
                "u_std": self.u_std.tolist(),
                "y_mean": self.y_mean.tolist(),
                "y_std": self.y_std.tolist(),
            }, handle, indent=4)

    def normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean) / self.u_std

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean
