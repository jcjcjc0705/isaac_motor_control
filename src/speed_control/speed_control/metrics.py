"""Shared evaluation metrics.

Training and visualisation used to compute R2 differently, which made their
numbers incomparable. Both now call :func:`r2_score`.
"""

import torch


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean per-channel R2, averaged over the last dimension.

    Accepts [..., C] or a 1-D sequence treated as a single channel. Channels
    whose variance is numerically zero contribute a denominator of 1 so a
    constant target cannot blow the score up.
    """
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
        y_pred = y_pred.unsqueeze(-1)

    total = 0.0
    channels = y_true.shape[-1]
    for channel in range(channels):
        true_channel = y_true[..., channel]
        variance = torch.var(true_channel, unbiased=False)
        if variance < 1e-6:
            variance = torch.tensor(1.0)
        sse = torch.mean((true_channel - y_pred[..., channel]) ** 2)
        total += (1.0 - sse / variance).item()
    return total / channels


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    return torch.mean((y_true - y_pred) ** 2).item()
