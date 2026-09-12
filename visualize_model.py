"""Plot open-loop predictions of a trained model against recorded data.

Rolls the checkpoint out over the episodes in ``cfg.viz_train_episodes`` and
over a slice of the test file, and writes one figure per episode to
``cfg.plot_dir`` with R2 and MSE per channel in the panel titles.

Usage:

    python visualize_model.py

All settings come from src/speed_control/speed_control/config.py. Edit that
file and re-run; this script takes no command-line arguments.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "speed_control"))

from speed_control.config import DEFAULT_CONFIG as cfg
from speed_control.features import Scaler, build_features
from speed_control.metrics import mse, r2_score
from speed_control.model import load_model

TEST_SLICE_LABEL = "test"


def load_episode(df: pd.DataFrame, episode_id: int) -> pd.DataFrame:
    return df[df["episode_id"] == episode_id]


def load_test_slice(config) -> pd.DataFrame:
    """Slice the held-out file according to the viz_test_* settings."""
    df = pd.read_csv(config.test_file)
    start = config.viz_test_start
    if start >= len(df):
        raise IndexError(
            f"viz_test_start={start} is past the end of {config.test_file} "
            f"({len(df)} rows)"
        )
    end = len(df) if config.viz_test_len is None else start + config.viz_test_len
    return df.iloc[start:min(end, len(df))].reset_index(drop=True)


def predict(model, scaler: Scaler, group: pd.DataFrame, config, device) -> np.ndarray:
    """Roll the model open-loop over an episode and return unscaled outputs."""
    features = build_features(group[list(config.input_cols)].values, config.history_window)
    u_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_norm = model(scaler.normalize_u(u_tensor), model.initial_states(1, device))
    return scaler.denormalize_y(y_norm.cpu())[0].numpy()


def plot_episode(label, group: pd.DataFrame, y_pred: np.ndarray, config) -> str:
    """Write one figure: the command signal plus every configured channel."""
    y_true = group[list(config.target_cols)].values
    time_axis = np.arange(len(group)) * config.dt

    num_panels = 1 + len(config.viz_channels)
    plt.figure(figsize=(12, 3 * num_panels))
    plt.suptitle(f"Open-loop prediction - episode {label}", fontsize=16)

    plt.subplot(num_panels, 1, 1)
    plt.plot(time_axis, group[config.input_cols[0]].values, "k--", label="Input")
    plt.title("Input signal (effort command)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    for panel, channel in enumerate(config.viz_channels):
        name = config.target_cols[channel]
        true_series = y_true[:, channel]
        pred_series = y_pred[:, channel]

        plt.subplot(num_panels, 1, panel + 2)
        plt.plot(time_axis, true_series, "g-", linewidth=2, alpha=0.6, label=f"True {name}")
        plt.plot(time_axis, pred_series, "r--", linewidth=2, label=f"Pred {name}")
        plt.title(f"{name} | R2 {r2_score(true_series, pred_series) * 100:.2f}% "
                  f"| MSE {mse(true_series, pred_series):.5f}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        if panel == len(config.viz_channels) - 1:
            plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    path = os.path.join(config.plot_dir, f"pred_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def visualize() -> None:
    device = torch.device("cpu")

    for path in (cfg.model_save_path, cfg.scaler_save_path, cfg.data_file):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    scaler = Scaler.load(cfg.scaler_save_path)
    model = load_model(cfg, device)
    print(f"Loaded {os.path.abspath(cfg.model_save_path)}")

    os.makedirs(cfg.plot_dir, exist_ok=True)
    df_train = pd.read_csv(cfg.data_file)

    targets = [(episode_id, load_episode(df_train, episode_id))
               for episode_id in cfg.viz_train_episodes]

    if cfg.viz_include_test:
        if os.path.exists(cfg.test_file):
            targets.append((TEST_SLICE_LABEL, load_test_slice(cfg)))
        else:
            print(f"Warning: {cfg.test_file} not found, skipping the test slice")

    for label, group in targets:
        if group is None or len(group) == 0:
            print(f"Episode {label} is empty, skipping")
            continue
        y_pred = predict(model, scaler, group, cfg, device)
        print(f"Saved {plot_episode(label, group, y_pred, cfg)} ({len(group)} steps)")


if __name__ == "__main__":
    visualize()
