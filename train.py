"""Train the cascaded NSSM on collected motor data.

All settings come from src/speed_control/speed_control/config.py. Edit that file
and re-run; this script takes no command-line arguments.
"""

import os
import subprocess
import sys
import time

import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional; training still runs without it
    SummaryWriter = None

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "speed_control"))

from speed_control.config import DEFAULT_CONFIG as cfg
from speed_control.dataset import (
    EpisodeDataset, describe_split, episode_signal_types, stratified_split,
)
from speed_control.metrics import r2_score
from speed_control.model import build_model


def set_seeds(config) -> None:
    torch.manual_seed(config.seed)
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Determinism enabled: cuDNN autotuning off, runs are slower")


def launch_tensorboard(config) -> None:
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", "runs", "--port", str(config.tensorboard_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"TensorBoard serving at http://localhost:{config.tensorboard_port}")
    except OSError as error:
        print(f"Warning: could not start TensorBoard: {error}")


def sequence_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Sum of per-channel mean squared error over the whole rollout."""
    return sum(
        torch.mean((y_true[:, :, c] - y_pred[:, :, c]) ** 2)
        for c in range(y_true.shape[-1])
    )


def run_epoch(model, loader, device, optimizer=None):
    """Run one pass. Passing an optimizer switches to training mode."""
    is_training = optimizer is not None
    model.train(is_training)

    loss_sum = 0.0
    r2_sum = 0.0
    with torch.set_grad_enabled(is_training):
        for u_batch, y_batch in loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(u_batch, model.initial_states(u_batch.size(0), device))
            loss = sequence_loss(y_batch, y_pred)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()
            r2_sum += r2_score(y_batch.detach(), y_pred.detach())

    return loss_sum / len(loader), r2_sum / len(loader)


def train() -> None:
    set_seeds(cfg)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.scaler_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working directory: {os.getcwd()}")
    print(f"Device: {device}")
    print(f"Model: state_dim={cfg.state_dim} output_dim={cfg.output_dim} "
          f"seq_len={cfg.seq_len} history_window={cfg.history_window}")

    if not os.path.exists(cfg.data_file):
        raise FileNotFoundError(f"Data file not found: {cfg.data_file}")

    print(f"Loading {os.path.abspath(cfg.data_file)}")
    df = pd.read_csv(cfg.data_file)

    dataset = EpisodeDataset(df, cfg)
    print(f"Episodes: {len(dataset)} usable of {df['episode_id'].nunique()} in file")

    dataset.scaler.save(cfg.scaler_save_path)
    print(f"Scaler saved to {os.path.abspath(cfg.scaler_save_path)}")

    signal_types = episode_signal_types(df, dataset, cfg)
    train_indices, val_indices = stratified_split(signal_types, cfg)
    print(f"Split at val_ratio={cfg.val_ratio}: "
          f"{len(train_indices)} train / {len(val_indices)} val")
    print(describe_split(signal_types, train_indices, val_indices))

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=cfg.batch_size, shuffle=False
    )

    model = build_model(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=cfg.lr_decay_gamma)

    writer = None
    if SummaryWriter is None:
        print("Warning: tensorboard is not installed, scalar logging disabled")
    else:
        run_dir = os.path.join("runs", f"{cfg.run_name}_{int(time.time())}")
        writer = SummaryWriter(run_dir)
        print(f"Logging to {run_dir}")
        if cfg.launch_tensorboard:
            launch_tensorboard(cfg)

    best_val_loss = float("inf")
    print("Training start")
    for epoch in range(cfg.epochs):
        train_loss, train_r2 = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_r2 = run_epoch(model, val_loader, device)

        if writer is not None:
            writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
            writer.add_scalars("Accuracy_R2", {"Train": train_r2, "Val": val_r2}, epoch)

        if (epoch + 1) % cfg.log_every == 0:
            print(f"Epoch [{epoch + 1}/{cfg.epochs}] "
                  f"loss {train_loss:.5f}/{val_loss:.5f} "
                  f"R2 {train_r2 * 100:.1f}%/{val_r2 * 100:.1f}% "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"  new best val loss {best_val_loss:.5f} "
                  f"(R2 {val_r2 * 100:.2f}%), saved")

        scheduler.step()

    if writer is not None:
        writer.close()
    print(f"Training finished. Best val loss: {best_val_loss:.6f}")
    print(f"Model: {os.path.abspath(cfg.model_save_path)}")


if __name__ == "__main__":
    train()
