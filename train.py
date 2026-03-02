import os
import sys
import time
import json
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

# 引入 Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'speed_control'))
from speed_control.config import DEFAULT_TRAIN_CONFIG

# --- Configuration ---
cfg = DEFAULT_TRAIN_CONFIG

STATE_DIM = cfg.state_dim
INPUT_DIM = cfg.input_dim
OUTPUT_DIM = cfg.output_dim
SEQ_LEN = cfg.seq_len
BATCH_SIZE = cfg.batch_size
LEARNING_RATE = cfg.learning_rate
EPOCHS = cfg.epochs
LR_DECAY_GAMMA = cfg.lr_decay_gamma
HISTORY_WINDOW = cfg.history_window

DATA_FILE = cfg.data_file
MODEL_SAVE_PATH = cfg.model_save_path
SCALER_SAVE_PATH = cfg.scaler_save_path
MODEL_DIR = cfg.model_dir
SCALER_DIR = cfg.scaler_dir


# --- Model Definitions ---

class NSSM(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim

        self.f_net = nn.Sequential(
            nn.Linear(state_dim + input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, state_dim),
        )

        self.g_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        )

        self.d_net = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, u_sequence, x_initial):
        batch_size = u_sequence.shape[0]
        seq_len = u_sequence.shape[1]

        x_current = x_initial
        y_pred_list = []

        for t in range(seq_len):
            u_t = u_sequence[:, t, :]
            y_pred_t = self.g_net(x_current) + self.d_net(u_t)
            y_pred_list.append(y_pred_t.unsqueeze(1))

            xu_vec = torch.cat((x_current, u_t), dim=1)
            x_next = self.f_net(xu_vec)
            x_current = x_next

        y_pred_sequence = torch.cat(y_pred_list, dim=1)
        return y_pred_sequence


class CascadedSystem(nn.Module):
    def __init__(self, state_dim, raw_input_dim, history_window):
        super().__init__()

        # Input: [u, diff] -> multiplier = 2
        self.feature_multiplier = 2
        self.motor_input_dim = (raw_input_dim * self.feature_multiplier) * history_window

        self.motor_output_dim = 1
        self.motor_net = NSSM(state_dim, self.motor_input_dim, output_dim=self.motor_output_dim)

        self.joint_input_dim = self.motor_input_dim + self.motor_output_dim
        self.joint_net = NSSM(state_dim, self.joint_input_dim, output_dim=1)

    def forward(self, u_sequence, x_init_motor, x_init_joint):
        pred_motor = self.motor_net(u_sequence, x_init_motor)
        motor_feature = pred_motor.detach()

        joint_input = torch.cat([u_sequence, motor_feature], dim=2)
        pred_joint = self.joint_net(joint_input, x_init_joint)

        return torch.cat([pred_motor, pred_joint], dim=2)


# --- Dataset ---

class MotorData(Dataset):
    def __init__(self, df, input_cols, target_cols, seq_len, history_window, scaler=None):
        self.seq_len = seq_len
        self.history_window = history_window

        grouped = df.groupby('episode_id')

        u_list = []
        y_list = []

        print(f"--- Processing {len(grouped)} episodes... ---")

        for ep_id, group in sorted(grouped):
            if len(group) == self.seq_len:
                u_raw = group[input_cols].values
                y_raw = group[target_cols].values

                u_diff = np.zeros_like(u_raw)
                u_diff[1:] = u_raw[1:] - u_raw[:-1]

                # Feature: [u, diff]
                u_features = np.concatenate([u_raw, u_diff], axis=1)

                u_augmented = []
                for i in range(history_window):
                    u_shifted = np.roll(u_features, i, axis=0)
                    u_shifted[:i] = 0.0
                    u_augmented.append(u_shifted)

                u_val = np.concatenate(u_augmented, axis=1)
                u_list.append(u_val)
                y_list.append(y_raw)

        self.u_data = torch.tensor(np.array(u_list), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(y_list), dtype=torch.float32)

        print(f"--- Valid Episodes: {self.u_data.shape[0]} / {len(grouped)} ---")

        if scaler is None:
            u_flat = self.u_data.view(-1, self.u_data.shape[-1])
            y_flat = self.y_data.view(-1, self.y_data.shape[-1])

            self.u_mean = u_flat.mean(dim=0)
            self.u_std = u_flat.std(dim=0)
            self.y_mean = y_flat.mean(dim=0)
            self.y_std = y_flat.std(dim=0)

            self.u_std[self.u_std < 1e-6] = 1.0
            self.y_std[self.y_std < 1e-6] = 1.0
        else:
            self.u_mean = torch.tensor(scaler["u_mean"], dtype=torch.float32)
            self.u_std = torch.tensor(scaler["u_std"], dtype=torch.float32)
            self.y_mean = torch.tensor(scaler["y_mean"], dtype=torch.float32)
            self.y_std = torch.tensor(scaler["y_std"], dtype=torch.float32)

        self.u_data = (self.u_data - self.u_mean) / self.u_std
        self.y_data = (self.y_data - self.y_mean) / self.y_std

    def get_scaler_dict(self):
        return {
            "u_mean": self.u_mean.tolist(),
            "u_std": self.u_std.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist()
        }

    def __len__(self):
        return self.u_data.shape[0]

    def __getitem__(self, idx):
        return self.u_data[idx], self.y_data[idx]


# --- Helper Functions ---

def calculate_r2(y_true, y_pred):
    target_var = torch.var(y_true, unbiased=False)
    if target_var < 1e-6:
        target_var = 1.0

    sse = torch.mean((y_true - y_pred) ** 2)
    r2 = 1.0 - (sse / target_var)
    return r2.item()


def diff(x):
    """計算序列的時間差分 (近似導數)"""
    return x[:, 1:, :] - x[:, :-1, :]


def get_interleaved_stratified_indices(total_episodes, mix_config, val_ratio=0.2):
    indices_train = []
    indices_val = []

    current_start_id = 0
    remaining_total = total_episodes

    step = int(1 / val_ratio)

    for item in mix_config:
        # ratio = item[1]
        ratio = item[1]

        if item == mix_config[-1]:
            count = remaining_total
        else:
            count = int(total_episodes * ratio)
            remaining_total -= count

        type_indices = np.arange(current_start_id, current_start_id + count)

        v_idx = type_indices[::step]
        v_set = set(v_idx)
        t_idx = np.array([idx for idx in type_indices if idx not in v_set])

        indices_train.extend(t_idx)
        indices_val.extend(v_idx)
        current_start_id += count

    return indices_train, indices_val


# --- Training Loop ---

def train():
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    print(f"--- Start training ---")
    print(f"--- Config: State={STATE_DIM}, Output={OUTPUT_DIM}, SeqLen={SEQ_LEN} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    print(f"--- Loading CSV from {DATA_FILE} ---")
    df = pd.read_csv(DATA_FILE)

    input_cols = ["input_u"]
    if OUTPUT_DIM == 2:
        target_cols = ["vel_motor", "vel_joint"]
    elif OUTPUT_DIM == 4:
        target_cols = ["pos_motor", "vel_motor", "pos_joint", "vel_joint"]
    else:
        raise ValueError(f"Unsupported OUTPUT_DIM: {OUTPUT_DIM}")

    full_dataset = MotorData(df, input_cols, target_cols, SEQ_LEN, HISTORY_WINDOW, scaler=None)

    scaler_dict = full_dataset.get_scaler_dict()
    with open(SCALER_SAVE_PATH, "w") as f:
        json.dump(scaler_dict, f, indent=4)
    print(f"--- Scaler saved to {SCALER_SAVE_PATH} ---")

    real_total_episodes = len(full_dataset)
    train_indices, val_indices = get_interleaved_stratified_indices(
        real_total_episodes,
        cfg.signal_to_mix,
        val_ratio=0.2
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CascadedSystem(STATE_DIM, cfg.input_dim, HISTORY_WINDOW).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)

    run_name = f"{cfg.run_name}_val_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    print(f"--- 🚀 Launching TensorBoard (Background)... ---")
    try:
        subprocess.Popen(
            ["tensorboard", "--logdir", "runs", "--port", "6006"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"--- ✅ TensorBoard is running at http://localhost:6006 ---")
    except Exception as e:
        print(f"--- ⚠️ Warning: Failed to auto-launch TensorBoard: {e}")

    best_val_loss = float("inf")

    print("--- Training Loop Start ---")
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss_sum = 0.0
        train_r2_sum = 0.0

        for u_batch, y_batch in train_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            curr_batch_size = u_batch.size(0)

            x_init_motor = torch.zeros(curr_batch_size, STATE_DIM).to(device)
            x_init_joint = torch.zeros(curr_batch_size, STATE_DIM).to(device)

            y_pred = model(u_batch, x_init_motor, x_init_joint)

            y_true_motor = y_batch[:, :, 0]
            y_pred_motor = y_pred[:, :, 0]
            y_true_joint = y_batch[:, :, 1]
            y_pred_joint = y_pred[:, :, 1]

            loss_motor = torch.mean((y_true_motor - y_pred_motor) ** 2)
            loss_joint = torch.mean((y_true_joint - y_pred_joint) ** 2)
            loss = loss_motor + loss_joint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            with torch.no_grad():
                train_r2_sum += calculate_r2(y_batch, y_pred)

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_r2 = train_r2_sum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_r2_sum = 0.0

        with torch.no_grad():
            for u_batch, y_batch in val_loader:
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                curr_batch_size = u_batch.size(0)

                x_init_motor = torch.zeros(curr_batch_size, STATE_DIM).to(device)
                x_init_joint = torch.zeros(curr_batch_size, STATE_DIM).to(device)

                y_pred = model(u_batch, x_init_motor, x_init_joint)

                y_true_motor = y_batch[:, :, 0]
                y_pred_motor = y_pred[:, :, 0]
                y_true_joint = y_batch[:, :, 1]
                y_pred_joint = y_pred[:, :, 1]

                loss_motor = torch.mean((y_true_motor - y_pred_motor) ** 2)
                loss_joint = torch.mean((y_true_joint - y_pred_joint) ** 2)
                loss = loss_motor + loss_joint

                val_loss_sum += loss.item()
                val_r2_sum += calculate_r2(y_batch, y_pred)

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_r2 = val_r2_sum / len(val_loader)

        # --- Logging ---
        curr_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalars("Loss", {"Train": avg_train_loss, "Val": avg_val_loss}, epoch)
        writer.add_scalars("Accuracy_R2", {"Train": avg_train_r2, "Val": avg_val_r2}, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Loss: {avg_train_loss:.5f}/{avg_val_loss:.5f} | "
                  f"R2: {avg_train_r2*100:.1f}%/{avg_val_r2*100:.1f}% | "
                  f"LR: {curr_lr:.8f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best Model Saved! Val Loss: {best_val_loss:.5f} (Acc: {avg_val_r2*100:.2f}%)")

        scheduler.step()

    writer.close()
    print(f"--- Training Finished. Best Val Loss: {best_val_loss:.6f} ---")


if __name__ == "__main__":
    train()