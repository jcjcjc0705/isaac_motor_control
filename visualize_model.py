import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import sys
import json

# 引入你的 Config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'speed_control'))
from speed_control.config import DEFAULT_TRAIN_CONFIG

# --- 自訂測試資料切片參數 ---
# 在這裡修改您想要擷取 test.csv 的範圍
TEST_START_STEP = 8800     # 從哪一步開始切
TEST_SEQ_LEN = 1100      # 總共要取幾步。如果想拿全部，可以設為 None

# --- NSSM Model 定義 ---
class NSSM(torch.nn.Module):
    def __init__(self, state_dim, input_dim, output_dim):
        super().__init__()
        self.state_dim = state_dim
        
        self.f_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + input_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, state_dim),
        )
        
        self.g_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, output_dim),
        )
        
        self.d_net = torch.nn.Linear(input_dim, output_dim, bias=False)

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
            
        return torch.cat(y_pred_list, dim=1)

class CascadedSystem(torch.nn.Module):
    def __init__(self, state_dim, raw_input_dim, history_window, output_dim):
        super().__init__()
        self.output_dim = output_dim

        # Input: [u, diff] -> multiplier = 2
        self.feature_multiplier = 2    
        base_input_dim = (raw_input_dim * self.feature_multiplier) * history_window

        self.stages = torch.nn.ModuleList()
        curr_input_dim = base_input_dim

        for i in range(output_dim):
            self.stages.append(NSSM(state_dim, curr_input_dim, output_dim=1))
            curr_input_dim += 1

    def forward(self, u_sequence, x_inits):
        current_input = u_sequence
        outputs = []

        for i in range(self.output_dim):
            pred = self.stages[i](current_input, x_inits[i])
            outputs.append(pred)
            current_input = torch.cat([current_input, pred.detach()], dim=2)

        return torch.cat(outputs, dim=2)

def calculate_metrics(y_true, y_pred):
    """計算 R2 Score 和 MSE"""
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)
    
    target_var = torch.var(y_true_tensor, unbiased=False)
    if target_var < 1e-6:
        target_var = 1.0

    sse = torch.mean((y_true_tensor - y_pred_tensor) ** 2)
    r2 = 1.0 - (sse / target_var)
    mse = sse
    
    return r2.item(), mse.item()

def visualize():
    cfg = DEFAULT_TRAIN_CONFIG
    device = torch.device("cpu") 

    # 1. 載入 Scaler
    if not os.path.exists(cfg.scaler_save_path):
        print(f"Error: Scaler file not found at {cfg.scaler_save_path}")
        return

    with open(cfg.scaler_save_path, "r") as f:
        scaler = json.load(f)
    
    u_mean = torch.tensor(scaler["u_mean"])
    u_std = torch.tensor(scaler["u_std"])
    y_mean = torch.tensor(scaler["y_mean"])
    y_std = torch.tensor(scaler["y_std"])

    # 2. 載入模型 (動態支援 OUTPUT_DIM)
    if not os.path.exists(cfg.model_save_path):
        print(f"Error: Model file not found at {cfg.model_save_path}")
        return

    REAL_INPUT_DIM = cfg.input_dim * cfg.history_window
    model = CascadedSystem(cfg.state_dim, cfg.input_dim, cfg.history_window, cfg.output_dim)
    
    try:
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    df = pd.read_csv(cfg.data_file)
    df_cols = df.columns.tolist()
    
    # 挑選不同類型的 Episode ID
    test_ids = [10, 100, 250, 400, "test"]

    if os.path.exists(cfg.data_file):
        df_train = pd.read_csv(cfg.data_file)
    else:
        print(f"Warning: Train data file not found at {cfg.data_file}")
        df_train = None

    test_data_path = os.path.join(cfg.save_dir, "test.csv")
    
    input_cols = ["input_u"]
    
    # 動態決定輸出欄位名稱 (支援新舊版與維度)
    j1_name = "vel_joint1" if "vel_joint1" in df_cols else "vel_joint"
    if cfg.output_dim == 2:
        target_cols = ["vel_motor", j1_name]
    elif cfg.output_dim == 3:
        target_cols = ["vel_motor", j1_name, "vel_joint2"]
    else:
        raise ValueError(f"Unsupported OUTPUT_DIM: {cfg.output_dim}")

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating plots in '{output_dir}/'...")

    for ep_id in test_ids:
        group = None
        
        if ep_id == "test":
            if os.path.exists(test_data_path):
                print(f"Loading Test Data from {test_data_path}...")
                df_test = pd.read_csv(test_data_path)

                total_len = len(df_test)
                start_idx = TEST_START_STEP
                end_idx = start_idx + TEST_SEQ_LEN if TEST_SEQ_LEN is not None else total_len
                end_idx = min(end_idx, total_len)

                if start_idx < total_len:
                    group = df_test.iloc[start_idx:end_idx].reset_index(drop=True)
                    print(f"   -> Slicing step {start_idx} to {end_idx} (Total {len(group)} steps)")
                else:
                    print(f"   -> Start index {start_idx} is out of bounds (max {total_len}). Skipping test plot.")
                    continue
            else:
                print(f"Test data file not found: {test_data_path}, skipping.")
                continue
        else:
            if df_train is not None:
                group = df_train[df_train["episode_id"] == ep_id]
            
        if group is None or len(group) == 0: 
            print(f"Episode {ep_id} not found or empty, skipping.")
            continue
        
        u_raw_numpy = group[input_cols].values 
        y_raw_numpy = group[target_cols].values

        u_diff = np.zeros_like(u_raw_numpy)
        u_diff[1:] = u_raw_numpy[1:] - u_raw_numpy[:-1]
        
        u_combined = np.concatenate([u_raw_numpy, u_diff], axis=1)

        u_augmented = []
        for i in range(cfg.history_window):
            u_shifted = np.roll(u_combined, i, axis=0)
            u_shifted[:i] = 0.0
            u_augmented.append(u_shifted)
        
        u_stacked = np.concatenate(u_augmented, axis=1)

        u_tensor = torch.tensor(u_stacked, dtype=torch.float32).unsqueeze(0)
        u_norm = (u_tensor - u_mean) / u_std

        with torch.no_grad():
            # 動態生成 x_inits 列表
            x_inits = [torch.zeros(1, cfg.state_dim).to(device) for _ in range(cfg.output_dim)]
            y_pred_norm = model(u_norm, x_inits)
            
        # 反正規化
        y_pred = y_pred_norm * y_std + y_mean
        y_pred_np = y_pred[0].detach().numpy()
        
        time_steps = np.arange(len(group)) * cfg.dt
        
        # 動態決定子圖數量 (Input + 所有 Output)
        total_plots = 1 + cfg.output_dim
        # 根據子圖數量調整圖片高度，避免擠在一起
        fig_height = 3 * total_plots 
        plt.figure(figsize=(12, fig_height))
        plt.suptitle(f"Prediction - {ep_id}", fontsize=16)
        
        # 1. 畫 Input 訊號
        plt.subplot(total_plots, 1, 1)
        plt.plot(time_steps, u_raw_numpy[:, 0], 'k--', label="Input")
        plt.title("Input Signal (Velocity Command)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        # 2. 畫所有 Output 訊號
        for i, col_name in enumerate(target_cols):
            plt.subplot(total_plots, 1, i + 2)
            
            y_true_seq = y_raw_numpy[:, i]
            y_pred_seq = y_pred_np[:, i] 

            r2, mse = calculate_metrics(y_true_seq, y_pred_seq)
            
            plt.plot(time_steps, y_true_seq, 'g-', linewidth=2, alpha=0.6, label=f"True {col_name}")
            plt.plot(time_steps, y_pred_seq, 'r--', linewidth=2, label=f"Pred {col_name}")
            
            plt.title(f"{col_name} | R²: {r2*100:.2f}% | MSE: {mse:.5f}")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            
            if i == len(target_cols) - 1:
                plt.xlabel("Time [s]")

        plt.tight_layout()
        plt.subplots_adjust(top=0.95) 
        filename = f"pred_{ep_id}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    visualize()