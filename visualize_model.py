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
    def __init__(self, state_dim, raw_input_dim, history_window):
        super().__init__()

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

    # 2. 載入模型
    if not os.path.exists(cfg.model_save_path):
        print(f"Error: Model file not found at {cfg.model_save_path}")
        return

    REAL_INPUT_DIM = cfg.input_dim * cfg.history_window
    model = CascadedSystem(cfg.state_dim, cfg.input_dim, cfg.history_window)
    
    try:
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()

    df = pd.read_csv(cfg.data_file)
    
    # 挑選不同類型的 Episode ID (涵蓋 RAMPS, PRBS, CHIRP)
    test_ids = [10, 100, 250, 400, "test"]

    if os.path.exists(cfg.data_file):
        df_train = pd.read_csv(cfg.data_file)
    else:
        print(f"Warning: Train data file not found at {cfg.data_file}")
        df_train = None

    # 定義測試資料路徑
    test_data_path = os.path.join(cfg.save_dir, "test.csv")
    
    input_cols = ["input_u"]
    
    # 動態決定輸出欄位名稱
    if cfg.output_dim == 2:
        target_cols = ["vel_motor", "vel_joint"]
    else:
        target_cols = ["pos_motor", "vel_motor", "pos_joint", "vel_joint"]

    # 建立輸出資料夾
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating plots in '{output_dir}/'...")

    for ep_id in test_ids:
        group = None
        
        # --- [分支] 判斷要讀哪個檔案 ---
        if ep_id == "test":
            # 如果是測試模式，讀取 test.csv
            if os.path.exists(test_data_path):
                print(f"Loading Test Data from {test_data_path}...")
                df_test = pd.read_csv(test_data_path)

                total_len = len(df_test)
                
                # ✅ 應用自訂的切片參數
                start_idx = TEST_START_STEP
                end_idx = start_idx + TEST_SEQ_LEN if TEST_SEQ_LEN is not None else total_len
                
                # 確保 end_idx 不會超出資料長度
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
            # 如果是訓練模式，從 df_train 篩選
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
            x_init_motor = torch.zeros(1, cfg.state_dim)
            x_init_joint = torch.zeros(1, cfg.state_dim)
            y_pred_norm = model(u_norm, x_init_motor, x_init_joint)
            
        # 反正規化
        y_pred = y_pred_norm * y_std + y_mean
        y_pred_np = y_pred[0].detach().numpy()
        
        time_steps = np.arange(len(group)) * cfg.dt
        
        plt.figure(figsize=(12, 9))
        plt.suptitle(f"Prediction - {ep_id}", fontsize=16)
        
        # 1. Input
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, u_raw_numpy[:, 0], 'k--', label="Input")
        plt.title("Input Signal")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Output
        for i, col_name in enumerate(target_cols):
            plt.subplot(3, 1, i + 2)
            
            y_true_seq = y_raw_numpy[:, i]
            y_pred_seq = y_pred_np[:, i] # 直接對應 [0, 1]

            r2, mse = calculate_metrics(y_true_seq, y_pred_seq)
            
            plt.plot(time_steps, y_true_seq, 'g-', linewidth=2, alpha=0.6, label=f"True {col_name}")
            plt.plot(time_steps, y_pred_seq, 'r--', linewidth=2, label=f"Pred {col_name}")
            
            plt.title(f"{col_name} | R²: {r2*100:.2f}% | MSE: {mse:.5f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if i == len(target_cols) - 1:
                plt.xlabel("Time [s]")

        plt.tight_layout()
        filename = f"pred_{ep_id}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

if __name__ == "__main__":
    visualize()