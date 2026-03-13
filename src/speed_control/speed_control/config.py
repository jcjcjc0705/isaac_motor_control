from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os
import time


TARGET_DATA_FILE = "data/train.csv"

STATE_DIM = 128   # 模型階數
OUTPUT_DIM = 3    # 輸出維度 (vel_m, vel_j)

# --- 其他設定 ---
EPISODE_LEN = 200          # 每個回合蒐集 200 步 -> 配合 Training Seq Len
RESET_LEN = 100            # 每個回合間隔 100 步讓馬達歸零
TOTAL_EPISODES = 500      # 總共蒐集 500 個回合  
DT = 0.05
AMPLITUDE = 10.0

@dataclass
class ExperimentConfig:
    data_file: str = TARGET_DATA_FILE
    state_dim: int = STATE_DIM
    output_dim: int = OUTPUT_DIM
    seq_len: int = RESET_LEN + EPISODE_LEN
    total_len: int = TOTAL_EPISODES * (EPISODE_LEN + RESET_LEN)

    dt: float = DT
    amplitude: float = AMPLITUDE

    episode_len: int = EPISODE_LEN
    reset_len: int = RESET_LEN
    total_episodes: int = TOTAL_EPISODES
    
    signal_type: str = "MIXED"
    signal_to_mix: List[Tuple] = field(default_factory=lambda: [
        ("RAMPS", 0.2),
        ("PRBS", 0.2),
        ("CHIRP", 0.2),
        ("MULTISINE", 0.4)
    ])

    signal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "RAMPS": (0.1, 1.5),
        "PRBS":  (40, 4),
        "CHIRP": (0.5, 8.0),
        "MULTISINE": (0.1, 6.0),
        "SMOOTH_NOISE": (0.1, 6.0)
    })

    input_dim: int = 1
    history_window: int = 60
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 200
    lr_decay_gamma: float = 0.99
    sliding_stride: Optional[int] = EPISODE_LEN + RESET_LEN

    save_dir: str = "data"
    model_dir: str = "models"
    scaler_dir: str = "scalers"

    model_save_path: str = ""
    scaler_save_path: str = ""

    def __post_init__(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.scaler_dir, exist_ok=True)

        if self.data_file:
            base_name = os.path.splitext(os.path.basename(self.data_file))[0]             
            self.model_save_path = os.path.join(
                self.model_dir, 
                f"{base_name}_dim{self.state_dim}_model.pth"
            )
            self.scaler_save_path = os.path.join(
                self.scaler_dir, 
                f"{base_name}_scaler.json"
            )

        self.sliding_stride = self.seq_len

    @property
    def save_path(self) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"motor_data_{timestamp}.csv"
        return os.path.join(self.save_dir, filename)

    @property
    def run_name(self) -> str:
        base_name = os.path.splitext(os.path.basename(self.data_file))[0]
        return f"{base_name}_dim{self.state_dim}"


DEFAULT_CONFIG = ExperimentConfig()

DEFAULT_TRAIN_CONFIG = ExperimentConfig()