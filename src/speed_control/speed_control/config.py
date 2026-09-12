"""Central configuration for the motor dynamics identification pipeline.

Every tunable parameter lives here, including training, visualisation and data
inspection settings. None of the scripts take command-line flags: edit this file
and re-run. The ROS data-collection side and the PyTorch training side both
import this module, so they can never disagree about episode layout, feature
width or safety limits.

The dataclass is frozen. Derive variants with ``dataclasses.replace`` instead of
mutating the shared instance.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_file: str = "data/train.csv"
    test_file: str = "data/test.csv"
    data_dir: str = "data"
    model_dir: str = "models"
    scaler_dir: str = "scalers"
    plot_dir: str = "plots"

    # ------------------------------------------------------------------
    # Episode layout, shared by collection and training
    # ------------------------------------------------------------------
    episode_len: int = 200      # excitation steps per episode
    reset_len: int = 100        # settling steps that drive the motor back to zero
    total_episodes: int = 500
    dt: float = 0.05            # control period in seconds, 20 Hz
    progress_every: int = 100   # command steps between collection progress lines

    # ------------------------------------------------------------------
    # CSV schema
    # ------------------------------------------------------------------
    input_cols: Tuple[str, ...] = ("input_u",)
    # Order matters: the first two targets belong to the motor stage, the last
    # two to the joint stage of CascadedSystem.
    target_cols: Tuple[str, ...] = ("pos_motor", "vel_motor", "pos_joint1", "vel_joint1")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    state_dim: int = 128        # latent state order of each NSSM stage
    history_window: int = 60    # past command steps stacked into each feature vector

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 200
    lr_decay_gamma: float = 0.99
    val_ratio: float = 0.2
    log_every: int = 10         # epochs between progress lines
    seed: int = 42
    deterministic: bool = False # cuDNN determinism: reproducible but slower
    launch_tensorboard: bool = False
    tensorboard_port: int = 6006

    # ------------------------------------------------------------------
    # Excitation signals
    # ------------------------------------------------------------------
    amplitude: float = 200.0
    signal_mix: Tuple[Tuple[str, float], ...] = (
        ("RAMPS", 0.2),
        ("PRBS", 0.2),
        ("CHIRP", 0.2),
        ("MULTISINE", 0.4),
    )
    # Per-type parameter range. Frequency in Hz, except PRBS which is hold steps.
    signal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "RAMPS": (0.1, 1.5),
        "PRBS": (20, 60),
        "CHIRP": (0.1, 2.0),
        "MULTISINE": (0.1, 3.0),
        "SMOOTH_NOISE": (0.1, 4.0),
    })

    # ------------------------------------------------------------------
    # Test-set collection (collector mode 2)
    # ------------------------------------------------------------------
    test_episodes: int = 10
    test_episode_len: int = 1000
    test_reset_len: int = 100
    test_signal_type: str = "SMOOTH_NOISE"

    # ------------------------------------------------------------------
    # Safety limits, radians
    # ------------------------------------------------------------------
    hard_limit: float = 1.57         # 90 deg, the real mechanical limit
    abort_limit: float = 1.4         # 80 deg, hand over to the recovery controller
    planner_safe_limit: float = 1.2  # 69 deg, virtual wall used when shaping signals
    effort_to_pos_gain: float = 0.01 # rough effort -> position gain of the plant
    max_slew_rate: float = 500.0     # effort units per step
    lookahead: float = 0.2           # seconds of forward prediction before aborting

    # ------------------------------------------------------------------
    # Recovery / reset PD controller
    # ------------------------------------------------------------------
    reset_kp: float = 100.0
    reset_kd: float = 10.0
    max_effort: float = 200.0
    settled_pos_tol: float = 0.02
    settled_vel_tol: float = 0.05

    # ------------------------------------------------------------------
    # Gain calibration (collector mode 3)
    # ------------------------------------------------------------------
    calib_effort: float = 200.0
    calib_start_duration: float = 0.15  # seconds of the first effort pulse
    calib_duration_step: float = 0.05   # pulse lengthening per round
    calib_reset_duration: float = 5.0
    calib_limit: float = 3.0            # abort angle for motor and joints

    # ------------------------------------------------------------------
    # Visualisation (visualize_model.py)
    # ------------------------------------------------------------------
    viz_train_episodes: Tuple[int, ...] = (10, 100, 250, 400)
    viz_include_test: bool = True
    viz_test_start: int = 8800          # first row sliced out of the test CSV
    viz_test_len: Optional[int] = 1100  # None means "to the end of the file"
    viz_channels: Tuple[int, ...] = (0, 1, 2, 3)  # target_cols indices: all four channels

    # ------------------------------------------------------------------
    # Data inspection (plot_data.py)
    # ------------------------------------------------------------------
    inspect_file: str = "data/train.csv"
    inspect_episode_id: int = 0

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def seq_len(self) -> int:
        """Rows per episode in the CSV: excitation phase plus reset phase."""
        return self.episode_len + self.reset_len

    @property
    def input_dim(self) -> int:
        return len(self.input_cols)

    @property
    def output_dim(self) -> int:
        return len(self.target_cols)

    @property
    def dataset_name(self) -> str:
        return os.path.splitext(os.path.basename(self.data_file))[0]

    @property
    def model_save_path(self) -> str:
        return os.path.join(
            self.model_dir, f"{self.dataset_name}_dim{self.state_dim}_model.pth"
        )

    @property
    def scaler_save_path(self) -> str:
        return os.path.join(self.scaler_dir, f"{self.dataset_name}_scaler.json")

    @property
    def run_name(self) -> str:
        return f"{self.dataset_name}_dim{self.state_dim}"


DEFAULT_CONFIG = ExperimentConfig()
