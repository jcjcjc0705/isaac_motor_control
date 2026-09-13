"""Central configuration for the motor dynamics identification pipeline.

Every tunable parameter lives here: data collection, model, training,
visualisation and data inspection. No script takes command-line flags, so the
way to change a setting is to edit this file and re-run. Both the ROS
collection side and the PyTorch training side import this module, which is what
keeps their episode layout, feature width and safety limits in agreement.

The dataclass is frozen. Build a variant with ``dataclasses.replace`` rather
than mutating ``DEFAULT_CONFIG``.
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
    dt: float = 0.05            # period of one recorded row, 20 Hz
    record_decimation: int = 3  # /joint_states messages per recorded row
    progress_every: int = 100   # recorded steps between collection progress lines

    # ------------------------------------------------------------------
    # CSV schema
    # ------------------------------------------------------------------
    # The model is fitted to the effort the simulator reports it applied, not
    # to the command that asked for it. The two differ by the latency of the
    # round trip, which is fixed within one collection but can land a message
    # or two apart between collections; effort_motor arrives in the same
    # message as the response to it, so it carries no such offset.
    input_cols: Tuple[str, ...] = ("effort_motor",)
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
    epochs: int = 400
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
    # Amplitude is not a setting: the amplitude property below derives it from
    # effort_to_pos_gain. signal_mix is the share of episodes each signal type
    # gets, and the shares are normalised by episode count, not by weight.
    signal_mix: Tuple[Tuple[str, float], ...] = (
        ("RAMPS", 0.2),
        ("PRBS", 0.2),
        ("CHIRP", 0.2),
        ("MULTISINE", 0.4),
    )
    # Per-type parameter range. Frequency in Hz, except PRBS which is hold
    # steps. MULTISINE ignores this and draws from its own bands.
    signal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "RAMPS": (0.1, 1.5),
        "PRBS": (20, 60),
        "CHIRP": (0.1, 2.0),
        "MULTISINE": (0.1, 3.0),
        "SMOOTH_NOISE": (0.1, 2.0),
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
    hard_limit: float = 1.57         # 90 deg, the mechanical travel limit
    abort_limit: float = 1.4         # 80 deg, hands the motor to the recovery controller
    planner_safe_limit: float = 1.2  # 69 deg, the angle signals are shaped to fit
    planner_margin: float = 0.05     # headroom kept below planner_safe_limit
    # Band on the planner's estimate of position plus lookahead velocity, the
    # quantity the collector aborts on when it passes hard_limit. It sits below
    # hard_limit because the estimate carries the same spread as the position
    # one. Only fast waveforms are held back by it; a slow one is bounded by
    # planner_safe_limit first.
    planner_lookahead_limit: float = 1.35
    # Radians of excursion per unit of held effort, measured by collector mode
    # 3. It describes the mechanism -- links, masses, friction -- so it has to
    # be re-measured whenever any of those change, and it is what modes 1 and 2
    # prompt for at startup. The value here is the default the prompt offers on
    # Enter.
    #
    # Use the peak gain mode 3 reports rather than the equilibrium gain: the
    # signal generators ask how far a waveform throws a joint, which includes
    # the overshoot on the way to the equilibrium angle.
    effort_to_pos_gain: float = 1.15
    # Seconds. The plant reaches the angle above only if the effort stays put;
    # a command that reverses sooner does not get that far. The peak predictor
    # low-passes the command with this time constant before scaling it by the
    # gain, which leaves a held effort at the full gain and attenuates a fast
    # alternating one. Measured by comparing the prediction against the
    # excursions a collected dataset actually reached.
    plant_time_constant: float = 0.3
    max_slew_rate: float = 500.0     # largest step-to-step change in the command
    lookahead: float = 0.2           # seconds of forward prediction before aborting

    # ------------------------------------------------------------------
    # Recovery controller, used during the reset phase and after an abort
    # ------------------------------------------------------------------
    # Both gains at zero means the recovery publishes no effort and lets the
    # joints coast to a stop on their own friction, which is what a rig with dry
    # friction does. Non-zero gains turn the reset into a PD loop closed through
    # the round trip to the simulator, so raise them only if the mechanism will
    # not settle on its own, and expect to re-tune them per mechanism.
    reset_kp: float = 0.0
    reset_kd: float = 0.0
    max_effort: float = 2.5         # ceiling on any effort the nodes publish
    settled_pos_tol: float = 0.02   # radians from zero that counts as settled
    settled_vel_tol: float = 0.05   # rad/s that counts as stopped
    settle_timeout: float = 20.0    # seconds of simulator time before giving up

    # ------------------------------------------------------------------
    # Gain calibration (collector mode 3)
    # ------------------------------------------------------------------
    # Mode 3 holds a constant effort until the joints stop moving, reads the
    # angle gravity balances it at, then repeats one step higher until a joint
    # reaches calib_limit. The slope of angle against effort is the gain it
    # reports.
    calib_start_effort: float = 0.25    # first effort held
    calib_effort_step: float = 0.25     # added each round
    calib_effort: float = 2.0           # highest effort to try
    calib_hold_duration: float = 8.0    # seconds allowed to reach equilibrium
    calib_reset_duration: float = 8.0   # seconds allowed to return to rest
    calib_limit: float = 1.2            # abort angle for motor and joints

    # ------------------------------------------------------------------
    # Visualisation (visualize_model.py)
    # ------------------------------------------------------------------
    viz_train_episodes: Tuple[int, ...] = (10, 100, 250, 400)
    viz_include_test: bool = True
    viz_test_start: int = 8800          # first row sliced out of the test CSV
    viz_test_len: Optional[int] = 1100  # None means "to the end of the file"
    viz_channels: Tuple[int, ...] = (0, 1, 2, 3)  # which target_cols to plot

    # ------------------------------------------------------------------
    # Data inspection (plot_data.py)
    # ------------------------------------------------------------------
    # plot_data.py reads every CSV under data_dir; this picks the episode whose
    # waveforms the per-episode panels show, in each file.
    inspect_episode_id: int = 0

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def seq_len(self) -> int:
        """Rows per episode in the CSV: excitation phase plus reset phase."""
        return self.episode_len + self.reset_len

    @property
    def safe_travel(self) -> float:
        """Radians of excursion an episode at full travel share is aimed at."""
        return self.planner_safe_limit - self.planner_margin

    @property
    def held_effort_amplitude(self) -> float:
        """The effort that fills the safe band when simply held, in effort units.

        Only the slowest waveforms are scaled to about this much; a fast one is
        given more, since it reverses before the joint has travelled that far.
        It is the reference the startup report prints, and it follows
        effort_to_pos_gain, which is why entering a freshly measured gain is
        enough to fit a collection to a new mechanism.
        """
        return self.safe_travel / self.effort_to_pos_gain

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
