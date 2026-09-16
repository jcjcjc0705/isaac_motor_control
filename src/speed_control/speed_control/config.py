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
    # Every output of a run -- datasets, checkpoint, scaler, plots and
    # TensorBoard logs -- is written under results_root/experiment. Give each
    # configuration its own experiment name and nothing a previous run wrote is
    # overwritten. The directories below are derived from it.
    experiment: str = "main-60-30"
    results_root: str = "results"

    # ------------------------------------------------------------------
    # Episode layout, shared by collection and training
    # ------------------------------------------------------------------
    # One recorded row is one physics step, one control period, and the dead
    # time: a command issued at row k reaches the joint for the state recorded
    # at row k+1. This requires the scene's graph to tick at twice the physics
    # rate, 60 Hz over 30 Hz. Messages then arrive at the tick rate with every
    # second one repeating a physics step, and the collector drops those.
    episode_len: int = 300      # excitation steps per episode, 10 s
    reset_len: int = 150        # settling steps, 5 s
    total_episodes: int = 500
    dt: float = 1.0 / 30.0      # period of one recorded row, 30 Hz
    record_decimation: int = 1  # distinct physics steps per recorded row
    progress_every: int = 100   # recorded steps between collection progress lines

    # ------------------------------------------------------------------
    # CSV schema
    # ------------------------------------------------------------------
    # The model is fitted to the efforts the simulator reports it applied,
    # which arrive in the same message as the response to them. Both joints
    # appear as both are commanded, so the input is the whole torque the
    # mechanism received and the plant learned is the undamped mechanism.
    input_cols: Tuple[str, ...] = ("effort_motor", "effort_joint1")
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
    test_reset_len: int = 150
    test_signal_type: str = "SMOOTH_NOISE"

    # ------------------------------------------------------------------
    # Safety limits, radians
    # ------------------------------------------------------------------
    hard_limit: float = 1.57         # 90 deg, the mechanical travel limit
    abort_limit: float = 1.4         # 80 deg, hands the motor to the recovery controller
    planner_safe_limit: float = 1.2  # 69 deg, the angle signals are shaped to fit
    planner_margin: float = 0.05     # headroom kept below planner_safe_limit
    # Band on the planner's estimate of position plus lookahead velocity, the
    # quantity the collector aborts on. Only fast waveforms are held back by
    # it; a slow one is bounded by planner_safe_limit first.
    planner_lookahead_limit: float = 1.35
    # Radians of excursion per unit of held effort, from collector mode 3, and
    # what modes 1 and 2 prompt for at startup. Re-measure after any change to
    # the links, masses or friction. Use the peak gain mode 3 reports, not the
    # equilibrium one.
    effort_to_pos_gain: float = 1.013
    # Seconds. The peak predictor low-passes the command with this time
    # constant before scaling it by the gain. Calibrate it by comparing the
    # predicted excursions against those a collected dataset reached.
    plant_time_constant: float = 0.3
    max_slew_rate: float = 500.0     # largest step-to-step change in the command
    lookahead: float = 0.2           # seconds of forward prediction before aborting
    # Steps over which the excitation is eased in from zero by a raised-cosine
    # window, applied at the start of every episode and whenever the recovery
    # controller hands the actuator back. Keep the window several times longer
    # than one period of the highest frequency the physics step can carry.
    fade_in_steps: int = 12          # 0.4 s at 30 Hz; 0 disables the fade

    # ------------------------------------------------------------------
    # Actuator model (actuator.py), applied to every published command
    # ------------------------------------------------------------------
    # Friction the joints apply to themselves. The scene carries none of its
    # own, so these are the whole of it; raise them to make a mechanism that
    # will not settle settle.
    #
    # The ceiling is a property of the pair, not of either joint: doubling both
    # makes the rig ring at half the physics step rate and never come to rest,
    # while doubling either one alone does not. After changing them, or the
    # physics step, kick the rig and check that the motion decays.
    b_viscous_motor: float = 0.15     # N*m*s/rad
    b_coulomb_motor: float = 0.0      # N*m, speed independent
    b_viscous_joint1: float = 0.025
    b_coulomb_joint1: float = 0.0
    max_damping_torque: float = 2.0   # ceiling on either joint's friction
    # Each joint's effective inertia, from the velocity one step of a known
    # torque produces. The actuator model scales the viscous term with it; too
    # small only makes the damping gentler, too large lets it ring.
    inertia_motor: float = 0.022      # kg*m^2
    inertia_joint1: float = 0.004

    # ------------------------------------------------------------------
    # Recovery controller, used during the reset phase and after an abort
    # ------------------------------------------------------------------
    # Both gains at zero means the recovery publishes no excitation and lets
    # the actuator model's friction stop the joints. Non-zero gains close a PD
    # loop through the round trip to the simulator and have to be re-tuned per
    # mechanism.
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
    viz_test_start: int = 9200          # first row sliced out of the test CSV
    viz_test_len: Optional[int] = 1150  # None means "to the end of the file"
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
        given more. It is the reference the startup report prints.
        """
        return self.safe_travel / self.effort_to_pos_gain

    @property
    def input_dim(self) -> int:
        return len(self.input_cols)

    @property
    def output_dim(self) -> int:
        return len(self.target_cols)

    @property
    def experiment_dir(self) -> str:
        """Directory holding everything this configuration produces."""
        return os.path.join(self.results_root, self.experiment)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.experiment_dir, "data")

    @property
    def model_dir(self) -> str:
        return os.path.join(self.experiment_dir, "models")

    @property
    def scaler_dir(self) -> str:
        return os.path.join(self.experiment_dir, "scalers")

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.experiment_dir, "plots")

    @property
    def tensorboard_dir(self) -> str:
        return os.path.join(self.experiment_dir, "runs")

    @property
    def data_file(self) -> str:
        return os.path.join(self.data_dir, "train.csv")

    @property
    def test_file(self) -> str:
        return os.path.join(self.data_dir, "test.csv")

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
