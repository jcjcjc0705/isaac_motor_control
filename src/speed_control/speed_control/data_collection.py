"""Record the plant's response to designed excitation signals.

Publishes effort commands on /joint_command, records /joint_states, and writes
one CSV per session. Two layers keep the rig inside its travel limits: the
signal generators shape every waveform to a predicted excursion, and this node
hands the motor to a recovery controller if the rig approaches the limit
anyway.

Every /joint_states message is one control step, so the node runs on the
simulator's clock and the row count depends on messages received rather than
seconds elapsed. A slow simulator therefore costs wall time, not data.

Damping belongs to the scene, which applies it on the physics step; this node
publishes only the excitation command.

Usage:

    ros2 run speed_control data_collector

Mode 1 records a training set and mode 2 a test set, both asking for
effort_to_pos_gain at startup. Mode 3 measures that gain. The gain describes
the mechanism, so after changing links, masses or friction, run mode 3 first
and type its answer at the mode 1 or 2 prompt; the excitation amplitude follows
from the gain, so nothing else needs adjusting to stay inside the travel.
Everything else comes from config.py.
"""

import math
import os
from dataclasses import replace

import numpy as np
import pandas as pd
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .config import DEFAULT_CONFIG
from .joint_state import JointStateTracker, build_command
from .limit_test import GainCalibrationNode, clamp, is_settled, recovery_effort
from .node_runner import run_node
from .signals import build_test_schedule, build_training_schedule, render_signal, seed_everything

TRAIN_MODE = "train"
TEST_MODE = "test"


class DataCollectorNode(Node):
    """Step through a schedule of episodes, logging commands and responses."""

    def __init__(self, cfg=DEFAULT_CONFIG, mode: str = TRAIN_MODE):
        super().__init__("motor_data_collector")

        if mode == TEST_MODE:
            self.cfg = replace(
                cfg,
                total_episodes=cfg.test_episodes,
                episode_len=cfg.test_episode_len,
                reset_len=cfg.test_reset_len,
            )
            self.schedule = build_test_schedule(self.cfg)
            self.output_path = self.cfg.test_file
        else:
            self.cfg = cfg
            self.schedule = build_training_schedule(self.cfg)
            self.output_path = self.cfg.data_file

        seed_everything(self.cfg.seed)

        self.tracker = JointStateTracker()
        self.phase = "EXCITE"
        self.is_recovering = False
        self.episode_index = 0
        self.step_in_phase = 0
        self.global_step = 0
        self.limit_hit_count = 0
        self.message_count = 0
        self.settling = True        # hold until the rig is at rest
        self.settle_t0 = None
        self.sim_time_0 = None
        self.u_cmd = 0.0            # excitation command, held between recorded steps
        self.sim_time = 0.0

        self.current_signal = render_signal(self.cfg, self.schedule[0], self.cfg.episode_len)
        self.rows = []

        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 50)
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)

        print(f"Mode: {mode}")
        print(f"Episodes: {len(self.schedule)} x "
              f"({self.cfg.episode_len} excite + {self.cfg.reset_len} reset) steps")
        print(f"effort_to_pos_gain: {self.cfg.effort_to_pos_gain}")
        print(f"Control: every /joint_states message, 1 row per "
              f"{self.cfg.record_decimation} messages")
        # Absolute path, since config holds it relative to the working directory
        # and the node can be launched from anywhere.
        print(f"Output: {os.path.abspath(self.output_path)}")

    # ------------------------------------------------------------------
    # ROS plumbing
    # ------------------------------------------------------------------
    def send_effort(self, effort: float) -> None:
        self.publisher.publish(
            build_command(self.get_clock().now().to_msg(), effort)
        )

    def on_joint_states(self, msg: JointState) -> None:
        """One control step per message: the message stream is the node's clock."""
        if not self.tracker.update(msg):
            return
        if not self.tracker.is_finite():
            print("\nError: /joint_states went non-finite, the simulator has "
                  "diverged. Stop and Play in Isaac Sim, then re-run. "
                  f"Discarding {len(self.rows)} rows.")
            self.send_effort(0.0)
            raise SystemExit

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.settling:
            # Episode 0 starts from rest, because training rolls every episode
            # out from a zero initial state.
            if self.settle_t0 is None:
                self.settle_t0 = stamp
            self.send_effort(recovery_effort(self.tracker, self.cfg))
            if is_settled(self.tracker, self.cfg):
                self.settling = False
                print(f"Rig settled after {stamp - self.settle_t0:.1f}s, "
                      f"starting collection")
            elif stamp - self.settle_t0 > self.cfg.settle_timeout:
                print(f"\nError: could not bring the rig to rest within "
                      f"{self.cfg.settle_timeout:.0f}s. "
                      f"motor {self.tracker.motor_pos:+.3f} rad "
                      f"{self.tracker.motor_vel:+.3f} rad/s, "
                      f"joint1 {self.tracker.joint1_pos:+.3f} rad "
                      f"{self.tracker.joint1_vel:+.3f} rad/s.\n"
                      f"The rig coasts to a stop on the damping that "
                      f"isaac_scripts/actuator_model.py applies, so check that "
                      f"the scene has it attached and is playing. Without it "
                      f"nothing dissipates energy and the rig never settles.")
                self.send_effort(0.0)
                raise SystemExit
            return
        if self.sim_time_0 is None:
            self.sim_time_0 = stamp
        self.sim_time = stamp - self.sim_time_0
        if self.tracker.exceeds(self.cfg.hard_limit):
            self.limit_hit_count += 1

        self.message_count += 1
        if self.message_count % self.cfg.record_decimation == 0:
            self.on_step()              # advance the schedule and record one row
        else:
            self.send_effort(self.u_cmd)    # hold the command between rows

    # ------------------------------------------------------------------
    # Excitation loop
    # ------------------------------------------------------------------
    def should_abort(self) -> bool:
        """True once the rig is at, or predicted to reach, its travel limit."""
        motor_pos = self.tracker.motor_pos
        joint_pos = self.tracker.joint1_pos
        predicted_motor = motor_pos + self.tracker.motor_vel * self.cfg.lookahead
        predicted_joint = joint_pos + self.tracker.joint1_vel * self.cfg.lookahead
        return (abs(motor_pos) > self.cfg.abort_limit
                or abs(joint_pos) > self.cfg.abort_limit
                or abs(predicted_motor) > self.cfg.hard_limit
                or abs(predicted_joint) > self.cfg.hard_limit)

    def excitation_command(self) -> float:
        """Excitation command for one step, handing over to recovery if needed.

        Returns the command that the motor model will act on, or NaN to mean
        "recovery owns the actuator this step" so the caller bypasses the model.
        """
        if not self.is_recovering and self.should_abort():
            self.is_recovering = True

        if not self.is_recovering:
            if self.step_in_phase < len(self.current_signal):
                return float(self.current_signal[self.step_in_phase])
            return 0.0

        if is_settled(self.tracker, self.cfg):
            self.is_recovering = False
            return 0.0
        return float("nan")

    def advance_episode(self) -> None:
        self.episode_index += 1
        if self.episode_index < len(self.schedule):
            self.current_signal = render_signal(
                self.cfg, self.schedule[self.episode_index], self.cfg.episode_len
            )
        if self.episode_index % 10 == 0:
            print(f"Progress: {self.episode_index}/{self.cfg.total_episodes} episodes, "
                  f"{len(self.rows)} rows, {self.limit_hit_count} limit hits")

    def report_progress(self, episode: int, phase: str, step_in_phase: int) -> None:
        """Print where the command that just went out sits in the schedule.

        The position comes in as arguments rather than off self, which by this
        point may already have rolled over to the next phase or episode.
        """
        step_in_episode = step_in_phase
        if phase == "RESET":
            step_in_episode += self.cfg.episode_len
        print(f"Step {self.global_step}: "
              f"episode {episode}/{self.cfg.total_episodes}, "
              f"step {step_in_episode}/{self.cfg.seq_len} ({phase})")

    def on_step(self) -> None:
        """One recorded step: advance the schedule, command, and log a row."""
        if self.episode_index >= self.cfg.total_episodes:
            self.finish()
            return

        # Captured before the counters below move on.
        recorded_episode = self.episode_index
        recorded_phase = self.phase
        recorded_step = self.step_in_phase
        signal_type = self.schedule[recorded_episode].signal_type

        if self.phase == "EXCITE":
            command = self.excitation_command()
        else:
            command = (0.0 if is_settled(self.tracker, self.cfg) else float("nan"))

        if command != command:          # NaN: recovery owns the actuator
            self.send_effort(recovery_effort(self.tracker, self.cfg))
            self.u_cmd = 0.0
        else:
            self.u_cmd = command
            self.send_effort(command)

        self.step_in_phase += 1
        if self.phase == "EXCITE":
            if self.step_in_phase >= self.cfg.episode_len:
                self.phase = "RESET"
                self.step_in_phase = 0
                self.is_recovering = False
        else:
            if self.step_in_phase >= self.cfg.reset_len:
                self.phase = "EXCITE"
                self.step_in_phase = 0
                self.advance_episode()

        # effort_motor, the pose and the timestamp all come from the same
        # message, so the row describes one instant. input_u is the command
        # issued at that instant, which the joint will not feel for a few
        # messages yet.
        self.rows.append([
            self.sim_time,
            self.global_step * self.cfg.dt,
            recorded_episode,
            self.u_cmd,
            self.tracker.motor_effort,
            signal_type,
        ] + self.tracker.as_row())

        if self.global_step % self.cfg.progress_every == 0:
            self.report_progress(recorded_episode, recorded_phase, recorded_step)
        self.global_step += 1

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def finish(self) -> None:
        print("\nCollection complete, saving")
        self.send_effort(0.0)
        self.save()
        raise SystemExit

    ROW_COLUMNS = (
        "time_actual", "time_ideal", "episode_id",
        "input_u", "effort_motor", "signal_type",
        "pos_motor", "vel_motor", "pos_joint1", "vel_joint1",
        "pos_joint2", "vel_joint2",
    )

    def save(self) -> None:
        """Write the recorded rows to the CSV."""
        if not self.rows:
            print("Error: no joint states were received, nothing to save")
            return

        df = pd.DataFrame(self.rows, columns=list(self.ROW_COLUMNS))
        if not self.tracker.has_joint2:
            df = df.drop(columns=["pos_joint2", "vel_joint2"])
        df["episode_id"] = df["episode_id"].astype(int)

        position_cols = [c for c in df.columns if c.startswith("pos_")]
        state_cols = (position_cols
                      + [c for c in df.columns if c.startswith("vel_")]
                      + ["effort_motor"])
        # Non-finite rows are counted separately, since a comparison against
        # NaN is False and they never appear in the limit count.
        non_finite = ~df[state_cols].map(lambda v: pd.notna(v)).all(axis=1)
        exceeded = df[position_cols].abs().gt(self.cfg.hard_limit).any(axis=1)
        failed = df[exceeded]["episode_id"].unique()

        os.makedirs(self.cfg.data_dir, exist_ok=True)
        df.to_csv(self.output_path, index=False)

        print("")
        print(f"Saved {len(df)} rows to {os.path.abspath(self.output_path)}")
        if non_finite.any():
            print(f"  WARNING non-finite rows   : {int(non_finite.sum())} "
                  f"({non_finite.mean() * 100:.2f}%) in episodes "
                  f"{df[non_finite]['episode_id'].unique().tolist()}")
            print("  The simulator diverged; this dataset is not usable.")
        print(f"  Rows over {self.cfg.hard_limit} rad : "
              f"{int(exceeded.sum())} ({exceeded.mean() * 100:.2f}%)")
        if len(failed):
            print(f"  Episodes affected        : {failed.tolist()}")
        delay = command_delay(df, self.cfg.dt)
        if delay == delay:
            print(f"  Command delay            : {delay:.2f} steps "
                  f"({delay * self.cfg.dt * 1000:.0f} ms) between input_u and "
                  f"effort_motor")
        print("")


def command_delay(df: pd.DataFrame, dt: float) -> float:
    """Recorded steps between issuing a command and the joint receiving it.

    ``input_u`` is what was published and ``effort_motor`` what the simulator
    reports it applied, so sliding one against the other until they match reads
    the latency off directly, with no model of the plant involved. The search
    is over fractional steps because the round trip is quantised by the message
    rate rather than by the recording rate.

    The latency is fixed within a collection and can differ between them, since
    it depends on where the publishing falls relative to the simulator's tick.
    That is why the model is fitted to effort_motor: a dataset carries its own
    latency, and this number is what to compare when two of them are used
    together.
    """
    if "effort_motor" not in df.columns:
        return float("nan")
    lags = np.arange(0.0, 8.01, 0.05)
    measured = []
    for _, group in df.groupby("episode_id"):
        u = group["input_u"].to_numpy(dtype=float)
        applied = group["effort_motor"].to_numpy(dtype=float)
        if len(u) < 16 or np.allclose(u, 0.0):
            continue
        if not (np.isfinite(u).all() and np.isfinite(applied).all()):
            continue
        index = np.arange(len(u))
        residuals = [
            np.mean((applied - np.interp(index - lag, index, u)) ** 2)
            for lag in lags
        ]
        measured.append(float(lags[int(np.argmin(residuals))]))
        if len(measured) >= 10:
            break
    return float(np.median(measured)) if measured else float("nan")


def prompt_mode() -> str:
    print("")
    print("Select a task:")
    print("  1  Collect training data")
    print("  2  Collect test data")
    print("  3  Calibrate effort_to_pos_gain  (run this first after changing the mechanism)")
    print("All other settings come from config.py")
    return input("Choice [1/2/3]: ").strip()


def prompt_gain(cfg=DEFAULT_CONFIG) -> float:
    """Ask for effort_to_pos_gain, empty input keeping the configured value.

    Asked before the node is built, because the signal generators size every
    waveform against this gain as the schedule is laid out.
    """
    print("")
    print("Enter the gain mode 3 measured on this mechanism, or press Enter for "
          "the config default.")
    while True:
        raw = input(f"effort_to_pos_gain [{cfg.effort_to_pos_gain}]: ").strip()
        if not raw:
            return cfg.effort_to_pos_gain
        try:
            value = float(raw)
        except ValueError:
            print("  Not a number")
            continue
        # The amplitude derivation divides by the gain, and a negative gain
        # would point the predicted excursion the wrong way.
        if value <= 0:
            print("  Must be greater than zero")
            continue
        return value


def report_planned_travel(cfg) -> None:
    """Print the travel the entered gain implies, before any effort is published.

    An implausible gain shows up here as an implausible excursion, rather than
    part-way through a collection.
    """
    print("")
    print(f"  effort_to_pos_gain   : {cfg.effort_to_pos_gain:.3f} rad per unit effort")
    print(f"  plant_time_constant  : {cfg.plant_time_constant:.2f} s")
    print(f"  travel aimed at      : {cfg.safe_travel:.3f} rad "
          f"({math.degrees(cfg.safe_travel):.1f} deg) at full travel share, "
          f"against a {cfg.hard_limit:.3f} rad limit")
    print(f"  held effort for that : {cfg.held_effort_amplitude:.3f} effort units; "
          f"faster waveforms are given more, up to {cfg.max_effort:.2f}")


def main(args=None) -> None:
    choice = prompt_mode()

    if choice == "3":
        run_node(GainCalibrationNode, args)
    else:
        mode = TEST_MODE if choice == "2" else TRAIN_MODE
        cfg = replace(DEFAULT_CONFIG, effort_to_pos_gain=prompt_gain())
        report_planned_travel(cfg)
        run_node(lambda: DataCollectorNode(cfg=cfg, mode=mode), args)


if __name__ == "__main__":
    main()
