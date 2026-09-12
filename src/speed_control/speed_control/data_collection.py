"""Record the plant's response to designed excitation signals.

Publishes effort commands on /joint_command, records /joint_states, and writes
one CSV per session. Two layers of protection keep the rig inside its travel
limits: the signal generators shape waveforms against a predicted drift, and
this node aborts to a PD recovery controller if the real hardware still gets
close to the wall.

All parameters come from config.py. The only runtime choice is which dataset to
record, which is asked interactively at startup.
"""

import os
from dataclasses import replace

import numpy as np
import pandas as pd
from rclpy.node import Node
from scipy.interpolate import interp1d
from sensor_msgs.msg import JointState

from .config import DEFAULT_CONFIG
from .joint_state import JointStateTracker
from .limit_test import GainCalibrationNode, is_settled, recovery_effort
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

        self.current_signal = render_signal(self.cfg, self.schedule[0], self.cfg.episode_len)
        self.command_log = []
        self.state_log = []

        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)
        self.create_timer(self.cfg.dt, self.on_timer)
        self.start_time_ns = self.get_clock().now().nanoseconds

        print(f"Mode: {mode}")
        print(f"Episodes: {len(self.schedule)} x "
              f"({self.cfg.episode_len} excite + {self.cfg.reset_len} reset) steps")
        # Absolute, because the paths in config are relative to the working
        # directory and this node is often launched from elsewhere.
        print(f"Output: {os.path.abspath(self.output_path)}")

    # ------------------------------------------------------------------
    # ROS plumbing
    # ------------------------------------------------------------------
    def elapsed_seconds(self) -> float:
        return (self.get_clock().now().nanoseconds - self.start_time_ns) / 1e9

    def send_effort(self, effort: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["motor"]
        msg.effort = [float(effort)]
        self.publisher.publish(msg)

    def on_joint_states(self, msg: JointState) -> None:
        if not self.tracker.update(msg):
            return
        if self.tracker.exceeds(self.cfg.hard_limit):
            self.limit_hit_count += 1
        self.state_log.append([self.elapsed_seconds()] + self.tracker.as_row())

    # ------------------------------------------------------------------
    # Excitation loop
    # ------------------------------------------------------------------
    def should_abort(self) -> bool:
        """True once the rig is at, or predicted to reach, its travel limit."""
        predicted_motor = self.tracker.motor_pos + self.tracker.motor_vel * self.cfg.lookahead
        predicted_joint = self.tracker.joint1_pos + self.tracker.joint1_vel * self.cfg.lookahead
        return (abs(self.tracker.motor_pos) > self.cfg.abort_limit
                or abs(self.tracker.joint1_pos) > self.cfg.abort_limit
                or abs(predicted_motor) > self.cfg.hard_limit
                or abs(predicted_joint) > self.cfg.hard_limit)

    def excitation_effort(self) -> float:
        """Effort for one excitation step, handing over to recovery if needed."""
        if not self.is_recovering and self.should_abort():
            self.is_recovering = True

        if not self.is_recovering:
            if self.step_in_phase < len(self.current_signal):
                return float(self.current_signal[self.step_in_phase])
            return 0.0

        if is_settled(self.tracker, self.cfg):
            self.is_recovering = False
            return 0.0
        return recovery_effort(self.tracker, self.cfg)

    def reset_effort(self) -> float:
        if is_settled(self.tracker, self.cfg):
            return 0.0
        return recovery_effort(self.tracker, self.cfg)

    def advance_episode(self) -> None:
        self.episode_index += 1
        if self.episode_index < len(self.schedule):
            self.current_signal = render_signal(
                self.cfg, self.schedule[self.episode_index], self.cfg.episode_len
            )
        if self.episode_index % 10 == 0:
            print(f"Progress: {self.episode_index}/{self.cfg.total_episodes} episodes, "
                  f"{len(self.command_log)} rows, {self.limit_hit_count} limit hits")

    def on_timer(self) -> None:
        if self.episode_index >= self.cfg.total_episodes:
            self.finish()
            return

        recorded_episode = self.episode_index
        signal_type = self.schedule[recorded_episode].signal_type

        if self.phase == "EXCITE":
            effort = self.excitation_effort()
            self.send_effort(effort)
            self.step_in_phase += 1
            if self.step_in_phase >= self.cfg.episode_len:
                self.phase = "RESET"
                self.step_in_phase = 0
                self.is_recovering = False
        else:
            effort = self.reset_effort()
            self.send_effort(effort)
            self.step_in_phase += 1
            if self.step_in_phase >= self.cfg.reset_len:
                self.phase = "EXCITE"
                self.step_in_phase = 0
                self.advance_episode()

        self.command_log.append([
            self.elapsed_seconds(),
            self.global_step * self.cfg.dt,
            recorded_episode,
            effort,
            signal_type,
        ])
        self.global_step += 1

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def finish(self) -> None:
        print("\nCollection complete, aligning and saving")
        self.send_effort(0.0)
        self.save()
        raise SystemExit

    def save(self) -> None:
        """Resample states onto the command clock and write the CSV.

        /joint_states arrives on the simulator's clock, not the command timer's,
        so the two streams are interpolated onto a common time base before they
        can share a row.
        """
        if not self.state_log:
            print("Error: no joint states were received, nothing to save")
            return

        commands = np.array(self.command_log, dtype=object)
        states = np.array(self.state_log, dtype=float)

        state_times, unique_index = np.unique(states[:, 0], return_index=True)
        state_values = states[unique_index, 1:]

        command_times = commands[:, 0].astype(float)
        aligned = interp1d(
            state_times, state_values, axis=0, kind="linear", fill_value="extrapolate"
        )(command_times)

        frame = {
            "time_actual": command_times,
            "time_ideal": commands[:, 1].astype(float),
            "episode_id": commands[:, 2].astype(int),
            "input_u": commands[:, 3].astype(float),
            "signal_type": commands[:, 4].astype(str),
            "pos_motor": aligned[:, 0],
            "vel_motor": aligned[:, 1],
            "pos_joint1": aligned[:, 2],
            "vel_joint1": aligned[:, 3],
        }
        if self.tracker.has_joint2:
            frame["pos_joint2"] = aligned[:, 4]
            frame["vel_joint2"] = aligned[:, 5]

        df = pd.DataFrame(frame)

        exceeded = (df["pos_motor"].abs() > self.cfg.hard_limit) | \
                   (df["pos_joint1"].abs() > self.cfg.hard_limit)
        if self.tracker.has_joint2:
            exceeded |= df["pos_joint2"].abs() > self.cfg.hard_limit
        failed = df[exceeded]["episode_id"].unique().astype(int)

        os.makedirs(self.cfg.data_dir, exist_ok=True)
        df.to_csv(self.output_path, index=False)

        print("")
        print(f"Saved {len(df)} rows to {os.path.abspath(self.output_path)}")
        print(f"  Rows over {self.cfg.hard_limit} rad : "
              f"{int(exceeded.sum())} ({exceeded.mean() * 100:.2f}%)")
        if len(failed):
            print(f"  Episodes affected        : {failed.tolist()}")
        print("")


def prompt_mode() -> str:
    print("")
    print("Select a task:")
    print("  1  Collect training data")
    print("  2  Collect test data")
    print("  3  Calibrate effort_to_pos_gain")
    print("All other settings come from config.py")
    return input("Choice [1/2/3]: ").strip()


def main(args=None) -> None:
    choice = prompt_mode()

    if choice == "3":
        run_node(GainCalibrationNode, args)
    else:
        mode = TEST_MODE if choice == "2" else TRAIN_MODE
        run_node(lambda: DataCollectorNode(mode=mode), args)


if __name__ == "__main__":
    main()
