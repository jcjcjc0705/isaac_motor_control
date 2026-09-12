"""Estimate the plant's effort-to-position gain by probing its travel limit.

Drives the motor with a fixed effort pulse whose duration grows each round,
returning to zero in between, until some joint passes ``cfg.calib_limit``. The
gain is then back-computed from the last pulse that did not hit the wall.

Feed the printed value into ``effort_to_pos_gain`` in config.py so the signal
generators can shape waveforms that stay inside the safe band.
"""

from rclpy.node import Node
from sensor_msgs.msg import JointState

from .config import DEFAULT_CONFIG
from .joint_state import JointStateTracker
from .node_runner import run_node


class GainCalibrationNode(Node):
    def __init__(self, cfg=DEFAULT_CONFIG):
        super().__init__("gain_calibration")
        self.cfg = cfg
        self.tracker = JointStateTracker()

        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)
        self.create_timer(cfg.dt, self.on_timer)

        self.phase = "DRIVE_POSITIVE"
        self.pulse_duration = cfg.calib_start_duration
        self.phase_elapsed = 0.0
        self.estimated_gain = 5.0

        self.get_logger().info(
            f"Gain calibration started, abort limit {cfg.calib_limit} rad"
        )

    def send_effort(self, effort: float) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["motor"]
        msg.effort = [float(effort)]
        self.publisher.publish(msg)

    def on_joint_states(self, msg: JointState) -> None:
        if self.tracker.update(msg):
            print(f"[state] {self.tracker.summary()}", end="\r")

    def report_and_stop(self) -> None:
        """Back-compute the gain from the last pulse that stayed inside the limit."""
        safe_duration = max(self.pulse_duration - self.cfg.dt, self.cfg.dt)
        ideal_distance = 0.5 * self.cfg.calib_effort * (safe_duration ** 2)
        self.estimated_gain = self.cfg.calib_limit / ideal_distance

        print("")
        print("Gain calibration result")
        print(f"  Last safe pulse    : {safe_duration:.2f} s")
        print(f"  effort_to_pos_gain : {self.estimated_gain:.3f}")
        print("  Copy this value into config.py")
        print("")

        self.send_effort(0.0)
        raise SystemExit

    def on_timer(self) -> None:
        if self.tracker.exceeds(self.cfg.calib_limit):
            self.report_and_stop()
            return

        if self.phase == "DRIVE_POSITIVE":
            if self.phase_elapsed < self.pulse_duration:
                self.send_effort(self.cfg.calib_effort)
                self.phase_elapsed += self.cfg.dt
            else:
                self.phase = "DRIVE_NEGATIVE"
                self.phase_elapsed = 0.0

        elif self.phase == "DRIVE_NEGATIVE":
            if self.phase_elapsed < self.pulse_duration:
                self.send_effort(-self.cfg.calib_effort)
                self.phase_elapsed += self.cfg.dt
            else:
                self.phase = "RECOVER"
                self.phase_elapsed = 0.0
                self.send_effort(0.0)
                print(f"\nPassed {self.pulse_duration:.2f} s "
                      f"(max {self.tracker.max_abs_position():.2f} rad), recovering")

        elif self.phase == "RECOVER":
            self.send_effort(recovery_effort(self.tracker, self.cfg))
            if self.phase_elapsed < self.cfg.calib_reset_duration:
                self.phase_elapsed += self.cfg.dt
                if is_settled(self.tracker, self.cfg) and self.phase_elapsed > 0.5:
                    self.phase_elapsed = self.cfg.calib_reset_duration
            else:
                self.phase = "DRIVE_POSITIVE"
                self.phase_elapsed = 0.0
                self.pulse_duration += self.cfg.calib_duration_step


def recovery_effort(tracker: JointStateTracker, cfg) -> float:
    """PD effort that drives the motor back to zero, clamped to max_effort."""
    effort = -cfg.reset_kp * tracker.motor_pos - cfg.reset_kd * tracker.motor_vel
    return max(-cfg.max_effort, min(cfg.max_effort, effort))


def is_settled(tracker: JointStateTracker, cfg) -> bool:
    return (abs(tracker.motor_pos) < cfg.settled_pos_tol
            and abs(tracker.motor_vel) < cfg.settled_vel_tol)


def main(args=None) -> None:
    run_node(GainCalibrationNode, args)


if __name__ == "__main__":
    main()
