"""Measure the plant's effort-to-position gain from its equilibrium angles.

Holds an effort until the joints stop, reads the angle gravity balances it at,
and repeats one step higher until a joint reaches ``cfg.calib_limit``. Both
joints are measured and the larger angle wins. It reports an equilibrium gain
and a peak gain; ``effort_to_pos_gain`` takes the peak one.

Usage: collector mode 3, ``ros2 run speed_control data_collector``. Run it
after any change to the mechanism and type its answer at the mode 1 or 2
prompt. Settings come from the calib_* block in config.py.
"""

import math

from rclpy.node import Node
from sensor_msgs.msg import JointState

from .config import DEFAULT_CONFIG
from .actuator import actuator_efforts
from .joint_state import JointStateTracker, build_command
from .node_runner import run_node


class GainCalibrationNode(Node):
    """Step the effort up, holding each level until the joints stop moving."""

    def __init__(self, cfg=DEFAULT_CONFIG):
        super().__init__("gain_calibration")
        self.cfg = cfg
        self.tracker = JointStateTracker()

        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 50)
        self.publisher = self.create_publisher(JointState, "/joint_command", 10)

        self.last_stamp = None      # skips a tick that carried no new physics
        self.effort = cfg.calib_start_effort
        self.samples = []           # (effort, equilibrium angle, peak angle)
        self.phase = "RECOVER"
        self.phase_t0 = None
        self.peak = 0.0

        print("")
        print(f"Gain calibration: holding {cfg.calib_start_effort} to "
              f"{cfg.calib_effort} in steps of {cfg.calib_effort_step}, "
              f"abort at {cfg.calib_limit} rad")
        print("All other settings come from config.py")
        print("")

    def send_effort(self, effort: float) -> None:
        """Publish the held effort with each joint's friction added to it."""
        self.publisher.publish(build_command(
            self.get_clock().now().to_msg(),
            actuator_efforts(self.tracker, self.cfg, effort),
        ))

    def at_rest(self) -> bool:
        """Moving slowly enough to call the current angle an equilibrium."""
        return (abs(self.tracker.motor_vel) < self.cfg.settled_vel_tol
                and abs(self.tracker.joint1_vel) < self.cfg.settled_vel_tol)

    def on_joint_states(self, msg: JointState) -> None:
        """One step per message, so the node runs on the simulator's clock."""
        if not self.tracker.update(msg):
            return
        if not self.tracker.is_finite():
            print("\nError: /joint_states went non-finite, the simulator has "
                  "diverged. Stop and Play in Isaac Sim, then re-run.")
            self.send_effort(0.0)
            raise SystemExit

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp == self.last_stamp:    # a tick that carried no new physics
            return
        self.last_stamp = stamp
        if self.phase_t0 is None:
            self.phase_t0 = stamp
        elapsed = stamp - self.phase_t0
        angle = self.tracker.max_abs_position()

        if angle > self.cfg.calib_limit:
            print(f"\n{self.effort:.3f} reached {angle:.3f} rad, past the "
                  f"{self.cfg.calib_limit} rad abort angle")
            self.report_and_stop()
            return

        if self.phase == "RECOVER":
            self.send_effort(0.0)
            if elapsed > self.cfg.calib_reset_duration or (
                    elapsed > 1.0 and is_settled(self.tracker, self.cfg)):
                self.phase, self.phase_t0, self.peak = "HOLD", stamp, 0.0
            return

        self.send_effort(self.effort)
        self.peak = max(self.peak, angle)
        print(f"[hold {self.effort:6.3f}] {self.tracker.summary()}", end="\r")
        settled = elapsed > 1.0 and self.at_rest()
        if not settled and elapsed < self.cfg.calib_hold_duration:
            return

        if settled:
            self.samples.append((self.effort, angle, self.peak))
            print(f"\n  effort {self.effort:6.3f}  ->  equilibrium "
                  f"{angle:.4f} rad, peak {self.peak:.4f} rad")
        else:
            print(f"\n  effort {self.effort:6.3f}  ->  still moving after "
                  f"{self.cfg.calib_hold_duration:.0f}s, not recorded")

        self.effort += self.cfg.calib_effort_step
        self.phase, self.phase_t0 = "RECOVER", stamp
        if self.effort > self.cfg.calib_effort + 1e-9:
            self.report_and_stop()

    def report_and_stop(self) -> None:
        """Fit angle against effort and print the slope."""
        self.send_effort(0.0)
        print("")
        print("Gain calibration result")
        if len(self.samples) < 2:
            print(f"  Only {len(self.samples)} usable level(s): nothing to fit.")
            print("  Lower calib_start_effort, or raise calib_limit if the rig "
                  "has more travel than the abort angle allows.")
            print("")
            raise SystemExit

        efforts = [s[0] for s in self.samples]
        equilibria = [s[1] for s in self.samples]
        peaks = [s[2] for s in self.samples]
        equilibrium_gain = _slope(efforts, equilibria)
        peak_gain = _slope(efforts, peaks)

        print(f"  Levels held        : {len(self.samples)} "
              f"({efforts[0]:.3f} to {efforts[-1]:.3f})")
        print(f"  Equilibrium gain   : {equilibrium_gain:.3f} rad per unit effort")
        print(f"  Peak gain          : {peak_gain:.3f} rad per unit effort")
        print(f"  Overshoot          : {peak_gain / equilibrium_gain:.2f}x"
              if equilibrium_gain > 1e-9 else "  Overshoot          : n/a")
        print("")
        print(f"  effort_to_pos_gain : {peak_gain:.3f}")
        print("  Type this at the effort_to_pos_gain prompt in collector mode 1 "
              "or 2.")
        print("")

        amplitude = (self.cfg.planner_safe_limit - self.cfg.planner_margin) / peak_gain
        print(f"  With it, excitation runs at |effort| <= {amplitude:.3f} and is "
              f"predicted to")
        print(f"  reach {math.degrees(amplitude * peak_gain):.1f} deg, inside the "
              f"{math.degrees(self.cfg.hard_limit):.0f} deg travel. Put it in "
              f"config.py as")
        print("  the new default once the mechanism is settled.")
        print("")
        raise SystemExit


def _slope(x, y) -> float:
    """Least-squares slope through the origin, which the plant passes through."""
    denominator = sum(v * v for v in x)
    if denominator < 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(x, y)) / denominator


def clamp(value: float, limit: float) -> float:
    """Clamp to +/-limit, mapping a non-finite value to zero.

    ``min``/``max`` return the bound when handed a NaN, so it is tested for.
    """
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def recovery_effort(tracker: JointStateTracker, cfg) -> float:
    """PD effort that drives the motor back to zero, clamped to max_effort."""
    effort = -cfg.reset_kp * tracker.motor_pos - cfg.reset_kd * tracker.motor_vel
    return clamp(effort, cfg.max_effort)


def is_settled(tracker: JointStateTracker, cfg) -> bool:
    """Every joint near zero and stopped."""
    pairs = (
        (tracker.motor_pos, tracker.motor_vel),
        (tracker.joint1_pos, tracker.joint1_vel),
        (tracker.joint2_pos, tracker.joint2_vel),
    )
    return all(abs(pos) < cfg.settled_pos_tol and abs(vel) < cfg.settled_vel_tol
               for pos, vel in pairs)


def main(args=None) -> None:
    run_node(GainCalibrationNode, args)


if __name__ == "__main__":
    main()
