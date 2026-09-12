"""Shared /joint_states parsing and /joint_command building for the ROS nodes."""

import math

from sensor_msgs.msg import JointState

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_command(stamp, effort: float) -> JointState:
    """Effort command for the motor, as published on /joint_command."""
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = ["motor"]
    msg.effort = [float(effort)]
    return msg


MOTOR_NAMES = {"motor"}
JOINT1_NAMES = {"joint", "joint1"}
JOINT2_NAMES = {"joint2"}


class JointStateTracker:
    """Latest pose of the motor and the links it drives.

    Positions are wrapped to +/-pi as they arrive, so every consumer sees the
    same angle. ``has_joint2`` reports whether the scene published a second
    link, which is detected from the messages rather than configured.
    """

    def __init__(self):
        self.motor_pos = 0.0
        self.motor_vel = 0.0
        self.joint1_pos = 0.0
        self.joint1_vel = 0.0
        self.joint2_pos = 0.0
        self.joint2_vel = 0.0
        self.has_joint2 = False

    def update(self, msg: JointState) -> bool:
        """Absorb one message. Returns True if it named a tracked joint."""
        matched = False
        for index, raw_name in enumerate(msg.name):
            name = raw_name.lower()
            if name in MOTOR_NAMES:
                self.motor_pos = wrap_to_pi(msg.position[index])
                self.motor_vel = msg.velocity[index]
            elif name in JOINT1_NAMES:
                self.joint1_pos = wrap_to_pi(msg.position[index])
                self.joint1_vel = msg.velocity[index]
            elif name in JOINT2_NAMES:
                self.joint2_pos = wrap_to_pi(msg.position[index])
                self.joint2_vel = msg.velocity[index]
                self.has_joint2 = True
            else:
                continue
            matched = True
        return matched

    def is_finite(self) -> bool:
        """False once the simulator has diverged and is publishing NaN."""
        return all(math.isfinite(v) for v in self.as_row())

    def max_abs_position(self) -> float:
        positions = [abs(self.motor_pos), abs(self.joint1_pos)]
        if self.has_joint2:
            positions.append(abs(self.joint2_pos))
        return max(positions)

    def exceeds(self, limit: float) -> bool:
        return self.max_abs_position() > limit

    def as_row(self) -> list:
        return [
            self.motor_pos, self.motor_vel,
            self.joint1_pos, self.joint1_vel,
            self.joint2_pos, self.joint2_vel,
        ]

    def summary(self) -> str:
        text = (f"motor {self.motor_pos:+.2f}/{self.motor_vel:+.2f} "
                f"joint1 {self.joint1_pos:+.2f}/{self.joint1_vel:+.2f}")
        if self.has_joint2:
            text += f" joint2 {self.joint2_pos:+.2f}/{self.joint2_vel:+.2f}"
        return text
