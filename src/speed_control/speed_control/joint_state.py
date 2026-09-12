"""Shared /joint_states parsing for the ROS nodes."""

from sensor_msgs.msg import JointState

MOTOR_NAMES = {"motor"}
JOINT1_NAMES = {"joint", "joint1"}
JOINT2_NAMES = {"joint2"}


class JointStateTracker:
    """Latest pose of the motor and the links it drives.

    The USD scenes differ in how many links they define, so ``has_joint2``
    records whether the second link was ever reported rather than being
    configured up front.
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
                self.motor_pos = msg.position[index]
                self.motor_vel = msg.velocity[index]
            elif name in JOINT1_NAMES:
                self.joint1_pos = msg.position[index]
                self.joint1_vel = msg.velocity[index]
            elif name in JOINT2_NAMES:
                self.joint2_pos = msg.position[index]
                self.joint2_vel = msg.velocity[index]
                self.has_joint2 = True
            else:
                continue
            matched = True
        return matched

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
