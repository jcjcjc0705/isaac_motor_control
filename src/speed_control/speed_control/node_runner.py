"""Shared entry-point plumbing for the ROS nodes."""

import rclpy


def run_node(factory, args=None) -> None:
    """Construct a node, spin it, and shut down cleanly.

    Both nodes stop themselves by raising SystemExit from a timer callback.
    The shutdown is guarded: rclpy raises if the context is already down, and
    which paths leave it down differs between ROS 2 releases.
    """
    rclpy.init(args=args)
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
