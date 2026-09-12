"""Shared entry-point plumbing for the ROS nodes."""

import rclpy


def run_node(factory, args=None) -> None:
    """Construct a node, spin it, and shut down cleanly.

    A node ends its own run by raising SystemExit from a callback. The shutdown
    is guarded so it is safe whether or not the rclpy context is still up.
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
