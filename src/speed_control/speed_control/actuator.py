"""Actuator model: the friction each joint applies to itself.

The simulator runs the mechanism without damping and this supplies it, so the
excitation and the friction are published together as one command per joint.

Both joints are commanded. No excitation is sent to the link's hinge, but its
friction reaches it through the same channel, since /joint_command is the only
way in.

Coefficients live in config.py and are bounded by the physics step: raising
them past what the step supports makes the rig ring at half the step rate and
never settle. After changing either the coefficients or the step, kick the rig
and check that the motion decays.
"""

import math


def damping_torque(velocity: float, viscous: float, coulomb: float,
                   limit: float, step: float, inertia: float) -> float:
    """Friction torque opposing a joint's motion.

    The viscous term is scaled by ``1 / (1 + viscous * step / inertia)``, which
    is what an implicit step of the same decay applies; pass ``inertia`` as 0
    to leave it unscaled. The total is capped at ``limit``, and a non-finite
    velocity yields zero.
    """
    if not math.isfinite(velocity):
        return 0.0
    effective = viscous
    if inertia > 0.0:
        effective = viscous / (1.0 + viscous * step / inertia)
    torque = -effective * velocity
    if coulomb and abs(velocity) > 1e-6:
        torque -= coulomb * (1.0 if velocity > 0.0 else -1.0)
    return max(-limit, min(limit, torque))


def actuator_efforts(tracker, cfg, excitation: float) -> dict:
    """Effort to publish for every joint: excitation plus that joint's friction."""
    return {
        "motor": excitation + damping_torque(
            tracker.motor_vel, cfg.b_viscous_motor, cfg.b_coulomb_motor,
            cfg.max_damping_torque, cfg.dt, cfg.inertia_motor,
        ),
        "joint1": damping_torque(
            tracker.joint1_vel, cfg.b_viscous_joint1, cfg.b_coulomb_joint1,
            cfg.max_damping_torque, cfg.dt, cfg.inertia_joint1,
        ),
    }
