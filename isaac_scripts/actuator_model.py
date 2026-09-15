# =============================================================================
#  Actuator model script
#
#  Gives a joint the damping a real actuator has, computed inside the simulator
#  and applied before every physics step, from the velocity that step is about
#  to integrate.
#
#  This is the alternative to speed_control/actuator.py, which does the same
#  from the ROS side. Exactly one of them may be active: with both, every joint
#  gets its friction twice. Inside is worth choosing when the graph ticks at
#  the physics rate, since the torque then reaches the joint a step sooner than
#  a reply sent over ROS can; outside is worth choosing while the model is
#  still being changed, since it needs no round trip through the Script Editor.
#
#  The coefficients are bounded by the physics step: too large for the step
#  and the rig rings at half the step rate and never settles. Passing an
#  inertia scales the viscous term the way an implicit step would, which lifts
#  that ceiling. The ceiling belongs to the pair rather than to either joint --
#  doubling both rings, doubling either one alone does not. After changing the
#  coefficients or the step, kick the rig and check that the motion decays.
#
#  How to use:
#    1. Open Window > Visual Scripting > Action Graph and load /World/ActionGraph
#    2. Add a Script Node to it (right-click the canvas -> search "Script Node")
#    3. Connect the tick node's output to the Script Node's Exec In, or setup()
#       never runs
#    4. Right-click the Script Node -> Copy Prim Path
#    5. Paste it into script_node_path at the bottom of this file
#    6. In the Stage panel, right-click a joint -> Copy Prim Path, and paste it
#       into joint_path below
#    7. Fill in the coefficients measured from the real actuator
#    8. For another joint, copy the whole apply_actuator(...) block and paste it
#       below with a different path and coefficients
#    9. Copy this entire file into Window > Script Editor and run it
#   10. Save the stage (Ctrl+S), then press Stop and Play
#
#  The model is written into the Script Node's inputs:script attribute, so the
#  code travels with the USD: after step 10 this file can be moved or deleted
#  and the scene still works. Run it again to change a coefficient or a joint.
#
#  Omit any coefficient you do not want (just delete that line).
#
#  Two things the scene has to provide:
#    - drive:angular:physics:damping and physxJoint:jointFriction set to 0 on
#      every joint listed below, since this script is what supplies them
#    - an articulation root, which is where the joint velocities are read from
#
#  The damping is applied as body torques on the two links a joint connects,
#  equal and opposite. That leaves the DOF efforts free for the articulation
#  controller to command, and leaves the joint velocity on /joint_states exact,
#  which a joint drive's own damping would not.
# =============================================================================

import json

import omni.usd
from pxr import Sdf, UsdPhysics

SCRIPT_NODE_TYPE = "omni.graph.scriptnode.ScriptNode"

_JOINTS = []


def apply_actuator(joint_path, b_viscous=None, b_coulomb=None, max_torque=None,
                   inertia=None):
    """Register one joint. Nothing reaches the stage until install() runs."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(joint_path)
    if not prim.IsValid():
        print(f"[FAIL] prim not found: {joint_path}")
        return
    if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
        print(f"[FAIL] {joint_path} is not a joint (type is {prim.GetTypeName()})")
        return

    entry = {"joint": joint_path}
    done = []
    if b_viscous is not None:
        entry["b_viscous"] = float(b_viscous)
        done.append(f"viscous damping = {float(b_viscous)} N*m*s/rad")
    if b_coulomb is not None:
        entry["b_coulomb"] = float(b_coulomb)
        done.append(f"coulomb friction = {float(b_coulomb)} N*m")
    if max_torque is not None:
        entry["max_torque"] = float(max_torque)
        done.append(f"torque ceiling = {float(max_torque)} N*m")
    if inertia is not None:
        entry["inertia"] = float(inertia)
        done.append(f"effective inertia = {float(inertia)} kg*m^2")
    if len(entry) == 1:
        print(f"[WARN] {joint_path} given no coefficients, skipping")
        return

    for rel in ("physics:body0", "physics:body1"):
        targets = prim.GetRelationship(rel).GetTargets()
        if targets:
            done.append(f"{rel.split(':')[1]} = {targets[0]}")
    axis = prim.GetAttribute("physics:axis")
    if axis and axis.HasAuthoredValue():
        done.append(f"axis = {axis.Get()} (resolved to world at run time)")

    for existing in list(_JOINTS):
        if existing["joint"] == joint_path:
            _JOINTS.remove(existing)
            print(f"[WARN] {joint_path} registered twice, keeping the later one")
    _JOINTS.append(entry)

    print(f"[OK] {joint_path}")
    for d in done:
        print(f"        {d}")


# The engine, as it will live inside the USD. It imports nothing from this
# repository and resolves each joint's axis and bodies from the stage at run
# time, so nothing about the mechanism is baked into the text.
_ENGINE = '''\
"""Actuator model: joint damping applied before each physics step.

This code and the JOINTS table below were written into the stage by
isaac_scripts/actuator_model.py. Make changes by editing that script and
running it again, not by editing here.
"""

import numpy as np
import omni.physx
import omni.usd
from pxr import Gf, UsdPhysics

JOINTS = __JOINTS__

AXIS_UNIT = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
_state = {}


def _first(target, kinds):
    for mod, cls in kinds:
        try:
            return getattr(__import__(mod, fromlist=[cls]), cls)(target)
        except Exception:
            continue
    return None


def _dofs(getter):
    try:
        v = getter()
    except Exception:
        return None
    if v is None:
        return None
    if hasattr(v, "tolist"):
        v = v.tolist()
    while isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (list, tuple)):
        v = v[0]
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None


def _articulation_root(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def _quat_rotate(q, v):
    """Rotate v by quaternion q given as (w, x, y, z)."""
    w, x, y, z = q
    t = 2.0 * np.cross([x, y, z], v)
    return np.asarray(v) + w * t + np.cross([x, y, z], t)


def setup(db):
    stage = omni.usd.get_context().get_stage()
    root = _articulation_root(stage)
    if root is None:
        print("[actuator] no prim carries ArticulationRootAPI")
        return
    art = _first(root, (("isaacsim.core.prims", "Articulation"),
                        ("isaacsim.core.prims", "SingleArticulation"),
                        ("omni.isaac.core.articulations", "Articulation")))
    if art is None:
        print("[actuator] could not reach the articulation at %s" % root)
        return
    try:
        art.initialize()
    except Exception:
        pass

    names = None
    for attr in ("dof_names", "joint_names"):
        if hasattr(art, attr):
            try:
                names = [str(n) for n in getattr(art, attr)]
            except Exception:
                names = None
            if names:
                break
    if not names:
        print("[actuator] the articulation did not report its DOF names")
        return

    # Resolve each joint to a DOF index, an axis, and the pair of links that
    # take the torque and its reaction.
    plan, body_paths = [], []
    for cfg in JOINTS:
        prim = stage.GetPrimAtPath(cfg["joint"])
        if not prim.IsValid():
            print("[actuator] %s is gone from the stage" % cfg["joint"])
            continue
        name = prim.GetName()
        if name not in names:
            print("[actuator] %s is not a DOF of %s (DOFs: %s)" % (name, root, names))
            continue
        parent = prim.GetRelationship("physics:body0").GetTargets()
        child = prim.GetRelationship("physics:body1").GetTargets()
        if not child:
            print("[actuator] %s has no body1 to push" % cfg["joint"])
            continue
        axis_attr = prim.GetAttribute("physics:axis")
        axis = str(axis_attr.Get()) if axis_attr and axis_attr.HasAuthoredValue() else "Z"
        rot = prim.GetAttribute("physics:localRot0")
        local = rot.Get() if rot and rot.HasAuthoredValue() else None
        if local is not None:
            imag = local.GetImaginary()
            local = (float(local.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))
        entry = dict(cfg)
        entry.update(dof=names.index(name), axis=axis, local_rot=local,
                     child=str(child[0]), parent=str(parent[0]) if parent else None)
        plan.append(entry)
        for p in (entry["child"], entry["parent"]):
            if p and p not in body_paths:
                body_paths.append(p)

    if not plan:
        print("[actuator] nothing to drive")
        return
    bodies = _first(body_paths, (("isaacsim.core.prims", "RigidPrim"),
                                 ("isaacsim.core.prims", "RigidPrimView"),
                                 ("omni.isaac.core.prims", "RigidPrimView")))
    if bodies is None:
        print("[actuator] could not reach the links %s" % body_paths)
        return
    try:
        bodies.initialize()
    except Exception:
        pass

    _state.update(art=art, bodies=bodies, plan=plan, paths=body_paths,
                  n=len(body_paths), warned=False)
    physx = omni.physx.get_physx_interface()
    # Before the step, not after: the torque is computed from the velocity the
    # step is about to integrate, so it lands in that step rather than the next
    # one. The plain subscription is the fallback where the ordered form is
    # missing, and costs one step of lag.
    try:
        _state["sub"] = physx.subscribe_physics_on_step_events(_on_step, pre_step=True, order=0)
    except (AttributeError, TypeError):
        _state["sub"] = physx.subscribe_physics_step_events(_on_step)
        print("[actuator] no pre-step subscription on this build, damping lags one step")
    for e in plan:
        print("[actuator] %s -> DOF %d, axis %s, viscous %s, coulomb %s, max %s"
              % (e["joint"], e["dof"], e["axis"], e.get("b_viscous", 0.0),
                 e.get("b_coulomb", 0.0), e.get("max_torque", float("inf"))))


def _world_axis(entry, quats, index_of):
    """The joint's axis in world coordinates at this instant.

    The axis is given in the joint frame, which is the parent link's frame turned
    by localRot0, so it moves with the parent and cannot be precomputed.
    """
    axis = np.array(AXIS_UNIT.get(entry["axis"], (0.0, 0.0, 1.0)))
    if entry["local_rot"] is not None:
        axis = _quat_rotate(entry["local_rot"], axis)
    parent = entry["parent"]
    if parent is not None and quats is not None and parent in index_of:
        axis = _quat_rotate(quats[index_of[parent]], axis)
    n = np.linalg.norm(axis)
    return axis / n if n > 1e-9 else axis


def _on_step(dt):
    art, bodies, plan = _state.get("art"), _state.get("bodies"), _state.get("plan")
    if not plan:
        return
    vel = _dofs(art.get_joint_velocities)
    if vel is None or not all(np.isfinite(v) for v in vel):
        return
    quats = None
    try:
        poses = bodies.get_world_poses()
        if isinstance(poses, tuple) and len(poses) > 1:
            q = poses[1]
            quats = q.tolist() if hasattr(q, "tolist") else list(q)
    except Exception:
        quats = None
    index_of = {p: i for i, p in enumerate(_state["paths"])}

    torques = np.zeros((_state["n"], 3), dtype=np.float32)
    for entry in plan:
        i = entry["dof"]
        if i >= len(vel):
            continue
        w = vel[i]
        viscous = entry.get("b_viscous", 0.0)
        inertia = entry.get("inertia", 0.0)
        if inertia > 0.0:
            # What an implicit step of the same decay would have applied. The
            # explicit form overshoots once the coefficient is large for the
            # step and feeds the joint energy instead of draining it.
            viscous = viscous / (1.0 + viscous * dt / inertia)
        tau = -viscous * w
        coulomb = entry.get("b_coulomb", 0.0)
        if coulomb and abs(w) > 1e-6:
            tau -= coulomb * (1.0 if w > 0.0 else -1.0)
        limit = entry.get("max_torque")
        if limit is not None:
            tau = max(-limit, min(limit, tau))
        if not np.isfinite(tau) or tau == 0.0:
            continue
        axis = _world_axis(entry, quats, index_of) * tau
        torques[index_of[entry["child"]]] += axis
        if entry["parent"] in index_of:
            # The reaction acts on the parent link, as a real bearing's
            # friction does on the arm it is mounted on.
            torques[index_of[entry["parent"]]] -= axis

    zeros = np.zeros_like(torques)
    for call in (
        lambda: bodies.apply_forces_and_torques_at_pos(
            forces=zeros, torques=torques, is_global=True),
        lambda: bodies.apply_forces_and_torques_at_pos(
            zeros, torques, None, None, True),
        lambda: bodies.apply_forces_and_torques_at_pos(
            torques=torques, is_global=True),
    ):
        try:
            call()
            return
        except Exception:
            continue
    if not _state.get("warned"):
        _state["warned"] = True
        print("[actuator] no usable body-torque API on this build")


def compute(db):
    # Nothing to do per graph tick: the work runs on the physics step, which is
    # the finer of the two.
    return True


def cleanup(db):
    sub = _state.pop("sub", None)
    if sub is not None:
        try:
            sub.unsubscribe()
        except Exception:
            pass
    _state.clear()
'''


def install(script_node_path):
    """Write the engine and the registered joints into an existing Script Node."""
    if not _JOINTS:
        print("[FAIL] no joints registered, nothing to install")
        return
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(script_node_path)
    if not prim.IsValid():
        print(f"[FAIL] prim not found: {script_node_path}")
        print("        add a Script Node to the Action Graph first, then "
              "right-click it -> Copy Prim Path")
        return
    node_type = prim.GetAttribute("node:type")
    kind = str(node_type.Get()) if node_type and node_type.HasAuthoredValue() else None
    if kind != SCRIPT_NODE_TYPE:
        print(f"[FAIL] {script_node_path} is not a Script Node "
              f"(node:type is {kind})")
        return

    script_attr = prim.GetAttribute("inputs:script")
    if not script_attr.IsValid():
        script_attr = prim.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String)
    use_path = prim.GetAttribute("inputs:usePath")
    if not use_path.IsValid():
        use_path = prim.CreateAttribute("inputs:usePath", Sdf.ValueTypeNames.Bool)

    script = _ENGINE.replace("__JOINTS__", json.dumps(_JOINTS, indent=4))
    script_attr.Set(script)
    use_path.Set(False)

    exec_in = prim.GetAttribute("inputs:execIn")
    if exec_in.IsValid() and not exec_in.GetConnections():
        print(f"[WARN] {script_node_path} has nothing connected to Exec In, so "
              f"setup() will not run. Wire a tick node's output to it.")

    print(f"[OK] {script_node_path}")
    print(f"        {len(script)} characters, {len(_JOINTS)} joint(s) written into the stage")
    print("        save the stage (Ctrl+S), then press Stop and Play")


# -------------------------------- Joint 1 ------------------------------------
apply_actuator(
    joint_path = "/World/Cylinder/motor",   # right-click -> Copy Prim Path
    b_viscous  = 0.15,                      # viscous damping (N*m*s/rad)
    b_coulomb  = 0.0,                       # dry friction, speed independent (N*m)
    max_torque = 2.0,                       # ceiling on this joint's torque (N*m)
    inertia    = 0.022,                     # effective inertia (kg*m^2)
)


# -------------------------------- Joint 2 ------------------------------------
apply_actuator(
    joint_path = "/World/Cube_02/joint1",   # the passive link's hinge
    b_viscous  = 0.025,
    b_coulomb  = 0.0,
    max_torque = 2.0,
    inertia    = 0.004,
)


# -------------------------------- Joint 3 ------------------------------------
# For another joint, copy a block above, change the path and coefficients,
# then remove the leading "#" from each line.
#
# apply_actuator(
#     joint_path = "/World/Cube_03/joint2",
#     b_viscous  = 0.025,
#     b_coulomb  = 0.01,
#     max_torque = 2.0,
#     inertia    = 0.004,
# )


# ------------------------------- Script Node ---------------------------------
install(
    script_node_path = "/World/ActionGraph/script_node",  # right-click -> Copy Prim Path
)
