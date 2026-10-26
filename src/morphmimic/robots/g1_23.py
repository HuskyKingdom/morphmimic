"""Unitree G1 23-DoF spec: mjlab's 29-DoF G1 with the six joints it lacks locked.

The 23-DoF G1 has one waist joint (yaw) and one wrist joint per arm (roll). Its
kinematics equal the 29-DoF G1's with waist roll / pitch and wrist pitch / yaw at zero
(Unitree's g1_23dof_rev_1_0.urdf and g1_29dof_rev_1_0.urdf: same joint origins), so the
model is mjlab's G1 MJCF with those six hinges locked (`LOCK_RANGE`, `LOCK_SOLREF`,
`LOCK_ARMATURE`):
the canonical layout, the identity translator, the actuators, the action scale and the
tracked bodies all stay the G1's. The policy's action on a locked slot is dropped
(`env.actions.CanonicalJointPositionAction`): that PD holds zero and helps the lock.

Mass beyond the locked joints follows the 23-DoF URDF: there are no waist roll / yaw
links (their 0.3 kg goes), and each wrist roll joint carries the 0.357 kg rubber hand
instead of the 29-DoF wrist (0.82 kg), so the wrist roll link takes the hand's inertial
and the locked wrist pitch / yaw links become massless.

On hardware the locked joints have no motor (Unitree SDK2 marks motor indices 13, 14,
20, 21, 27 and 28 invalid on the 23-DoF G1); `deploy/hardware/g1_node.py` reads them as
zero and sends them no command.
"""

from __future__ import annotations

import re
from dataclasses import replace

import mujoco
import numpy as np

from morphmimic.robots.spec import RobotSpec, Translator

LOCKED_JOINTS: tuple[str, ...] = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
# Half-width of a locked joint's range (rad). The soft limits (90%) then clip reset
# noise and reference angles of the locked slots to under 2 mrad.
LOCK_RANGE = 0.002
# Joint-limit solref (time constant, damping ratio): stiffer than MuJoCo's 0.02 s
# default, and at least two physics steps (2 x 0.005 s) for a stable constraint.
LOCK_SOLREF = (0.01, 1.0)
# Armature of the locked joints (kg m^2). A soft limit's torque stiffness scales with
# the inertia of its joint, and the bodies past the locked wrist joints are massless, so
# with the motor's armature (0.004) self-contacts on the hand open the wrist lock by
# 0.15 rad; with 1.0 every lock stays under 6 mrad (plain MuJoCo, mjlab's solver
# settings, all other joints jumping to random targets at their effort limits). It
# acts on the locked joint's own coordinate only, which does not move.
LOCK_ARMATURE = 1.0

# Bodies the 23-DoF G1 does not have: a negligible inertial keeps the tree unchanged.
_MASSLESS_BODIES: tuple[str, ...] = (
  "waist_yaw_link",
  "waist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)
_TINY_MASS = 1e-3
_TINY_INERTIA = 1e-7

# Left rubber hand, in the wrist roll joint frame (g1_23dof_rev_1_0.urdf,
# left_wrist_roll_rubber_hand); the right hand mirrors y, ixy and iyz.
_HAND_MASS = 0.35692864
_HAND_IPOS = (0.10794656650, 0.00163511945, 0.00202244863)
# (ixx, iyy, izz, ixy, ixz, iyz)
_HAND_INERTIA = (
  0.00019613494735,
  0.00200280358206,
  0.00194181412808,
  -0.00000419816908,
  -0.00003950860580,
  0.00000249774203,
)


def _set_inertial(
  body: mujoco.MjsBody,
  mass: float,
  ipos: tuple[float, float, float],
  fullinertia: tuple[float, ...],
) -> None:
  """Sets the body's inertial from a full inertia (ixx, iyy, izz, ixy, ixz, iyz) at
  `ipos`, as principal moments and axes (the MJCF's diagonal inertia excludes
  `fullinertia`)."""
  ixx, iyy, izz, ixy, ixz, iyz = fullinertia
  tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
  moments, axes = np.linalg.eigh(tensor)
  if np.linalg.det(axes) < 0:
    axes[:, 0] = -axes[:, 0]
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, axes.flatten())
  body.explicitinertial = True
  body.mass = mass
  body.ipos = list(ipos)
  body.iquat = quat.tolist()
  body.inertia = moments.tolist()


def get_g1_23_spec() -> mujoco.MjSpec:
  """mjlab's G1 MjSpec with the six joints locked and the 23-DoF masses."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

  spec = get_spec()
  for name in LOCKED_JOINTS:
    joint = spec.joint(name)
    joint.range = [-LOCK_RANGE, LOCK_RANGE]
    joint.solref_limit = list(LOCK_SOLREF)
  tiny = (_TINY_INERTIA,) * 3 + (0.0,) * 3
  for name in _MASSLESS_BODIES:
    _set_inertial(spec.body(name), _TINY_MASS, (0.0, 0.0, 0.0), tiny)
  ixx, iyy, izz, ixy, ixz, iyz = _HAND_INERTIA
  for side, s in (("left", 1.0), ("right", -1.0)):
    x, y, z = _HAND_IPOS
    _set_inertial(
      spec.body(f"{side}_wrist_roll_link"),
      _HAND_MASS,
      (x, s * y, z),
      (ixx, iyy, izz, s * ixy, ixz, s * iyz),
    )
  return spec


def _locks(actuator) -> bool:
  """Whether `actuator` (an mjlab actuator config) drives only locked joints; raises if
  it drives locked and other joints."""
  hits = [any(re.fullmatch(e, j) for j in LOCKED_JOINTS) for e in actuator.target_names_expr]
  if any(hits) and not all(hits):
    raise ValueError(f"actuator {actuator.target_names_expr} mixes locked and free joints")
  return all(hits)


def get_g1_23_robot_cfg():
  """The G1's EntityCfg (collisions, keyframe, actuators: gains, effort limits and so
  the action scale unchanged) on the locked model, with `LOCK_ARMATURE` on the locked
  joints."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg

  cfg = get_g1_robot_cfg()
  assert cfg.articulation is not None
  actuators = tuple(
    replace(a, armature=LOCK_ARMATURE) if _locks(a) else a
    for a in cfg.articulation.actuators
  )
  return replace(
    cfg,
    spec_fn=get_g1_23_spec,
    articulation=replace(cfg.articulation, actuators=actuators),
  )


def g1_23_spec() -> RobotSpec:
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE

  from morphmimic.constants import (
    JOINT_NAMES,
    SEED_CSV_JOINT_COLUMNS,
    TRACKED_BODY_NAMES,
  )

  return RobotSpec(
    name="g1_23",
    entity_cfg_fn=get_g1_23_robot_cfg,
    action_scale=dict(G1_ACTION_SCALE),
    joint_names=JOINT_NAMES,
    root_body="pelvis",
    anchor_body="torso_link",
    imu_body="pelvis",
    tracked_bodies=TRACKED_BODY_NAMES,
    foot_geom_regex=r"^(left|right)_foot[1-7]_collision$",
    com_body="torso_link",
    translator=Translator.identity(len(JOINT_NAMES)),
    csv_joint_columns=SEED_CSV_JOINT_COLUMNS,
    height_scale=1.0,
    contact_budget=(35, 250),
    locked_joints=LOCKED_JOINTS,
  )
