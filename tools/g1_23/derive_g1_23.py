"""Derive the 23-DoF G1's deployment constants from the G1's, without a GPU.

The g1_23 model is the G1 model with six joints locked (`morphmimic/robots/g1_23.py`):
same joints, kinematics, actuators and timing. So

- `deploy/sim2sim/robots/g1_23.json` is `g1.json` with the robot name and the locked
  joints (what `deploy/sim2sim/export_assets.py --robot g1_23 --bake` writes; re-bake on
  the training machine to check it), and
- the `g1_23` entry of `deploy/vr_teleop/gmr_configs/stand.json` is the G1's stand with
  the locked joints at zero: same feet and base, the tracked bodies of its reference
  state moved by forward kinematics.

  python tools/g1_23/derive_g1_23.py --g1-xml <mjlab>/asset_zoo/robots/unitree_g1/xmls/g1.xml

Needs only mujoco and numpy (the G1 MJCF ships with mjlab).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROBOTS_DIR = REPO / "deploy/sim2sim/robots"
STAND_FILE = REPO / "deploy/vr_teleop/gmr_configs/stand.json"
# morphmimic.robots.g1_23.LOCKED_JOINTS (not imported: that module needs mjlab).
LOCKED_JOINTS = (
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def derive_constants(g1: dict) -> dict:
  out = dict(g1)
  out["robot"] = "g1_23"
  out["about"] = (
    "Robot constants for the sim2sim and deployment (deploy/sim2sim/obs.py)."
    " Derived from g1.json by tools/g1_23/derive_g1_23.py; export_assets.py --bake"
    " writes the same."
  )
  missing = [j for j in LOCKED_JOINTS if j not in g1["native_joint_names"]]
  if missing:
    raise ValueError(f"g1.json has no joints {missing}")
  out["locked_joints"] = list(LOCKED_JOINTS)
  return out


def stand_state(model: mujoco.MjModel, constants: dict, entry: dict,
                joint_pos: dict[str, float]) -> np.ndarray:
  """The stand's reference state [240] (`fit_gmr_configs.write_stand`'s layout) with
  the entry's base and `joint_pos`."""
  data = mujoco.MjData(model)
  data.qpos[:3] = entry["base_pos"]
  data.qpos[3:7] = entry["base_quat"]
  native = constants["native_joint_names"]
  for name in native:
    data.qpos[model.joint(name).qposadr[0]] = joint_pos[name]
  mujoco.mj_kinematics(model, data)
  q = np.array([joint_pos[n] for n in native])
  tr = constants["translator"]
  perm, sign, offset = (np.asarray(tr[k], float) for k in ("perm", "sign", "offset"))
  q_canon = sign * (q[perm.astype(int)] - offset)
  bodies = [model.body(b).id for b in constants["tracked_bodies"]]
  pos, quat = data.xpos[bodies], data.xquat[bodies]
  quat = np.where(quat[:, :1] < 0.0, -quat, quat)
  zeros = np.zeros(3 * len(bodies))
  return np.concatenate((q_canon, np.zeros(len(q_canon)), pos.ravel(), quat.ravel(), zeros, zeros))


def derive_stand(model: mujoco.MjModel, constants: dict, g1_entry: dict) -> dict:
  # The model must reproduce the G1's own stand state first.
  check = stand_state(model, constants, g1_entry, g1_entry["joint_pos"])
  err = float(np.abs(check - np.asarray(g1_entry["state"])).max())
  if err > 1e-4:
    raise ValueError(f"the G1 model does not reproduce the G1 stand state (max error {err:.2e})")
  joint_pos = {n: (0.0 if n in LOCKED_JOINTS else a) for n, a in g1_entry["joint_pos"].items()}
  entry = dict(g1_entry)
  entry["joint_pos"] = joint_pos
  entry["state"] = np.round(stand_state(model, constants, g1_entry, joint_pos), 6).tolist()
  entry["derived_from"] = "g1, locked joints at zero (tools/g1_23/derive_g1_23.py)"
  return entry


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--g1-xml", type=Path, default=None,
                      help="mjlab's G1 MJCF (default: from the installed mjlab)")
  args = parser.parse_args()
  if args.g1_xml is None:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    args.g1_xml = G1_XML
  model = mujoco.MjModel.from_xml_path(str(args.g1_xml))

  g1 = json.loads((ROBOTS_DIR / "g1.json").read_text())
  constants = derive_constants(g1)
  (ROBOTS_DIR / "g1_23.json").write_text(json.dumps(constants, indent=1) + "\n")
  print(f"[derive] {ROBOTS_DIR / 'g1_23.json'}")

  stand = json.loads(STAND_FILE.read_text())
  stand["g1_23"] = derive_stand(model, g1, stand["g1"])
  STAND_FILE.write_text(json.dumps(stand, indent=1) + "\n")
  moved = np.abs(np.asarray(stand["g1_23"]["state"]) - np.asarray(stand["g1"]["state"])).max()
  print(f"[derive] {STAND_FILE}: g1_23 stand (max state change from the G1's: {moved:.3f})")


if __name__ == "__main__":
  main()
