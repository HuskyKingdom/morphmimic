"""Read-only check of the G1's motor numbering, before the first run of a g1_23 policy.

Prints the angle of every motor slot the SDK reports, next to the joint each slot holds
in the two layouts `g1_node.py` knows (`--motor-layout sdk`: the 29-slot order of
Unitree SDK2's `G1JointIndex`, which leaves 13, 14, 20, 21, 27 and 28 empty on the
23-DoF G1; `compact`: motors 0..22 without gaps). It never sends a command, so the
robot stays in whatever mode it is in: put it in damping (or hang it) first, then move
one joint by hand, e.g. the left wrist, and see which slot changes:

  sdk layout:     the left wrist roll is slot 19; slots 20 and 21 stay still
  compact layout: the left wrist roll is slot 17

  uv run --group export python -m deploy.hardware.probe_motors --net <iface>
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.hardware.g1_node import MOTOR_LAYOUTS, motor_map
from deploy.sim2sim.obs import Robot


def slot_names(robot: Robot, layout: str) -> list[str]:
  """The joint on each of the robot's motor slots under `layout` ("-": none)."""
  motor = motor_map(robot, layout)
  names = ["-"] * robot.num_native
  for joint, slot in enumerate(motor):
    if slot >= 0:
      names[slot] = robot.native_joint_names[joint]
  return names


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--net", required=True, help="network interface to the robot (DDS)")
  parser.add_argument("--hz", type=float, default=2.0, help="print rate")
  args = parser.parse_args()

  import unitree_interface

  robot = Robot.load("g1_23")
  columns = {layout: slot_names(robot, layout) for layout in MOTOR_LAYOUTS}
  sdk = unitree_interface.UnitreeInterface.create_g1(args.net)
  print("[probe] reading only, no command is sent; Ctrl-C stops", flush=True)
  try:
    while True:
      state = sdk.read_low_state()
      q = np.asarray(state.motor.q, float)
      print(f"\nmode_machine {state.mode_machine}, {len(q)} slots")
      print(f"{'slot':>4}  {'q (rad)':>8}  " + "  ".join(f"{l:<28}" for l in MOTOR_LAYOUTS))
      for i, v in enumerate(q[: robot.num_native]):
        print(f"{i:>4}  {v:>8.3f}  " + "  ".join(f"{columns[l][i]:<28}" for l in MOTOR_LAYOUTS))
      time.sleep(1.0 / args.hz)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
