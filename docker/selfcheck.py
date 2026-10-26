"""Checks the container on a cluster node, without network: GPUs, driver, the Python
stack, the g1_23 model, and (with --subset-yaml) a short GPU rollout of the g1_23 env
that measures how far its locked joints move.

  python docker/selfcheck.py                                  # stack and model
  python docker/selfcheck.py --subset-yaml /data/seed_g1_23_npz50/subsets/tiny.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def section(title: str) -> None:
  print(f"\n== {title}", flush=True)


def check_stack() -> None:
  section("python and GPUs")
  import torch

  print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda)
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
      capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(out)
    driver = int(out.splitlines()[0].split(",")[-1].strip().split(".")[0])
    if driver < 575:
      print(f"WARNING: driver {driver} < 575; warp-lang and torch cu128 need CUDA 12.9 (575+)")
  except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as e:
    print("nvidia-smi unavailable:", e)
  if not torch.cuda.is_available():
    raise SystemExit("torch sees no GPU: run the container with --nv (Apptainer) or --gpus all")
  for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.0f} GiB")

  section("warp, mujoco, mjlab")
  import mujoco
  import mujoco_warp
  import warp as wp

  wp.init()
  print("warp", wp.config.version, "| devices", [str(d) for d in wp.get_cuda_devices()])
  print("mujoco", mujoco.__version__, "| mujoco_warp", getattr(mujoco_warp, "__version__", "?"))
  for key in ("WARP_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
    path = os.environ.get(key)
    if path:
      os.makedirs(path, exist_ok=True)
      ok = os.access(path, os.W_OK)
      print(f"{key}={path} ({'writable' if ok else 'NOT writable: bind a writable /cache'})")


def check_g1_23() -> None:
  section("g1_23 model")
  import numpy as np
  from mjlab.entity.entity import Entity

  from morphmimic.robots import get_robot
  from morphmimic.robots.morphology import morphology_vector

  spec = get_robot("g1_23")
  model = Entity(spec.entity_cfg_fn()).spec.compile()
  for name in spec.locked_joints:
    j = model.joint(name)
    print(f"{name:24s} range {np.round(j.range, 4)} armature {model.dof_armature[j.dofadr[0]]:.3f}")
  import mujoco

  print(f"total mass {mujoco.mj_getTotalmass(model):.3f} kg (Unitree g1_23dof_rev_1_0.urdf: 32.107)")
  m = morphology_vector("g1_23").numpy()
  print(f"morphology vector [{m.size}], range {m.min():.2f} .. {m.max():.2f}")


def check_rollout(subset_yaml: str, num_envs: int, steps: int) -> None:
  section(f"g1_23 env rollout ({num_envs} envs, {steps} steps, zero action)")
  import torch

  from morphmimic.env.env_cfg import morphmimic_env_cfg
  from morphmimic.env.morphmimic_env import MorphMimicEnv
  from morphmimic.robots import get_robot

  spec = get_robot("g1_23")
  env = MorphMimicEnv(morphmimic_env_cfg("g1_23", subset_yaml=subset_yaml, num_envs=num_envs), "cuda:0")
  try:
    env.reset()
    robot = env.scene["robot"]
    locked = [spec.native_index(j) for j in spec.locked_joints]
    action = torch.zeros(num_envs, 29, device="cuda:0")
    worst = torch.zeros(len(locked), device="cuda:0")
    for _ in range(steps):
      _, _, terminated, truncated, _ = env.step(action)
      worst = torch.maximum(worst, robot.data.joint_pos[:, locked].abs().max(0).values)
      done = torch.nonzero(terminated | truncated).flatten()
      if len(done):
        env.reset(env_ids=done)
    for name, v in zip(spec.locked_joints, worst.tolist(), strict=True):
      print(f"{name:24s} max |q| {1e3 * v:.2f} mrad")
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--subset-yaml", default=None, help="g1_23 clips: also run the env")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps", type=int, default=250)
  args = parser.parse_args()
  check_stack()
  check_g1_23()
  if args.subset_yaml:
    check_rollout(args.subset_yaml, args.num_envs, args.steps)
  print("\nselfcheck done", flush=True)


if __name__ == "__main__":
  main()
