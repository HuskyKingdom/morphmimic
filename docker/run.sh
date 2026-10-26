#!/usr/bin/env bash
# Runs a command in the training container on a cluster node, with the network off.
#
#   SIF=/path/morphmimic_g1-23.sif DATA=/path/data bash docker/run.sh python docker/selfcheck.py
#   IMAGE=morphmimic:g1-23        DATA=/path/data bash docker/run.sh bash docker/pipeline_g1_23.sh data
#
# SIF (Apptainer / Singularity) or IMAGE (Docker) picks the runtime. Inside the
# container: this checkout at /workspace/morphmimic (its code runs, its logs/ and out/
# are written here), $DATA at /data (the directory holding seed/), $CACHE at /cache
# (Warp, Inductor and Triton compile caches; default <repo>/.cache, keep it between jobs).
# The GPUs are the ones the job was given (CUDA_VISIBLE_DEVICES passes through).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${DATA:?set DATA to the directory that holds seed/}
CACHE=${CACHE:-$REPO/.cache}
mkdir -p "$CACHE"

if [[ -n "${SIF:-}" ]]; then
  APP=$(command -v apptainer || command -v singularity) \
    || { echo "no apptainer or singularity on this node" >&2; exit 1; }
  # --cleanenv keeps the host's Python and CUDA variables out; the job's GPUs go in
  # explicitly (an empty CUDA_VISIBLE_DEVICES would hide every GPU, so only when set).
  ENV_ARGS=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ENV_ARGS=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi
  exec "$APP" exec --nv --cleanenv ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    --bind "$REPO:/workspace/morphmimic,$DATA:/data,$CACHE:/cache" \
    --pwd /workspace/morphmimic \
    "$SIF" "$@"
elif [[ -n "${IMAGE:-}" ]]; then
  GPUS=${CUDA_VISIBLE_DEVICES:+\"device=$CUDA_VISIBLE_DEVICES\"}
  exec docker run --rm --network none --gpus "${GPUS:-all}" --ipc host \
    --user "$(id -u):$(id -g)" \
    -v "$REPO:/workspace/morphmimic" -v "$DATA:/data" -v "$CACHE:/cache" \
    -w /workspace/morphmimic "$IMAGE" "$@"
else
  echo "set SIF=<image.sif> (Apptainer) or IMAGE=<docker image>" >&2
  exit 1
fi
