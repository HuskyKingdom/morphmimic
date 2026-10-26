#!/usr/bin/env bash
# The 23-DoF G1 pipeline, run inside the container (docker/run.sh) from the repo root.
#
#   bash docker/pipeline_g1_23.sh data     # G1 CSVs -> g1_23 npz (locked joints zeroed), subsets
#   bash docker/pipeline_g1_23.sh check    # self-check plus a GPU rollout of the locked model
#   bash docker/pipeline_g1_23.sh smoke    # 60 iterations on the tiny subset (compile, memory)
#   bash docker/pipeline_g1_23.sh train    # the paper's single-robot run (resumes if possible)
#   bash docker/pipeline_g1_23.sh export   # ONNX + g1_23.json, sim2sim assets, a sim2sim video
#
# Variables (defaults): SEED=/data/seed  NPZ=/data/seed_g1_23_npz50  RUN=g1_23
# SUBSET=all  ITERS=40000  NUM_ENVS=4096  GPUS=0 (e.g. "0,1,2,3": data parallel, the LRs
# scale with sqrt(#GPUs))  WORKERS=$(nproc)  IT=<export iteration, default ITERS>
# CLIP=walk_ff_loop_315_R_001__A257 (the clip of the sim2sim check).
set -euo pipefail

STAGE=${1:?usage: pipeline_g1_23.sh data|check|smoke|train|export}
SEED=${SEED:-/data/seed}
NPZ=${NPZ:-/data/seed_g1_23_npz50}
RUN=${RUN:-g1_23}
SUBSET=${SUBSET:-all}
ITERS=${ITERS:-40000}
NUM_ENVS=${NUM_ENVS:-4096}
GPUS=${GPUS:-0}
WORKERS=${WORKERS:-$(nproc)}
IT=${IT:-$ITERS}
CLIP=${CLIP:-walk_ff_loop_315_R_001__A257}

gpu_args() {
  if [[ "$GPUS" == *,* ]]; then
    echo "--gpu-ids ${GPUS//,/ }"
  else
    echo "--gpu $GPUS"
  fi
}

case "$STAGE" in
  data)
    first=$(find "$SEED" -path '*/csv/*/*.csv' -print -quit)
    [[ -n "$first" ]] || { echo "no <csv>/<session>/<clip>.csv under $SEED" >&2; exit 1; }
    csv_dir=$(dirname "$(dirname "$first")")
    metadata=$(ls "$SEED"/seed_metadata_v00*.csv | sort | tail -1)
    echo "[data] CSVs $csv_dir, metadata $metadata -> $NPZ ($WORKERS workers)"
    python -m morphmimic.scripts.convert_seed --robot g1_23 --csv-dir "$csv_dir" \
      --metadata "$metadata" --out-dir "$NPZ" --workers "$WORKERS"
    python -m morphmimic.scripts.build_subsets --manifest "$NPZ/manifest.csv" \
      --out-dir "$NPZ/subsets"
    ;;
  check)
    python docker/selfcheck.py --subset-yaml "$NPZ/subsets/tiny.yaml"
    ;;
  smoke)
    python -m morphmimic.scripts.train --robot g1_23 --subset-yaml "$NPZ/subsets/tiny.yaml" \
      $(gpu_args) --num-envs 512 --iterations 60 --warmup-env-steps 20 --wm-batch 512 \
      --imag-batch 512 --log-every 10 --ckpt-every 50 --kp-interface --kp-masking full \
      --log-dir logs --run-name "smoke_$RUN"
    ;;
  train)
    resume=()
    if [[ -f "logs/$RUN/latest.pt" ]]; then
      resume=(--resume "logs/$RUN/latest.pt")
      echo "[train] resuming logs/$RUN/latest.pt"
    fi
    python -m morphmimic.scripts.train --robot g1_23 --subset-yaml "$NPZ/subsets/$SUBSET.yaml" \
      $(gpu_args) --num-envs "$NUM_ENVS" --kp-interface --kp-masking full \
      --library-dtype float16 --iterations "$ITERS" --ckpt-every 4000 \
      --log-dir logs --run-name "$RUN" "${resume[@]}"
    ;;
  export)
    python -m deploy.export_onnx "logs/$RUN/ckpt_$IT.pt" "out/$RUN"
    python -m deploy.sim2sim.export_assets --robot g1_23 --subset-yaml "$NPZ/subsets/1h.yaml" \
      --clips "$CLIP" --out assets --bake
    git --no-pager diff --stat -- deploy/sim2sim/robots/g1_23.json 2>/dev/null || true
    python -m deploy.run --deploy "out/$RUN" --assets assets --robot g1_23 --clip "$CLIP" \
      --interface motion --delay-ms 20 --video "out/$RUN/$CLIP.mp4"
    echo "[export] out/$RUN: policy.onnx, g1_23.json; sim2sim video out/$RUN/$CLIP.mp4"
    ;;
  *)
    echo "unknown stage $STAGE" >&2
    exit 1
    ;;
esac
