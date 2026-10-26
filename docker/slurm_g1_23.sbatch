#!/usr/bin/env bash
# Slurm job for one stage of the g1_23 pipeline in the Apptainer image. Edit the
# #SBATCH lines and the three paths for your cluster, then from the repo root:
#
#   sbatch --export=ALL,STAGE=data   --gres=gpu:0 docker/slurm_g1_23.sbatch   # CPU only
#   sbatch --export=ALL,STAGE=check  docker/slurm_g1_23.sbatch
#   sbatch --export=ALL,STAGE=smoke  docker/slurm_g1_23.sbatch
#   sbatch --export=ALL,STAGE=train  docker/slurm_g1_23.sbatch               # ~5 h on 1 H100
#   sbatch --export=ALL,STAGE=export docker/slurm_g1_23.sbatch
#
# A train job that hits the time limit resumes from logs/$RUN/latest.pt when resubmitted.
#SBATCH --job-name=g1_23
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
set -euo pipefail

: "${STAGE:?sbatch --export=ALL,STAGE=data|check|smoke|train|export}"
# Paths on the cluster: the image, the data root (holds seed/), the compile cache.
export SIF=${SIF:-$HOME/images/morphmimic_g1-23.sif}
export DATA=${DATA:-$HOME/data}
export CACHE=${CACHE:-$HOME/.cache/morphmimic}
export WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
echo "[slurm] $STAGE on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-none}, $WORKERS CPUs"
bash docker/run.sh env SEED="${SEED:-/data/seed}" NPZ="${NPZ:-/data/seed_g1_23_npz50}" \
  RUN="${RUN:-g1_23}" SUBSET="${SUBSET:-all}" ITERS="${ITERS:-40000}" GPUS="${GPUS:-0}" \
  NUM_ENVS="${NUM_ENVS:-4096}" WORKERS="$WORKERS" ${IT:+IT=$IT} ${CLIP:+CLIP=$CLIP} \
  bash docker/pipeline_g1_23.sh "$STAGE"
