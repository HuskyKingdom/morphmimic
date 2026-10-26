#!/usr/bin/env bash
# Downloads the BONES-SEED G1 motions on a machine WITH network, for the offline
# cluster. The dataset is gated: accept its licence once at
# https://huggingface.co/datasets/bones-studio/seed and log in (`hf auth login`, or set
# HF_TOKEN). g1.tar.gz is 23.5 GB; leave room for it unpacked as well.
#
#   bash docker/fetch_seed.sh /path/to/data          # -> /path/to/data/seed/...
#
# Then copy /path/to/data/seed to the cluster (rsync, disk) and point DATA at its parent.
set -euo pipefail

DATA=${1:?usage: fetch_seed.sh <data_dir>}
SEED="$DATA/seed"
mkdir -p "$SEED"

if ! command -v hf >/dev/null; then
  echo "[fetch] installing the Hugging Face CLI (pip install --user huggingface_hub)"
  python3 -m pip install --user -q "huggingface_hub[cli]"
  export PATH="$HOME/.local/bin:$PATH"
fi

hf download bones-studio/seed --repo-type dataset --local-dir "$SEED" \
  --include "g1.tar.gz" "metadata/seed_metadata_v004.csv" "LICENSE.md" "README.md"

echo "[fetch] unpacking g1.tar.gz"
tar -xzf "$SEED/g1.tar.gz" -C "$SEED"
cp "$SEED/metadata/seed_metadata_v004.csv" "$SEED/"
# convert_seed expects <csv-dir>/<session>/<clip>.csv.
CSV_DIR=$(dirname "$(find "$SEED" -path '*/csv/*/*.csv' -print -quit)")
CSV_DIR=$(dirname "$CSV_DIR")
echo "[fetch] G1 CSV root: $CSV_DIR ($(find "$CSV_DIR" -name '*.csv' | wc -l) clips)"
echo "[fetch] metadata:   $SEED/seed_metadata_v004.csv"
echo "[fetch] you can delete $SEED/g1.tar.gz now"
