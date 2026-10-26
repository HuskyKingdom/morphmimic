#!/usr/bin/env bash
# Builds the training image on a machine WITH network and packs it for the offline
# cluster. Run from the repo root on Linux x86_64 with Docker (and, for the .sif,
# Apptainer or Singularity):
#
#   bash docker/build_offline.sh            # -> dist/morphmimic_g1-23.tar.gz (+ .sif)
#
# Copy dist/ and the repo to the cluster. There, either `docker load -i <tar.gz>` (a
# cluster that runs Docker) or use the .sif with Apptainer (most Slurm clusters).
set -euo pipefail

IMAGE=${IMAGE:-morphmimic:g1-23}
OUT=${OUT:-dist}
NAME=${IMAGE//[:\/]/_}
mkdir -p "$OUT"

docker build -f docker/Dockerfile -t "$IMAGE" .
# The image must work with the network off: the self-check imports everything.
docker run --rm --network none "$IMAGE" python -c \
  "import morphmimic, mjlab, mujoco_warp, warp, torch, onnxruntime; print('offline import ok')"

docker save "$IMAGE" | gzip -1 > "$OUT/$NAME.tar.gz"
echo "[build] $OUT/$NAME.tar.gz"

if command -v apptainer >/dev/null || command -v singularity >/dev/null; then
  APP=$(command -v apptainer || command -v singularity)
  "$APP" build --force "$OUT/$NAME.sif" "docker-daemon://$IMAGE"
  echo "[build] $OUT/$NAME.sif"
else
  echo "[build] no apptainer here: build the .sif on the cluster with"
  echo "        apptainer build $NAME.sif docker-archive://$NAME.tar.gz"
fi
(cd "$OUT" && sha256sum ./* > SHA256SUMS)
echo "[build] checksums in $OUT/SHA256SUMS"
