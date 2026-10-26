# MorphMimic training image for a GPU cluster without network access.
#
# Build it on a machine with network (Linux x86_64, Docker), then carry the image to the
# cluster as a tar (docker/build_offline.sh). Everything training needs is inside:
# Python 3.11, the locked environment of uv.lock (torch cu128, mjlab, mujoco-warp,
# warp-lang, export tools), gcc for Triton / Inductor's CUDA-graph compile, and EGL for
# MuJoCo videos. Nothing is downloaded at run time.
#
#   docker build -f docker/Dockerfile -t morphmimic:g1-23 .
#
# Host requirement on the cluster: an NVIDIA driver with CUDA 12.9 or newer (575+);
# warp-lang < 1.18 and torch cu128 run on it (README "Setup").

FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ARG UV_VERSION=0.9.5
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates curl \
      libegl1 libgl1 libglvnd0 libgles2 libosmesa6 \
      procps htop tmux \
    && rm -rf /var/lib/apt/lists/*

# uv and a managed Python 3.11 that live in the image.
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN uv python install 3.11

WORKDIR /workspace/morphmimic

# Dependencies first (cached while only the code changes). The GMR workspace member is
# only needed by the teleop group, which the cluster does not install; its
# pyproject.toml is enough for uv to read the lock.
COPY pyproject.toml uv.lock ./
COPY third_party/GMR/pyproject.toml third_party/GMR/pyproject.toml
RUN uv sync --frozen --no-install-project --group export

# The code, installed editable into /opt/venv: bind-mount a checkout over
# /workspace/morphmimic on the cluster and its code is what runs.
COPY . .
RUN uv sync --frozen --group export \
    && /opt/venv/bin/python -c "import morphmimic, mjlab, mujoco_warp, warp, torch, onnxruntime; print('ok', torch.__version__)"

# Offline at run time: no package sync, no hub or wandb traffic, headless MuJoCo, and
# every cache under /cache (bind a writable, persistent directory there: the Warp
# kernels and Inductor graphs compiled on the first run are then reused).
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    UV_OFFLINE=1 \
    UV_NO_SYNC=1 \
    UV_FROZEN=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_MODE=offline \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XDG_CACHE_HOME=/cache \
    WARP_CACHE_PATH=/cache/warp \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    MPLCONFIGDIR=/cache/matplotlib \
    WANDB_DIR=/cache/wandb \
    PYTHONUNBUFFERED=1

CMD ["python", "docker/selfcheck.py"]
