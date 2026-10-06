#!/usr/bin/env bash
# Start the openpi server/dev container with this repo bind-mounted at /app,
# so edits on the host take effect inside the container immediately (no rebuild).
#
# Usage:
#   ./run.sh shell              # interactive bash inside the container
#   ./run.sh train              # full pi0.5 fine-tune ($TRAIN_CONFIG, exp name $EXP)
#   ./run.sh norm-stats         # compute normalization stats for $TRAIN_CONFIG
#   ./run.sh                    # serve default policy (env=$ENV, port=$PORT)
#   ./run.sh serve --env libero      # override env; PORT=... sets the port
#   ./run.sh <any command...>   # run an arbitrary command in the container
#
# Env vars:
#   TRAIN_CONFIG=pi05_g1        train config for `train` / `norm-stats`
#   EXP=g1_pickplace_v1         --exp-name for `train`; names the checkpoint directory
#   XLA_MEM_FRACTION=0.9        JAX device memory fraction (default 0.75 OOMs at batch 32)
#   PORT=8017                   host/container port for the websocket server
#   ENV=aloha_sim               default policy env (aloha|aloha_sim|droid|libero)
#   CONFIG=<name> CKPT=<dir>    serve a trained checkpoint instead of a default policy,
#                               e.g. CONFIG=pi05_simpk CKPT=27500 ./run.sh
#                               CKPT is resolved inside the container (repo root = /app)
#   GPUS=all                    passed to --gpus (set GPUS=none for CPU only)
#   OPENPI_DATA_HOME=~/.cache/openpi
#   NO_BUILD=1                  skip the image build (recommended once the image is built)
#   NAME=openpi_dev             container name
#
# Checkpoints land on the host at ./checkpoints/$TRAIN_CONFIG/$EXP/, because the repo is
# bind-mounted at /app. Run scripts/prune_checkpoints.py alongside a long training run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=openpi_server
NAME="${NAME:-openpi_dev}"
# 8000/8001 are occupied on this host by an unrelated long-running process,
# so default to 8017. Clients must be pointed at the same port.
PORT="${PORT:-8017}"
ENV_MODE="${ENV:-aloha_sim}"
# tyro expects the enum *name* (uppercase), accept either casing from the user.
ENV_MODE="$(echo "$ENV_MODE" | tr '[:lower:]' '[:upper:]')"
GPUS="${GPUS:-all}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Training defaults, used by the `train` and `norm-stats` subcommands.
# TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_g1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_g1_pickplace}"
EXP="${EXP:-g1_pickplace_v1}"
# JAX preallocates 75% of device memory by default; 0.9 is what a full pi0.5 fine-tune
# needs to avoid OOM at batch 32.
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME"

# --- build -------------------------------------------------------------------
if [[ "${NO_BUILD:-0}" != "1" ]]; then
  echo ">>> Building $IMAGE ..."
  # --network=host: DNS does not resolve inside docker's default build network on this
  # host, so apt-get cannot reach archive.ubuntu.com without it.
  docker build --network=host -t "$IMAGE" -f "$REPO_DIR/scripts/docker/serve_policy.Dockerfile" "$REPO_DIR"
fi

# --- decide what to run ------------------------------------------------------
# Only allocate a TTY when we actually have one (so run.sh also works from CI/scripts).
if [[ -t 0 && -t 1 ]]; then INTERACTIVE=(-it); else INTERACTIVE=(); fi
case "${1:-serve}" in
  shell|bash|sh)
    shift || true
    CMD=(/bin/bash)
    ;;
  train)
    # Extra flags are passed straight through to train.py:
    #   ./run.sh train                  -> $TRAIN_CONFIG, --exp-name=$EXP
    #   ./run.sh train --batch-size=16  -> same, with an override (use if batch 32 OOMs)
    # Pass NO_BUILD=1 to skip the image rebuild, which is what you want once it is built.
    shift || true
    CMD=(/bin/bash -lc "uv run scripts/train.py $TRAIN_CONFIG --exp-name=$EXP $*")
    ;;
  norm-stats|norm)
    shift || true
    CMD=(/bin/bash -lc "uv run scripts/compute_norm_stats.py --config-name $TRAIN_CONFIG $*")
    ;;
  serve)
    shift || true
    if [[ -n "${CKPT:-}" ]]; then
      [[ -n "${CONFIG:-}" ]] || { echo "ERROR: CKPT set but CONFIG is not; pass the train config name too." >&2; exit 1; }
      # Make a host-absolute CKPT path addressable inside the container, where the repo is /app.
      CKPT_IN="${CKPT/#$REPO_DIR/\/app}"
      [[ "$CKPT_IN" = /* ]] || CKPT_IN="/app/$CKPT_IN"
      CMD=(/bin/bash -lc "uv run scripts/serve_policy.py --port $PORT $* policy:checkpoint --policy.config=$CONFIG --policy.dir=$CKPT_IN")
    else
      CMD=(/bin/bash -lc "uv run scripts/serve_policy.py --env $ENV_MODE --port $PORT $*")
    fi
    ;;
  *)
    CMD=(/bin/bash -lc "$*")
    ;;
esac

# --- port preflight (host networking => host port must be free) --------------
if [[ "${CMD[*]}" == *serve_policy.py* ]]; then
  if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
    echo "ERROR: port ${PORT} is already in use on the host:" >&2
    ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" >&2 || true
    echo "Free it, or rerun with a different port:  PORT=8018 ./run.sh" >&2
    exit 1
  fi
fi

# Remove a stale container with the same name.
docker rm -f "$NAME" >/dev/null 2>&1 || true

GPU_ARGS=()
[[ "$GPUS" != "none" ]] && GPU_ARGS=(--gpus "$GPUS")

echo ">>> Starting $NAME (repo mounted read-write at /app, port $PORT)"
exec docker run --rm "${INTERACTIVE[@]}" \
  --name "$NAME" \
  --init \
  --network=host \
  "${GPU_ARGS[@]}" \
  -v "$REPO_DIR:/app" \
  -v /mnt/nas:/mnt/nas \
  -v "$OPENPI_DATA_HOME:/openpi_assets" \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -w /app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  -e UV_PROJECT_ENVIRONMENT=/.venv \
  -e UV_LINK_MODE=copy \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_LEROBOT_HOME=/app \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION" \
  "$IMAGE" "${CMD[@]}"
