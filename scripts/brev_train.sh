#!/usr/bin/env bash
# One-shot setup + training for cosmos2_8b_g1_pickplace on a fresh cloud GPU box (e.g. NVIDIA Brev).
#
#   git clone git@github.com:vajpayeerajat/openpi.git && cd openpi && git checkout cosmos_2_8b_as_vision
#   HF_TOKEN=hf_xxx bash scripts/brev_train.sh [extra train_pytorch.py flags]
#
# What it does (every step is skipped when already done, so just re-run it after a disconnect or a new instance):
#   1. picks the biggest writable disk for the HF cache, data and checkpoints
#   2. installs ffmpeg (torchcodec), uv and the Python 3.11 venv
#   3. checks the HF login, write access and access to the gated Cosmos-Reason2-8B model
#   4. downloads the train/val dataset and links it to ./train and ./val (the config's local_root)
#   5. starts training in the background (survives SSH disconnects), logging to $WORK_DIR/logs.
#      Checkpoints + TensorBoard go to the private HF repo $HF_REPO_ID; only the newest checkpoint stays on disk.
#      If the run already has checkpoints on the Hub it resumes from the latest one, otherwise it starts fresh.
#
# Every setting below can be overridden from the environment, e.g. EXP_NAME=stage1_bs32 BATCH_SIZE=32 bash ...
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-cosmos2_8b_g1_pickplace}"
EXP_NAME="${EXP_NAME:-stage1_brev}"
HF_REPO_ID="${HF_REPO_ID:-rajat-vajpayee/cosmos2_8b_g1_pickplace}"  # checkpoints + tensorboard (private)
DATA_REPO_ID="${DATA_REPO_ID:-rajat-vajpayee/unitree_g1_data-830-episodes_split}"
VLM_REPO_ID="${VLM_REPO_ID:-nvidia/Cosmos-Reason2-8B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"  # each save is 6.6 GB on the Hub (2.2 GB without the optimizer)
VAL_INTERVAL="${VAL_INTERVAL:-42}"
NUM_WORKERS="${NUM_WORKERS:-$(( $(nproc) > 18 ? 16 : $(nproc) - 2 ))}"
RESUME="${RESUME:-auto}"  # auto | yes | no
VENV="${VENV:-$HOME/venvs/openpi_cosmos}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
log() { printf '\n\033[1;32m[brev_train] %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[brev_train] ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------------------------------
# 1. Disk: put everything big on the writable mount with the most free space.
# ---------------------------------------------------------------------------------------------------------------------
if [[ -z "${WORK_DIR:-}" ]]; then
    best_avail=0
    for d in /ephemeral /workspace /data /mnt "$HOME"; do
        [[ -d "$d" && -w "$d" ]] || continue
        avail=$(df -Pk "$d" | awk 'NR==2 {print $4}')
        if (( avail > best_avail )); then best_avail=$avail; WORK_DIR="$d/openpi_work"; fi
    done
fi
mkdir -p "$WORK_DIR"/{hf_cache,data,checkpoints,logs}
export HF_HOME="$WORK_DIR/hf_cache"
log "Work dir: $WORK_DIR ($(df -Ph "$WORK_DIR" | awk 'NR==2 {print $4}') free; need ~60 GB: 17 model + 7 data + checkpoints)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this needs a GPU instance."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------------------------------------------------
# 2. System packages, uv and the venv.
# ---------------------------------------------------------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null; then
    log "Installing ffmpeg (needed by torchcodec to decode the dataset videos)"
    sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git
fi

if ! command -v uv >/dev/null; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

PY="$VENV/bin/python"
if [[ ! -f "$VENV/.openpi_cosmos_installed" || "${FORCE_INSTALL:-0}" == 1 ]]; then
    log "Creating venv $VENV and installing dependencies (~10 min)"
    export UV_HTTP_TIMEOUT=900 GIT_LFS_SKIP_SMUDGE=1 VIRTUAL_ENV="$VENV"
    uv venv "$VENV" --python 3.11 --allow-existing
    uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    overrides="$WORK_DIR/uv_overrides.txt"
    printf "ml-dtypes==0.4.1\ntensorstore==0.1.74\ntransformers==4.57.1\ntorch==2.7.1\ntorchvision==0.22.1\ndatasets==3.6.0\ntorchcodec==0.5\n" > "$overrides"
    uv pip install -e . -e packages/openpi-client --override "$overrides"
    uv pip install tensorboard
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, 'cuda ok')"
    touch "$VENV/.openpi_cosmos_installed"
else
    log "Venv already set up: $VENV (FORCE_INSTALL=1 to reinstall)"
fi
HF="$VENV/bin/hf"

# ---------------------------------------------------------------------------------------------------------------------
# 3. Hugging Face auth: HF_TOKEN from the environment, or an existing / interactive login.
# ---------------------------------------------------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" ]] && ! "$HF" auth whoami >/dev/null 2>&1; then
    log "Not logged in to Hugging Face. Paste a token with WRITE access (or re-run with HF_TOKEN=...)"
    "$HF" auth login
fi
log "Checking Hugging Face access"
"$PY" - "$HF_REPO_ID" "$VLM_REPO_ID" <<'EOF'
import sys
import huggingface_hub as hub
from huggingface_hub.errors import GatedRepoError
repo_id, vlm = sys.argv[1:]
api = hub.HfApi()
print("logged in as", api.whoami()["name"])
try:
    hub.hf_hub_download(vlm, "config.json")
except GatedRepoError:
    sys.exit(f"No access to the gated model {vlm}: request it at https://huggingface.co/{vlm}")
# Fails here (instead of after loading the 8B model) if the token cannot write.
api.create_repo(repo_id, private=True, exist_ok=True)
print(f"checkpoint repo ready: https://huggingface.co/{repo_id}")
EOF

log "Downloading $VLM_REPO_ID into the HF cache (17 GB, skipped if cached)"
"$HF" download "$VLM_REPO_ID" --quiet >/dev/null

# ---------------------------------------------------------------------------------------------------------------------
# 4. Data: the config reads local_root="train" / "val" relative to the repo root.
# ---------------------------------------------------------------------------------------------------------------------
log "Downloading dataset $DATA_REPO_ID (7.3 GB, skipped if present)"
"$HF" download "$DATA_REPO_ID" --repo-type dataset --include "train/*" "val/*" \
    --local-dir "$WORK_DIR/data" --quiet >/dev/null
for split in train val; do
    if [[ -L "$split" ]]; then
        ln -sfn "$WORK_DIR/data/$split" "$split"
    elif [[ -e "$split" ]]; then
        log "./$split already exists and is not a symlink; using it as is"
    else
        ln -s "$WORK_DIR/data/$split" "$split"
    fi
done

norm_stats="assets/$CONFIG_NAME/g1_pickplace/norm_stats.json"
[[ -f "$norm_stats" ]] || die "$norm_stats is missing; it is committed in git, so pull the branch."

# ---------------------------------------------------------------------------------------------------------------------
# 5. Train.
# ---------------------------------------------------------------------------------------------------------------------
if [[ "$RESUME" == auto ]]; then
    RESUME=$("$PY" - "$HF_REPO_ID" "$CONFIG_NAME/$EXP_NAME/" <<'EOF'
import sys
import huggingface_hub as hub
repo_id, prefix = sys.argv[1:]
files = hub.HfApi().list_repo_files(repo_id)
print("yes" if any(f.startswith(prefix) and f.endswith("/optimizer.pt") for f in files) else "no")
EOF
)
fi
if [[ "$RESUME" == yes ]]; then
    run_mode=--resume
    log "Resuming $CONFIG_NAME/$EXP_NAME from its latest checkpoint on the Hub"
else
    run_mode=--overwrite
    log "Starting $CONFIG_NAME/$EXP_NAME from scratch"
fi

train_log="$WORK_DIR/logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
cmd=("$PY" scripts/train_pytorch.py "$CONFIG_NAME"
    --exp-name "$EXP_NAME"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --save-interval "$SAVE_INTERVAL"
    --val-interval "$VAL_INTERVAL"
    --checkpoint-base-dir "$WORK_DIR/checkpoints"
    --hf-repo-id "$HF_REPO_ID"
    "$run_mode"
    "$@")
printf '%q ' "${cmd[@]}" > "$WORK_DIR/logs/last_command.txt"
log "Command: $(cat "$WORK_DIR/logs/last_command.txt")"

# setsid + nohup: training keeps running when the SSH session drops. Ctrl-C below only stops watching the log.
HF_HOME="$HF_HOME" setsid nohup "${cmd[@]}" > "$train_log" 2>&1 < /dev/null &
echo $! > "$WORK_DIR/logs/train.pid"
log "Training started (pid $(cat "$WORK_DIR/logs/train.pid")). Log: $train_log
    Watch again:  tail -f $train_log
    Stop:         kill \$(cat $WORK_DIR/logs/train.pid)
    Hub:          https://huggingface.co/$HF_REPO_ID/tree/main/$CONFIG_NAME/$EXP_NAME
    Ctrl-C now only stops watching; training keeps running."
sleep 2
tail -f "$train_log"
