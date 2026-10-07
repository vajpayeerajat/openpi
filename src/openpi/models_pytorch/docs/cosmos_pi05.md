# pi05 with Cosmos-Reason2-8B as the VLM

PyTorch only. Code: `src/openpi/models/cosmos_config.py`, `src/openpi/models_pytorch/cosmos_pytorch.py`.
Configs: `cosmos2_8b_g1_pickplace` (stage 1) and `cosmos2_8b_lora_g1_pickplace` (stage 2) in `src/openpi/training/config.py`.

## Architecture

- The VLM is Cosmos-Reason2-8B (Qwen3-VL-8B architecture: 36 layers, width 4096, 32 query heads, 8 KV heads,
  head_dim 128). The LM head is dropped.
- The prefix is `[<vision_start> 64 image tokens <vision_end>] x 3 cameras + prompt`. The prompt uses the pi05 format with
  the discretised state. The prefix is causal, as in Qwen3-VL pretraining, and uses Qwen3-VL's 3D M-RoPE positions
  and DeepStack visual injection. This prefix forward was checked to match HF `Qwen3VLModel` bit-for-bit.
- The action expert is new and trained from scratch (~560M params). It has 36 layers with the same head geometry as the
  VLM, width 768, SwiGLU MLP 2048, QK-norm and adaRMSNorm timestep conditioning (pi05). At every layer its queries
  attend to the VLM's keys/values of that layer, as in pi0.
- Stage 1: the VLM is fully frozen and runs under `no_grad`. Only the action expert and the action/time projections
  train.
- Stage 2: stage 1 plus LoRA r=16 on all LLM linear layers, plus a trainable vision merger.
- Checkpoints hold only trainable weights (~2 GB, not 17 GB). The VLM is reloaded from `vlm_path`.

## What to train

| Mode | Config | Trainable | Notes |
|---|---|---|---|
| Frozen VLM (action head only) | `cosmos2_8b_g1_pickplace` | action expert + projections (~560M) | cheapest; VLM runs under no_grad |
| Last-N decoder layers | `cosmos2_8b_last2_g1_pickplace` | + last 2 LLM layers (~200M, LR x0.1) | layers below L-N run under no_grad; change N with `--model.train-llm-last-n-layers 4` |
| LoRA | `cosmos2_8b_lora_g1_pickplace` | + LoRA r=16 on all LLM layers + vision merger | backprops through the whole 8B, ~3x slower |

The action expert reads the VLM's K/V at every layer, so last-N only adapts the features it sees in the top N layers.
In the final layer only `input_layernorm`/`k_proj`/`k_norm`/`v_proj` can affect the loss, so only those train.
The trainable VLM LR is `vlm_lr_multiplier` x the schedule (`--model.vlm-lr-multiplier`). Either VLM-training config
can warm-start from a stage-1 checkpoint with `--pytorch-weight-path`. The newly trainable VLM weights are not in that
checkpoint, so they start from the pretrained values.

## One-time setup

The repo's main env pins `transformers==4.53.2` (with `transformers_replace` patches) and a torch build without
Blackwell (sm_120) kernels. This model needs `transformers>=4.57` and torch built for cu128, so it has its own venv:

```bash
uv venv ~/venvs/openpi_cosmos --python 3.11
export VIRTUAL_ENV=~/venvs/openpi_cosmos UV_HTTP_TIMEOUT=900
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
printf "ml-dtypes==0.4.1\ntensorstore==0.1.74\ntransformers==4.57.1\ntorch==2.7.1\ntorchvision==0.22.1\ndatasets==3.6.0\ntorchcodec==0.5\n" > /tmp/ovr.txt
uv pip install -e . -e packages/openpi-client --override /tmp/ovr.txt
uv pip install tensorboard
# datasets>=4 breaks this lerobot version (torch.stack on a Column); torchcodec>0.5 needs torch 2.11 / CUDA 13.
```

Cosmos-Reason2 is gated on Hugging Face. Request access at https://huggingface.co/nvidia/Cosmos-Reason2-8B. Your HF
token must belong to an account that has been granted access. Until then, `--model.vlm-path Qwen/Qwen3-VL-8B-Instruct`
runs the identical architecture.

## Train

Data: `~/scratch/openpi/train` (204 episodes, 130,856 frames) and `~/scratch/openpi/val` (51 episodes). Both are
LeRobot v2.1, 30 fps, 28-dim state/action. `HF_LEROBOT_HOME` must point at their parent so that the repo ids
`train` and `val` resolve.

```bash
export HF_LEROBOT_HOME=$HOME/scratch/openpi
PY=~/venvs/openpi_cosmos/bin/python
# norm stats -> ./assets/cosmos2_8b_g1_pickplace/train (a random 40k-frame subset is plenty; the full set takes ~3 h)
JAX_PLATFORMS=cpu $PY scripts/compute_norm_stats.py --config-name cosmos2_8b_g1_pickplace --max-frames 40000
# stage 1 (val loss every 500 steps)
$PY scripts/train_pytorch.py cosmos2_8b_g1_pickplace --exp-name run1
# last 2 decoder layers + action expert (optional; warm-start from stage 1 or train from scratch)
$PY scripts/train_pytorch.py cosmos2_8b_last2_g1_pickplace --exp-name run1 \
    --pytorch-weight-path checkpoints/cosmos2_8b_g1_pickplace/run1/29999
# stage 2 (optional)
$PY scripts/train_pytorch.py cosmos2_8b_lora_g1_pickplace --exp-name run1 \
    --pytorch-weight-path checkpoints/cosmos2_8b_g1_pickplace/run1/29999
```

## Monitor

`scripts/train_pytorch.py` writes TensorBoard logs to `checkpoints/<config>/<exp>/tensorboard`:

- `train/*`: loss, grad_norm, learning_rate, time_per_step (every `log_interval` steps).
- `val/*` and `train_eval/*` (every `val_interval` steps): the flow-matching `loss`, plus metrics from fully denoised
  action chunks in normalised action space. These are `l1`/`mse` per body part, `all/first_step_l1`, and
  `all/acc@0.05` / `all/acc@0.1`, the fraction of action values within that tolerance. Flow matching has no
  classification accuracy, so `acc@tol` is the closest equivalent. `train_eval` runs on one training batch without
  augmentation; compare it with `val` to spot overfitting.

```bash
~/venvs/openpi_cosmos/bin/tensorboard --logdir checkpoints/cosmos2_8b_g1_pickplace --port 6006 --bind_all
```

## Serve

```bash
$PY scripts/serve_policy.py --port 8017 policy:checkpoint \
    --policy.config=cosmos2_8b_g1_pickplace --policy.dir=checkpoints/cosmos2_8b_g1_pickplace/run1/29999
```
