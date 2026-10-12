"""Grad-CAM + causal input analysis of a G1 pick-place checkpoint on the val split, for BOTH model families:

  pi05    openpi pi05 (PaliGemma + Gemma action expert, JAX)     checkpoint dir has params/
  cosmos  pi05 with Cosmos-Reason2-8B (Qwen3-VL) as the VLM (PyTorch)   checkpoint dir has model.safetensors

Unlike rajat_text_openpi_vis.py / rajat_text_visualization.py (attention maps on an arbitrary video with a fake state),
this runs on real val samples (3 cameras, real state, real 50-step action chunk) through the exact training transforms,
so every number is measured against the demonstrated action.

What is measured, per val sample (frames sampled across episodes of every task):

  1. Action accuracy   sampled 50-step chunk (fixed noise) vs the ground-truth chunk, per joint group, compared with a
                       "hold still" baseline (repeat the current state). skill = MAE(model) / MAE(hold); < 1 is better.
  2. Grad-CAM          score = -flow-matching loss on the GT chunk (averaged over a fixed set of noise/time draws, i.e.
                       the training objective). Gradients w.r.t. the LLM input tokens (vision-tower output for image
                       patches, + DeepStack features for Qwen3-VL; token embeddings for the prompt). Per camera:
                       cam = ReLU(sum_c alpha_c * A_c), alpha = mean gradient over that camera's patches.
                       Per token: |grad . activation| -> share of attribution per input group (cameras, task words,
                       state digits, template). Computed for all joints and separately for right arm / right hand.
  3. Causal ablations  relative change of the same loss when an input is removed: each camera masked out, state set to
                       mid-range, and the instruction swapped (pink<->blue task; two-object tasks swap the order).
                       Gradients say where the model is sensitive; ablations say what it actually needs.
  4. Object grounding  octopus segmented by colour on the ego view; CAM lift = (CAM mass on object) / (object area
                       fraction). > 1 means the map concentrates on the object. On two-object tasks the instructed
                       object is compared against the other one.

Output (out_dir): report.md (verdicts + recommended actions), summary.json, per_sample.csv, fig_*.png, samples/*.png.
`--compare dirA dirB ...` writes a side-by-side compare.md from existing summary.json files.

Usage (pi05, JAX venv):
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python rajat_gradcam_analysis.py \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/unfrozen_vlm/pi05_g1_pickplace/830_episode_train_20000/15770
Usage (cosmos, torch venv; the frozen VLM loads from the HF cache, ~/.cache/huggingface by default):
  JAX_PLATFORMS=cpu ~/venvs/openpi_cosmos/bin/python rajat_gradcam_analysis.py \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/cosmos_as_vlm/cosmos2_8b_g1_pickplace/stage1_brev/17000
Compare:
  python rajat_gradcam_analysis.py --compare plots/gradcam/<run1> plots/gradcam/<run2>
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time as _time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

CAMERA_KEYS = ("observation.images.ego_view", "observation.images.ego_left", "observation.images.ego_right")
CAMERA_NAMES = ("ego_view", "left_wrist", "right_wrist")
MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # same order as in the prefix
ACTION_GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 14), "right_arm": slice(14, 21),
                 "right_hand": slice(21, 28)}
REAL_DIM = 28
CAM_TARGETS = ("all", "right_arm", "right_hand", "left_arm")  # Grad-CAM score variants (left hand never moves)
INPUT_GROUPS = ("ego_view", "left_wrist", "right_wrist", "prompt:task", "prompt:state", "prompt:template")
SHORT_HORIZON = 10  # steps (0.5 s at 20 fps): roughly what is executed before the next inference

# Verdict thresholds (relative loss increase when an input is removed / swapped)
NEGLIGIBLE = 0.02
LANG_IGNORED, LANG_USED = 0.05, 0.25
SKILL_GOOD, SKILL_USELESS = 0.5, 0.9
LIFT_GOOD, LIFT_WEAK = 1.5, 0.8
MIN_OBJ_AREA, MAX_OBJ_AREA = 0.003, 0.25
IDLE_HAND_RANGE = 0.05  # episodes whose right-hand action never moves more than this are idle (no demonstration)


# ============================================================================
# Val data
# ============================================================================


def _read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _decode_frames(path, indices):
    import imageio.v3 as iio

    want, out = set(int(i) for i in indices), {}
    for i, frame in enumerate(iio.imiter(path, plugin="pyav")):
        if i in want:
            out[i] = frame
        if i >= max(want):
            break
    return [out[int(i)] for i in indices]


def _hand_phases(actions):
    """approach / holding / open-after-grasp per frame, from the GT right-hand trajectory (deviation from the start)."""
    c = actions[:, ACTION_GROUPS["right_hand"]].mean(1)
    dev = np.abs(c - c[0])
    if dev.max() < 1e-3:
        return np.array(["approach"] * len(c))
    closed = dev / dev.max() > 0.5
    first = int(np.argmax(closed))
    return np.where(closed, "holding", np.where(np.arange(len(c)) < first, "approach", "open (after grasp)"))


def is_idle(actions):
    return float(np.ptp(actions[:, ACTION_GROUPS["right_hand"]].mean(1))) < IDLE_HAND_RANGE


def idle_episodes(split_dir):
    """Episodes where the right hand never grasps: recorded with a pick-up prompt but the robot does nothing."""
    split_dir = Path(split_dir)
    if not (split_dir / "meta" / "episodes.jsonl").exists():
        return None
    info = json.load(open(split_dir / "meta" / "info.json"))
    idle, frames, idle_frames, by_task = [], 0, 0, {}
    for ep in _read_jsonl(split_dir / "meta" / "episodes.jsonl"):
        e = ep["episode_index"]
        a = np.stack(pd.read_parquet(split_dir / info["data_path"].format(episode_chunk=e // info["chunks_size"],
                                                                         episode_index=e), columns=["action"])["action"])
        frames += len(a)
        if is_idle(a):
            idle.append(e)
            idle_frames += len(a)
            by_task[ep["tasks"][0]] = by_task.get(ep["tasks"][0], 0) + 1
    return {"split": str(split_dir), "n_episodes": info["total_episodes"], "idle_episodes": idle,
            "n_idle": len(idle), "idle_frame_frac": idle_frames / max(frames, 1), "idle_by_task": by_task}


def load_val_samples(val_dir, episodes_per_task, frames_per_episode, horizon, seed, skip_idle=()):
    val_dir = Path(val_dir)
    info = json.load(open(val_dir / "meta" / "info.json"))
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(val_dir / "meta" / "tasks.jsonl")}
    by_task = {}
    for ep in _read_jsonl(val_dir / "meta" / "episodes.jsonl"):
        if ep["episode_index"] not in set(skip_idle):
            by_task.setdefault(ep["tasks"][0], []).append(ep["episode_index"])
    rng = np.random.default_rng(seed)
    samples = []
    for task_index, task in sorted(tasks.items()):
        eps = sorted(by_task.get(task, []))
        for ep in sorted(rng.choice(eps, size=min(episodes_per_task, len(eps)), replace=False).tolist()):
            chunk = ep // info["chunks_size"]
            df = pd.read_parquet(val_dir / info["data_path"].format(episode_chunk=chunk, episode_index=ep))
            state = np.stack(df["observation.state"]).astype(np.float32)
            action = np.stack(df["action"]).astype(np.float32)
            n = len(df)
            frames = np.unique(np.linspace(0.03 * n, 0.93 * n, frames_per_episode).astype(int))
            phases = _hand_phases(action)
            imgs = {k: _decode_frames(val_dir / info["video_path"].format(episode_chunk=chunk, video_key=k,
                                                                          episode_index=ep), frames)
                    for k in CAMERA_KEYS}
            for j, t in enumerate(frames):
                idx = np.minimum(np.arange(t, t + horizon), n - 1)
                samples.append({
                    "episode": ep, "frame": int(t), "progress": float(t / n), "task_index": task_index,
                    "task": task, "phase": str(phases[t]), "state": state[t], "gt_actions": action[idx],
                    "images": {k: imgs[k][j] for k in CAMERA_KEYS},
                })
        print(f"[data] task {task_index}: {len([s for s in samples if s['task_index'] == task_index])} samples")
    return samples, tasks


def swapped_prompt(task_index, tasks):
    """Single-object tasks swap pink<->blue; two-object tasks swap the order (0<->1, 2<->3)."""
    return tasks[{0: 1, 1: 0, 2: 3, 3: 2}[task_index]]


def two_object(task_index):
    return task_index in (2, 3)


def target_object(task_index, phase):
    """Colour of the object the instruction asks to pick first (only meaningful before the first grasp)."""
    first = {0: "pink", 1: "blue", 2: "pink", 3: "blue"}[task_index]
    return first if phase == "approach" else None


# ============================================================================
# Transforms (the ones the G1 checkpoints were trained with, norm stats from the checkpoint)
# ============================================================================


def build_transforms(config, checkpoint, asset_file):
    from openpi import transforms as _transforms
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config
    if asset_file :
        assets = [asset_file]
    else:
        assets = [p.parent for p in Path(checkpoint, "assets").glob("*/norm_stats.json")]
    if len(assets) != 1:
        raise FileNotFoundError(f"Expected one assets/<id>/norm_stats.json in {checkpoint}, found {assets}")
    norm_stats = _normalize.load(assets[0])
    # Both pi05 (830-episode runs) and cosmos were trained with LeRobotG1DataConfig: G1Inputs, delta arms / absolute
    # hands, quantile norm. (config.py's pi05_g1_pickplace now points at the Dex3 variant; the transforms are the same.)
    data_cfg = _config.LeRobotG1DataConfig(repo_id="g1_pickplace").create(config.assets_dirs, config.model)
    inp = _transforms.compose([
        *data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.model_transforms.inputs,
    ])
    out = _transforms.compose([
        *data_cfg.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm),
        *data_cfg.data_transforms.outputs,
    ])
    q01, q99 = norm_stats["state"].q01, norm_stats["state"].q99
    return inp, out, ((q01 + q99) / 2).astype(np.float32)


def transform_sample(inp, s, *, prompt=None, state=None):
    # Copies: DeltaActions subtracts the state from the action chunk in place.
    raw = {**{k: s["images"][k] for k in CAMERA_KEYS},
           "observation.state": (s["state"] if state is None else state).copy(),
           "action": s["gt_actions"].copy(), "prompt": s["task"] if prompt is None else prompt}
    return inp(raw)


def to_model_batch(items):
    """List of transformed dicts -> numpy batch (images float32 [-1, 1], HWC)."""
    return {
        "image": {k: np.stack([it["image"][k] for it in items]).astype(np.float32) / 127.5 - 1.0 for k in MODEL_CAMERAS},
        "image_mask": {k: np.stack([np.asarray(it["image_mask"][k]) for it in items]) for k in MODEL_CAMERAS},
        "state": np.stack([it["state"] for it in items]).astype(np.float32),
        "tokenized_prompt": np.stack([it["tokenized_prompt"] for it in items]),
        "tokenized_prompt_mask": np.stack([it["tokenized_prompt_mask"] for it in items]),
    }


def noise_and_times(n, horizon, dim, seed):
    """Fixed noise draws and stratified times from the training distribution Beta(1.5, 1) * 0.999 + 0.001."""
    noise = np.random.default_rng(seed).standard_normal((n, horizon, dim)).astype(np.float32)
    t = (((np.arange(n) + 0.5) / n) ** (2.0 / 3.0) * 0.999 + 0.001).astype(np.float32)
    return noise, t


def group_matrix(action_dim):
    g = np.zeros((len(CAM_TARGETS), action_dim), np.float32)
    for i, name in enumerate(CAM_TARGETS):
        g[i, slice(0, REAL_DIM) if name == "all" else ACTION_GROUPS[name]] = 1.0
    return g


def pi05_prompt_string(prompt, state):
    """The exact string PaligemmaTokenizer / CosmosTokenizer encode for pi05 with the discrete state."""
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return cleaned, f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "


def token_groups(cleaned, full, offsets):
    """Char offsets per token -> (word_map {word: [token idx]}, {"task", "state", "other": [token idx]})."""
    import re

    from vis_utils import STOPWORDS

    def tokens_in(lo, hi):
        return [i for i, (s, e) in enumerate(offsets) if s >= 0 and s < hi and e > lo]

    t0 = full.index(cleaned)
    task_idx = tokens_in(t0, t0 + len(cleaned))
    s0 = full.find("State: ") + len("State: ")
    state_idx = tokens_in(s0, full.find(";", s0))
    other = [i for i in range(len(offsets)) if i not in set(task_idx) | set(state_idx)]
    word_map = {}
    for m in re.finditer(r"[A-Za-z]+", cleaned):
        w = m.group().lower()
        idx = tokens_in(t0 + m.start(), t0 + m.end())
        if w in STOPWORDS or len(w) < 2 or not idx:
            continue
        key, k = w, 2
        while key in word_map:
            key, k = f"{w}#{k}", k + 1
        word_map[key] = idx
    return word_map, {"task": task_idx, "state": state_idx, "other": other}


# ============================================================================
# Backends. Both expose:
#   layout: cams -> prefix indices [tpi], grid, text_start
#   losses(batch, actions, noise, t) -> [B, D] per-dim flow-matching MSE (mean over draws and horizon)
#   gradcam(batch1, actions1, noise, t, gmat) -> {target: (act [P,E] f32, grad [P,E] f32, extra [(act, grad)] per cam)}
#   sample(batch, noise1) -> [B, H, D] normalized actions
# ============================================================================


class Pi05Backend:
    kind = "pi05"

    def __init__(self, config, checkpoint, num_steps):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask
        from openpi.models.pi0 import posemb_sincos
        from openpi.models.tokenizer import PaligemmaTokenizer

        self.jax, self.jnp, self._model = jax, jnp, _model
        self.cfg = config.model
        print(f"[pi05] loading {checkpoint}/params")
        model = config.model.load(_model.restore_params(os.path.join(checkpoint, "params")))
        graphdef, self.state = nnx.split(model)
        self._graphdef = graphdef
        self.tokenizer = PaligemmaTokenizer(self.cfg.max_token_len)
        self.label = "pi05 (PaliGemma 3B, JAX)"

        def embed(state, obs):
            m = nnx.merge(graphdef, state)
            return m.embed_prefix(_model.preprocess_observation(None, obs, train=False))

        def sqerr(state, tokens, mask, ar, x_t, t, u_t):
            m = nnx.merge(graphdef, state)
            rep = x_t.shape[0] // tokens.shape[0]
            _, kv = m.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, 1) - 1)
            kv = jax.tree.map(lambda a: jnp.repeat(a, rep, axis=1), kv)
            mask_r = jnp.repeat(mask, rep, axis=0)
            # Model.embed_suffix for pi05 (inlined: it type-checks an Observation it does not use for pi05)
            act = m.action_in_proj(x_t)
            temb = posemb_sincos(t, m.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            cond = nnx.swish(m.time_mlp_out(nnx.swish(m.time_mlp_in(temb))))
            h = x_t.shape[1]
            smask = jnp.ones((x_t.shape[0], h), jnp.bool_)
            full = jnp.concatenate([jnp.repeat(mask_r[:, None], h, axis=1),
                                    make_attn_mask(smask, jnp.array([True] + [False] * (h - 1)))], axis=-1)
            pos = jnp.sum(mask_r, -1)[:, None] + jnp.cumsum(smask, -1) - 1
            (_, out), _ = m.PaliGemma.llm([None, act], mask=full, positions=pos, kv_cache=kv, adarms_cond=[None, cond])
            v = m.action_out_proj(out[:, -h:])
            return jnp.square(v.astype(jnp.float32) - u_t)

        def losses(state, obs, actions, noise, t):
            tokens, mask, ar = embed(state, obs)
            b, r = actions.shape[0], noise.shape[0]
            a = jnp.repeat(actions, r, 0)
            nz, tt = jnp.tile(noise, (b, 1, 1)), jnp.tile(t, b)
            x_t = tt[:, None, None] * nz + (1 - tt[:, None, None]) * a
            se = sqerr(state, tokens, mask, ar, x_t, tt, nz - a)
            return se.reshape(b, r, *se.shape[1:]).mean((1, 2))

        def group_losses(tokens, state, mask, ar, actions, noise, t, gmat):
            r = noise.shape[0]
            a = jnp.repeat(actions, r, 0)
            x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * a
            per_dim = sqerr(state, tokens, mask, ar, x_t, t, noise - a).mean((0, 1))
            return (gmat @ per_dim) / gmat.sum(1)

        self._embed = jax.jit(embed)
        self._losses = jax.jit(losses)
        self._jac = jax.jit(jax.jacrev(group_losses))
        self._sample = jax.jit(lambda state, obs, noise: nnx.merge(graphdef, state).sample_actions(
            jax.random.key(0), obs, num_steps=num_steps, noise=noise))

        tpi = (_model.IMAGE_RESOLUTION[0] // 14) ** 2
        self.grid = int(np.sqrt(tpi))
        self.cam_index = {c: np.arange(i * tpi, (i + 1) * tpi) for i, c in enumerate(CAMERA_NAMES)}
        self.text_start = 3 * tpi
        self.special_index = np.array([], int)

    def _obs(self, batch):
        return self._model.Observation.from_dict(self.jax.tree.map(self.jnp.asarray, batch))

    def losses(self, batch, actions, noise, t):
        return np.asarray(self._losses(self.state, self._obs(batch), self.jnp.asarray(actions), noise, t))

    def gradcam(self, batch, actions, noise, t, gmat):
        tokens, mask, ar = self._embed(self.state, self._obs(batch))
        jac = self._jac(tokens, self.state, mask, ar, self.jnp.asarray(actions), noise, t, gmat)  # [G, 1, P, E]
        act = np.asarray(tokens[0], np.float32)
        return {g: (act, -np.asarray(jac[i, 0], np.float32), []) for i, g in enumerate(CAM_TARGETS)}

    def sample(self, batch, noise1):
        b = batch["state"].shape[0]
        return np.asarray(self._sample(self.state, self._obs(batch), self.jnp.asarray(np.repeat(noise1[None], b, 0))))

    def text_groups(self, prompt, state28):
        cleaned, full = pi05_prompt_string(prompt, state28)
        sp = self.tokenizer._tokenizer  # noqa: SLF001
        proto = sp.encode(full, out_type="immutable_proto")
        offsets = [(-1, -1)] + [(p.begin, p.end) for p in proto.pieces]  # BOS first
        ids = np.asarray([sp.bos_id()] + [p.id for p in proto.pieces])
        return (ids, *token_groups(cleaned, full, offsets))


class CosmosBackend:
    kind = "cosmos"

    def __init__(self, config, checkpoint, num_steps):
        import torch

        from openpi.models.tokenizer import CosmosTokenizer
        from openpi.models_pytorch.cosmos_pytorch import CosmosPi05Pytorch
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        self.torch, self.make_att_2d_masks = torch, make_att_2d_masks
        self.cfg = config.model
        self.num_steps = num_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[cosmos] loading {self.cfg.vlm_path} + {checkpoint}/model.safetensors")
        self.model = CosmosPi05Pytorch(self.cfg)
        self.model.load_trainable_weights(os.path.join(checkpoint, "model.safetensors"))
        self.model.to(self.device).eval()
        self.tokenizer = CosmosTokenizer(self.cfg.max_token_len, self.cfg.vlm_path)
        vlm_state = "frozen VLM" if not self.model.vlm_requires_grad else "VLM partly trained"
        self.label = f"Cosmos-Reason2-8B pi05 ({vlm_state}, PyTorch)"
        tpi, self.grid = self.model.tokens_per_image, self.model.llm_grid
        self.tpi = tpi
        self.cam_index = {c: i * (tpi + 2) + 1 + np.arange(tpi) for i, c in enumerate(CAMERA_NAMES)}
        self.text_start = 3 * (tpi + 2)
        self.special_index = np.array([[i * (tpi + 2), i * (tpi + 2) + tpi + 1] for i in range(3)]).ravel()

    def _inputs(self, batch):
        from openpi.models import model as _model

        t = self.torch
        obs = _model.Observation.from_dict({
            "image": {k: t.from_numpy(v).to(self.device) for k, v in batch["image"].items()},
            "image_mask": {k: t.from_numpy(v).to(self.device) for k, v in batch["image_mask"].items()},
            "state": t.from_numpy(batch["state"]).to(self.device),
            "tokenized_prompt": t.from_numpy(batch["tokenized_prompt"]).to(self.device),
            "tokenized_prompt_mask": t.from_numpy(batch["tokenized_prompt_mask"]).to(self.device),
        })
        return obs, self.model._preprocess_observation(obs, train=False)  # noqa: SLF001

    def _prefix(self, inputs, leaf):
        """compute_prefix_kv, but with gradients enabled down to the LLM input tokens + DeepStack features."""
        m, t = self.model, self.torch
        embs, pad, att, positions, next_pos, vis_idx, deepstack = m.embed_prefix(*inputs)
        if leaf:
            embs = embs.detach().requires_grad_(True)
            deepstack = [d.detach().requires_grad_(True) for d in deepstack]
        att_2d = self.make_att_2d_masks(pad, att)
        att_2d = att_2d | t.eye(att_2d.shape[-1], dtype=t.bool, device=att_2d.device)[None]
        cos, sin = m.vlm.language_model.rotary_emb(embs, positions)
        h, kv = embs, []
        for i in range(m.num_llm_layers):
            ds = deepstack[i] if i < len(deepstack) else None
            h, k, v = m._llm_layer(i, h, cos, sin, att_2d[:, None], vis_idx, ds)  # noqa: SLF001
            kv.append((k, v))
        return kv, pad, next_pos, embs, deepstack

    def _sqerr(self, prefix, actions, noise, t):
        th = self.torch
        kv, pad, next_pos = prefix[:3]
        b, r = actions.shape[0], noise.shape[0]
        a = th.from_numpy(actions).to(self.device).repeat_interleave(r, 0)
        nz = th.from_numpy(noise).to(self.device).repeat(b, 1, 1)
        tt = th.from_numpy(t).to(self.device).repeat(b)
        x_t = tt[:, None, None] * nz + (1 - tt[:, None, None]) * a
        kv_r = [(k.repeat_interleave(r, 0), v.repeat_interleave(r, 0)) for k, v in kv]
        v_t = self.model.run_action_expert(kv_r, pad.repeat_interleave(r, 0), next_pos.repeat_interleave(r, 0), x_t, tt)
        return ((v_t.float() - (nz - a)) ** 2).view(b, r, *v_t.shape[1:])

    def losses(self, batch, actions, noise, t):
        th = self.torch
        with th.no_grad(), th.autocast(self.device.type, dtype=th.bfloat16):
            _, inputs = self._inputs(batch)
            return self._sqerr(self._prefix(inputs, leaf=False), actions, noise, t).mean((1, 2)).float().cpu().numpy()

    def gradcam(self, batch, actions, noise, t, gmat):
        th = self.torch
        with th.enable_grad(), th.autocast(self.device.type, dtype=th.bfloat16):
            _, inputs = self._inputs(batch)
            prefix = self._prefix(inputs, leaf=True)
            per_dim = self._sqerr(prefix, actions, noise, t).mean((0, 1, 2))
            g = th.from_numpy(gmat).to(self.device)
            group = (g @ per_dim) / g.sum(1)
        embs, deepstack = prefix[3], prefix[4]
        out = {}
        for i, name in enumerate(CAM_TARGETS):
            grads = th.autograd.grad(-group[i], [embs, *deepstack], retain_graph=i < len(CAM_TARGETS) - 1)
            extra = [(d[0].detach().float().cpu().numpy(), gd[0].float().cpu().numpy()) for d, gd in zip(deepstack, grads[1:])]
            out[name] = (embs[0].detach().float().cpu().numpy(), grads[0][0].float().cpu().numpy(), extra)
        return out

    def sample(self, batch, noise1):
        th = self.torch
        obs, _ = self._inputs(batch)
        noise = th.from_numpy(np.repeat(noise1[None], batch["state"].shape[0], 0)).to(self.device)
        return self.model.sample_actions(self.device, obs, noise=noise, num_steps=self.num_steps).float().cpu().numpy()

    def text_groups(self, prompt, state28):
        cleaned, full = pi05_prompt_string(prompt, state28)
        if self.tokenizer._tokenizer is None:  # noqa: SLF001
            self.tokenizer._load()  # noqa: SLF001
        enc = self.tokenizer._tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)  # noqa: SLF001
        return (np.asarray(enc["input_ids"]), *token_groups(cleaned, full, list(enc["offset_mapping"])))


def detect_backend(checkpoint, config_name):
    from openpi.training import config as _config

    if os.path.exists(os.path.join(checkpoint, "model.safetensors")):
        if config_name is None:
            from safetensors import safe_open

            with safe_open(os.path.join(checkpoint, "model.safetensors"), "pt") as f:
                keys = list(f.keys())
            if any("lora_" in k for k in keys):
                config_name = "cosmos2_8b_lora_g1_pickplace"
            elif any(k.startswith("vlm.language_model.layers.") for k in keys):
                config_name = "cosmos2_8b_last2_g1_pickplace"
            else:
                config_name = "cosmos2_8b_g1_pickplace"
        config = _config.get_config(config_name)
        if config_name == "cosmos2_8b_last2_g1_pickplace":
            layers = {int(k.split(".")[3]) for k in keys if k.startswith("vlm.language_model.layers.")}
            config = dataclasses.replace(config, model=dataclasses.replace(config.model,
                                                                           train_llm_last_n_layers=len(layers)))
        return CosmosBackend, config, config_name
    if os.path.isdir(os.path.join(checkpoint, "params")):
        config_name = config_name or "pi05_g1_pickplace"
        return Pi05Backend, _config.get_config(config_name), config_name
    raise FileNotFoundError(f"{checkpoint}: neither model.safetensors (cosmos) nor params/ (pi05). Note: the "
                            "stage-2 LoRA dirs synced without model.safetensors cannot be analysed.")


# ============================================================================
# Attribution helpers
# ============================================================================


def token_attribution(act, grad, extra, backend):
    """|grad . activation| per prefix token; DeepStack contributions are added onto the image tokens."""
    r = (grad * act).sum(-1)
    for ci, cam in enumerate(CAMERA_NAMES):
        idx = backend.cam_index[cam]
        n = len(idx)
        for a, g in extra:  # deepstack: [3 * tpi, D], camera-major
            r[idx] += (g[ci * n:(ci + 1) * n] * a[ci * n:(ci + 1) * n]).sum(-1)
    return np.abs(r)


def token_valid(img, grid):
    """[grid*grid] bool: patch tokens that are mostly real image (not the letterbox band)."""
    m = content_mask(img).astype(np.float32)
    h, w = m.shape
    cells = m[: h // grid * grid, : w // grid * grid].reshape(grid, h // grid, grid, w // grid).mean((1, 3))
    return cells.ravel() > 0.5


def gradcam_maps(act, grad, extra, backend, valid):
    """Classic Grad-CAM per camera: alpha = mean gradient over the camera's real-image patches,
    cam = ReLU(sum_c alpha_c * A_c); letterbox patches are left out of alpha and zeroed in the map."""
    maps = {}
    for ci, cam in enumerate(CAMERA_NAMES):
        idx, v = backend.cam_index[cam], valid[cam]
        a, g = act[idx], grad[idx]
        m = a @ g[v].mean(0)
        n = len(idx)
        for ae, ge in extra:
            m = m + ae[ci * n:(ci + 1) * n] @ ge[ci * n:(ci + 1) * n][v].mean(0)
        maps[cam] = (np.maximum(m, 0.0) * v).reshape(backend.grid, backend.grid)
    return maps


def input_group_shares(r, backend, text_groups):
    shares = {c: r[backend.cam_index[c]].sum() for c in CAMERA_NAMES}
    for g, label in [("task", "prompt:task"), ("state", "prompt:state"), ("other", "prompt:template")]:
        shares[label] = r[backend.text_start + np.asarray(text_groups[g], int)].sum() if text_groups[g] else 0.0
    shares["prompt:template"] += r[backend.special_index].sum() if len(backend.special_index) else 0.0
    total = sum(shares[k] for k in INPUT_GROUPS) + 1e-12
    return {k: float(shares[k] / total) for k in INPUT_GROUPS}


def word_shares(r, backend, word_map):
    raw = {}
    for w, idx in word_map.items():
        raw[w.split("#")[0]] = raw.get(w.split("#")[0], 0.0) + r[backend.text_start + np.asarray(idx)].sum()
    total = sum(raw.values()) + 1e-12
    return {w: float(v / total) for w, v in raw.items()}


def octopus_masks(img):
    import scipy.ndimage as ndi

    hsv = mcolors.rgb_to_hsv(img.astype(np.float32) / 255.0)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    def largest(m):
        lab, n = ndi.label(ndi.binary_opening(m, iterations=2))
        if n == 0:
            return np.zeros_like(m)
        return lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))

    return {"pink": largest((h > 0.80) & (h < 0.97) & (s > 0.35) & (v > 0.35)),
            "blue": largest((h > 0.45) & (h < 0.54) & (s > 0.38) & (v > 0.5))}


def content_mask(img):
    """False on the black letterbox bands added by resize_with_pad."""
    rows, cols = img.max(axis=(1, 2)) > 0, img.max(axis=(0, 2)) > 0
    return rows[:, None] & cols[None, :]


def upsample(cam, size):
    return np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize((size[1], size[0]), Image.BICUBIC)).clip(0)


def cam_lift(cam_up, mask, valid):
    cam_up = cam_up * valid
    area = mask[valid].mean()
    if not (MIN_OBJ_AREA <= area <= MAX_OBJ_AREA) or cam_up.sum() <= 0:
        return np.nan
    return float((cam_up[mask].sum() / cam_up.sum()) / area)


# ============================================================================
# Main analysis
# ============================================================================


def analyse(args):
    Backend, config, config_name = detect_backend(args.checkpoint, args.config_name)
    step = Path(args.checkpoint.rstrip("/")).name
    run = Path(args.checkpoint.rstrip("/")).parent.name
    out_dir = Path(args.out_dir or f"./plots/gradcam/{config_name}__{run}__{step}")
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    backend = Backend(config, args.checkpoint, args.num_steps)
    inp, out_tf, mid_state = build_transforms(config, args.checkpoint, args.asset)
    horizon, adim = config.model.action_horizon, config.model.action_dim
    data_check = {"val": idle_episodes(args.val_dir), "train": idle_episodes(args.train_dir)}
    for k, d in data_check.items():
        if d:
            print(f"[data] {k}: {d['n_idle']}/{d['n_episodes']} idle episodes ({100 * d['idle_frame_frac']:.1f}% of frames)")
    skip = [] if args.include_idle else data_check["val"]["idle_episodes"]
    samples, tasks = load_val_samples(args.val_dir, args.episodes_per_task, args.frames_per_episode, horizon, args.seed,
                                      skip_idle=skip)
    noise, times = noise_and_times(args.n_noise, horizon, adim, args.seed)
    gmat = group_matrix(adim)
    sample_noise = np.random.default_rng(args.seed + 1).standard_normal((horizon, adim)).astype(np.float32)

    variants = ["base", "-ego_view", "-left_wrist", "-right_wrist", "-all_cameras", "scene_swap", "-state", "swap_prompt"]
    rows, cams_to_plot = [], []
    t0 = _time.time()
    for si, s in enumerate(samples):
        base = transform_sample(inp, s)
        n_state = transform_sample(inp, s, state=mid_state)
        swap = transform_sample(inp, s, prompt=swapped_prompt(s["task_index"], tasks))
        # Scene swap: all 3 cameras from a sample of another episode/task (in-distribution images, wrong scene).
        other = samples[(si + len(samples) // 2) % len(samples)]
        items = []
        for v in variants:
            it = {**base, "image": dict(base["image"]), "image_mask": dict(base["image_mask"])}
            if v.startswith("-") and (v[1:] in CAMERA_NAMES or v == "-all_cameras"):
                for c in CAMERA_NAMES if v == "-all_cameras" else [v[1:]]:
                    key = MODEL_CAMERAS[CAMERA_NAMES.index(c)]
                    it["image_mask"][key] = np.False_
                    it["image"][key] = np.zeros_like(it["image"][key])
            elif v == "scene_swap":
                from openpi.shared import image_tools

                for raw_key, key in zip(CAMERA_KEYS, MODEL_CAMERAS):
                    h, w = base["image"][key].shape[:2]
                    it["image"][key] = image_tools.resize_with_pad(other["images"][raw_key], h, w)
            elif v == "-state":
                it["tokenized_prompt"], it["tokenized_prompt_mask"] = n_state["tokenized_prompt"], n_state["tokenized_prompt_mask"]
            elif v == "swap_prompt":
                it["tokenized_prompt"], it["tokenized_prompt_mask"] = swap["tokenized_prompt"], swap["tokenized_prompt_mask"]
            items.append(it)
        batch = to_model_batch(items)
        actions = np.repeat(base["actions"][None].astype(np.float32), len(variants), 0)

        # 1) losses of every variant, per action dim -> per joint group
        per_dim = backend.losses(batch, actions, noise, times)  # [V, D]
        row = {k: s[k] for k in ("episode", "frame", "progress", "task_index", "phase")}
        for vi, v in enumerate(variants):
            row[f"loss[{v}]"] = float(per_dim[vi, :REAL_DIM].mean())
            for g, sl in ACTION_GROUPS.items():
                row[f"loss[{v}]_{g}"] = float(per_dim[vi, sl].mean())

        # 2) sampled actions (base + swapped prompt) vs GT and the hold-still baseline, in robot units
        sampled = backend.sample(to_model_batch([items[0], items[-1]]), sample_noise)
        abs_actions = [out_tf({"state": base["state"], "actions": a})["actions"] for a in sampled]
        gt, hold = s["gt_actions"], np.repeat(s["state"][None], horizon, 0)
        for g, sl in ACTION_GROUPS.items():
            for hname, hz in [("short", SHORT_HORIZON), ("full", horizon)]:
                row[f"mae_{hname}_{g}"] = float(np.abs(abs_actions[0][:hz, sl] - gt[:hz, sl]).mean())
                row[f"hold_{hname}_{g}"] = float(np.abs(hold[:hz, sl] - gt[:hz, sl]).mean())
            row[f"swap_dact_short_{g}"] = float(np.abs(abs_actions[1][:SHORT_HORIZON, sl] - abs_actions[0][:SHORT_HORIZON, sl]).mean())

        # 3) Grad-CAM on the base sample
        one = to_model_batch([items[0]])
        state28 = np.asarray(base["state"])[:REAL_DIM]
        ids, word_map, tgroups = backend.text_groups(s["task"], state28)
        if not np.array_equal(ids, np.asarray(base["tokenized_prompt"])[np.asarray(base["tokenized_prompt_mask"], bool)]):
            raise RuntimeError("Prompt token reconstruction differs from the transformed sample")
        gc = backend.gradcam(one, base["actions"][None].astype(np.float32), noise, times, gmat)
        ego_img = np.asarray(base["image"]["base_0_rgb"])
        masks = octopus_masks(ego_img)
        valid = content_mask(ego_img)
        tok_valid = {c: token_valid(np.asarray(base["image"][k]), backend.grid) for c, k in zip(CAMERA_NAMES, MODEL_CAMERAS)}
        sample_maps = {}
        for target, (act, grad, extra) in gc.items():
            r = token_attribution(act, grad, extra, backend)
            maps = gradcam_maps(act, grad, extra, backend, tok_valid)
            sample_maps[target] = maps
            for k, v in input_group_shares(r, backend, tgroups).items():
                row[f"share[{target}]_{k}"] = v
            if target == "all":
                for w, v in word_shares(r, backend, word_map).items():
                    row[f"word_{w}"] = v
            up = upsample(maps["ego_view"], ego_img.shape[:2])
            for color, m in masks.items():
                row[f"lift[{target}]_{color}"] = cam_lift(up, m, valid)
        for color, m in masks.items():
            row[f"area_{color}"] = float(m.mean())
        rows.append(row)
        cams_to_plot.append({"sample": s, "images": {c: np.asarray(base["image"][k]) for c, k in zip(CAMERA_NAMES, MODEL_CAMERAS)},
                             "maps": sample_maps, "row": row})
        el = _time.time() - t0
        print(f"[{si + 1}/{len(samples)}] task {s['task_index']} ep {s['episode']} f {s['frame']} ({s['phase']}) "
              f"loss {row['loss[base]']:.4f}  skill(r_arm) {row['mae_short_right_arm'] / max(row['hold_short_right_arm'], 1e-9):.2f}"
              f"  eta {el / (si + 1) * (len(samples) - si - 1) / 60:.1f} min")

    df = pd.DataFrame(rows)
    df[[c for c in df.columns if c.startswith("word_")]] = df[[c for c in df.columns if c.startswith("word_")]].fillna(0.0)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    meta = {"checkpoint": str(args.checkpoint), "config": config_name, "model": backend.label, "kind": backend.kind,
            "n_samples": len(df), "n_noise": args.n_noise, "num_steps": args.num_steps, "val_dir": str(args.val_dir),
            "grid": backend.grid, "include_idle": bool(args.include_idle), "data_check": data_check}
    summary = summarize(df, meta)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    make_figures(df, cams_to_plot, out_dir, meta)
    save_cams(cams_to_plot, out_dir)
    write_report(summary, df, out_dir)
    print(f"\nReport: {out_dir / 'report.md'}")


# ============================================================================
# Summary + verdicts
# ============================================================================


def _rel(df, v, sub=None, group=None):
    d = df if sub is None else df[sub]
    if len(d) == 0:
        return float("nan")
    col = f"loss[{v}]" + (f"_{group}" if group else "")
    base = f"loss[base]" + (f"_{group}" if group else "")
    return float(d[col].mean() / d[base].mean() - 1.0)


def summarize(df, meta):
    s = {"meta": meta, "val_loss": float(df["loss[base]"].mean())}
    s["val_loss_by_group"] = {g: float(df[f"loss[base]_{g}"].mean()) for g in ACTION_GROUPS}
    acc = {}
    for g in ACTION_GROUPS:
        for h in ("short", "full"):
            m, b = df[f"mae_{h}_{g}"].mean(), df[f"hold_{h}_{g}"].mean()
            acc[f"{g}_{h}"] = {"mae": float(m), "hold_mae": float(b), "skill": float(m / max(b, 1e-9))}
    s["accuracy"] = acc
    s["accuracy_by_phase"] = {
        p: {g: float(d[f"mae_short_{g}"].mean() / max(d[f"hold_short_{g}"].mean(), 1e-9)) for g in ("right_arm", "right_hand")}
        for p, d in df.groupby("phase")}
    s["ablation"] = {v: {"all": _rel(df, v), **{g: _rel(df, v, group=g) for g in ("right_arm", "right_hand")}}
                     for v in ("-ego_view", "-left_wrist", "-right_wrist", "-all_cameras", "scene_swap", "-state")}
    # Instruction matters only where it disambiguates: frames with both octopi visible on the ego view.
    two = (df["area_pink"] >= MIN_OBJ_AREA) & (df["area_blue"] >= MIN_OBJ_AREA)
    one = ~two
    s["language"] = {
        "swap_two_object": _rel(df, "swap_prompt", two), "swap_single_object": _rel(df, "swap_prompt", one),
        "n_two_object": int(two.sum()),
        "swap_two_object_approach": _rel(df, "swap_prompt", two & (df["phase"] == "approach")),
        "swap_dact_short_right_arm_two_object": float(df.loc[two, "swap_dact_short_right_arm"].mean()) if two.any() else float("nan"),
        "gt_motion_short_right_arm": float(df["hold_short_right_arm"].mean()),
    }
    s["grad_shares"] = {t: {k: float(df[f"share[{t}]_{k}"].mean()) for k in INPUT_GROUPS} for t in CAM_TARGETS}
    s["grad_shares_by_phase"] = {p: {k: float(d[f"share[all]_{k}"].mean()) for k in INPUT_GROUPS} for p, d in df.groupby("phase")}
    s["word_shares"] = {c[5:]: float(df[c].mean()) for c in df.columns if c.startswith("word_")}

    # grounding: target object lift (single object: the visible one; two-object: instructed vs other, approach only)
    g = {}
    for t in ("all", "right_arm"):
        lifts, tgt, oth = [], [], []
        for _, r in df.iterrows():
            obj = {0: "pink", 1: "blue", 2: "pink", 3: "blue"}[r["task_index"]]
            other = "blue" if obj == "pink" else "pink"
            if r["phase"] == "approach":
                lifts.append(r[f"lift[{t}]_{obj}"])
                if r["area_pink"] >= MIN_OBJ_AREA and r["area_blue"] >= MIN_OBJ_AREA:
                    tgt.append(r[f"lift[{t}]_{obj}"])
                    oth.append(r[f"lift[{t}]_{other}"])
        lifts = np.array(lifts, float)
        pair = np.array([(a, b) for a, b in zip(tgt, oth) if np.isfinite(a) and np.isfinite(b)])
        g[t] = {"approach_target_lift_median": float(np.nanmedian(lifts)) if np.isfinite(lifts).any() else float("nan"),
                "approach_frac_lift_gt1": float(np.nanmean(lifts[np.isfinite(lifts)] > 1)) if np.isfinite(lifts).any() else float("nan"),
                "n": int(np.isfinite(lifts).sum()),
                "two_object_target_vs_other": [float(np.median(pair[:, 0])), float(np.median(pair[:, 1]))] if len(pair) else None,
                "two_object_frac_target_gt_other": float(np.mean(pair[:, 0] > pair[:, 1])) if len(pair) else float("nan"),
                "n_pairs": int(len(pair))}
    s["grounding"] = g
    s["verdicts"], s["actions"] = verdicts(s)
    return s


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.1f}%" if pct else f"{x:.3f}"


def verdicts(s):
    """Rule-based conclusions. Each verdict: (area, status OK/WARN/FAIL/INFO, text). Plus recommended actions."""
    v, acts = [], []
    frozen_vlm = s["meta"]["kind"] == "cosmos" and "frozen" in s["meta"]["model"]

    # --- action accuracy
    for g in ("right_arm", "right_hand"):
        sk = s["accuracy"][f"{g}_short"]["skill"]
        st = "OK" if sk < SKILL_GOOD else ("WARN" if sk < SKILL_USELESS else "FAIL")
        v.append(("accuracy", st, f"{g}: first {SHORT_HORIZON} steps MAE {s['accuracy'][f'{g}_short']['mae']:.4f} vs "
                                  f"hold-still {s['accuracy'][f'{g}_short']['hold_mae']:.4f} (skill {sk:.2f}; <{SKILL_GOOD} good, "
                                  f">{SKILL_USELESS} no better than not moving)"))
    worst_phase = max(s["accuracy_by_phase"].items(), key=lambda kv: kv[1]["right_arm"])
    v.append(("accuracy", "INFO", f"Hardest phase: '{worst_phase[0]}' (right-arm skill {worst_phase[1]['right_arm']:.2f}). "
                                  + ", ".join(f"{p}: {d['right_arm']:.2f}" for p, d in s["accuracy_by_phase"].items())))
    if s["accuracy"]["right_hand_short"]["skill"] >= SKILL_USELESS:
        acts.append("Right-hand (grasp) predictions are no better than holding still: check grasp timing in the data, "
                    "and weight/over-sample the grasp/release frames.")
    if worst_phase[1]["right_arm"] > SKILL_GOOD:
        acts.append(f"Errors concentrate in the '{worst_phase[0]}' phase: inspect samples/ for that phase and "
                    "consider more demonstrations or longer training on it.")

    # --- what is used (causal)
    abl = s["ablation"]
    cams = {c: abl[f"-{c}"]["all"] for c in CAMERA_NAMES}
    vision = max(abl["-all_cameras"]["all"], abl["scene_swap"]["all"])
    sd = abl["-state"]["all"]
    st = "FAIL" if vision < NEGLIGIBLE else ("WARN" if vision < 0.1 else "OK")
    v.append(("inputs", st, f"Vision: removing all cameras {_fmt(abl['-all_cameras']['all'], True)}, showing another "
                            f"episode's scene {_fmt(abl['scene_swap']['all'], True)} "
                            f"({'ignored' if st == 'FAIL' else 'weakly used' if st == 'WARN' else 'used'})."))
    v.append(("inputs", "INFO", "Single camera removed: " + ", ".join(f"{c} {_fmt(d, True)}" for c, d in cams.items())
              + (" -> the cameras are redundant; none is individually essential." if max(cams.values()) < NEGLIGIBLE <= vision
                 else "")))
    v.append(("inputs", "INFO", f"Removing the state from the prompt: loss {_fmt(sd, True)}"))
    if vision < NEGLIGIBLE:
        acts.append("Vision is not used. " + (
            "With a frozen VLM, unfreeze the vision merger / LoRA (stage 2) so the expert gets task-relevant features."
            if frozen_vlm else "Add image augmentation and object-position diversity; consider a higher-resolution ego view."))
    if sd > 3 * max(vision, NEGLIGIBLE):
        v.append(("inputs", "WARN", f"Proprioception dominates: the state matters {sd / max(vision, NEGLIGIBLE):.1f}x more than "
                                    "all cameras together. The policy largely continues its own trajectory, and reacts "
                                    "to the scene less than to where its arm is."))
        acts.append("Reduce state shortcutting: state dropout / noise on the discretised state during training, and "
                    "test with objects placed at positions not seen in training.")
    if cams["left_wrist"] >= max(cams["right_wrist"], NEGLIGIBLE):
        v.append(("inputs", "WARN", "The left wrist camera matters as much as or more than the right one, although only the "
                                    "right arm/hand do the task (left hand never moves): likely a spurious cue."))
        acts.append("Try dropping or randomly masking the left wrist camera during training.")

    # --- language
    lang = s["language"]
    sw = lang["swap_two_object"]
    if np.isfinite(sw):
        st = "FAIL" if sw < LANG_IGNORED else ("WARN" if sw < LANG_USED else "OK")
        word = "IGNORED" if st == "FAIL" else ("WEAK" if st == "WARN" else "USED")
        v.append(("language", st, f"Instruction {word} where both octopi are visible ({int(s['language']['n_two_object'])} frames): "
                                  "swapping the instruction raises loss "
                                  f"{_fmt(sw, True)} (approach phase {_fmt(lang['swap_two_object_approach'], True)}); "
                                  f"first-{SHORT_HORIZON}-step right-arm action changes by {lang['swap_dact_short_right_arm_two_object']:.4f} "
                                  f"(typical GT motion {lang['gt_motion_short_right_arm']:.4f})."))
        if st != "OK":
            acts.append("The policy picks the object from the scene, not from the instruction. More two-object "
                        "episodes with both orders, and/or prompt-swap negatives, are needed before trusting it on "
                        "multi-object scenes." + (" A frozen VLM's text features may also be too generic for the expert; "
                                                  "LoRA on the LLM (stage 2) is the next lever." if frozen_vlm else ""))
    v.append(("language", "INFO", f"Frames with only one octopus visible: swapping the instruction changes loss "
                                  f"{_fmt(lang['swap_single_object'], True)} (the scene alone determines the action there)."))

    # --- grounding
    gr = s["grounding"]["all"]
    lift = gr["approach_target_lift_median"]
    if np.isfinite(lift):
        st = "OK" if lift >= LIFT_GOOD else ("WARN" if lift >= LIFT_WEAK else "FAIL")
        v.append(("grounding", st, f"Before the grasp, Grad-CAM on the ego view puts {lift:.2f}x the uniform density on the "
                                   f"target octopus (median of {gr['n']} frames; {100 * gr['approach_frac_lift_gt1']:.0f}% of frames > 1)."))
        if st == "FAIL":
            acts.append("Grad-CAM does not concentrate on the object to pick: inspect samples/*approach*.png. If it "
                        "lights up the robot's own arm or the background, the model is tracking itself, not the object.")
    if gr["two_object_target_vs_other"]:
        a, b = gr["two_object_target_vs_other"]
        st = "OK" if gr["two_object_frac_target_gt_other"] > 0.6 else "WARN"
        v.append(("grounding", st, f"Both octopi visible, approach: lift on instructed object {a:.2f} vs other {b:.2f} "
                                   f"(instructed > other in {100 * gr['two_object_frac_target_gt_other']:.0f}% of {gr['n_pairs']} frames)."))

    # --- gradient vs ablation agreement
    gs = s["grad_shares"]["all"]
    top_grad = max(INPUT_GROUPS, key=lambda k: gs[k])
    v.append(("inputs", "INFO", f"Largest gradient attribution: {top_grad} ({100 * gs[top_grad]:.0f}%). Shares: "
                                + ", ".join(f"{k} {100 * gs[k]:.0f}%" for k in INPUT_GROUPS)))

    # --- data
    tr = s["meta"]["data_check"].get("train")
    if tr and tr["idle_frame_frac"] > 0.05:
        v.insert(0, ("data", "FAIL", f"{tr['n_idle']}/{tr['n_episodes']} train episodes ({100 * tr['idle_frame_frac']:.1f}% of "
                                     "frames) are idle: the right hand never grasps, the robot barely moves (empty table or "
                                     "a person handling the object), yet they carry a 'Pick up ...' prompt. They teach "
                                     "'stay still' under the same instruction and image distribution as real demos."))
        acts.insert(0, f"Remove the {tr['n_idle']} idle train episodes (and the val ones, which also make the val loss look "
                       f"better than it is) and retrain/fine-tune; list in summary.json -> meta.data_check. This is the "
                       "cheapest fix and most likely contributor to the state reliance below.")
    if not acts:
        acts.append("No blocking issue found on these val samples. Next: closed-loop evaluation on the robot.")
    return v, acts


# ============================================================================
# Figures
# ============================================================================

PALETTE = {"ego_view": "#4C72B0", "left_wrist": "#A0A0A0", "right_wrist": "#55A868", "prompt:task": "#C44E52",
           "prompt:state": "#DD8452", "prompt:template": "#D9D9D9"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 10, "figure.dpi": 100})


def _bbox(img):
    m = content_mask(img)
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _overlay(img, cam):
    """Heat fades in with intensity (cold regions keep the image), scaled to the 99.5th percentile, letterbox cropped."""
    valid = content_mask(img)
    up = upsample(cam, img.shape[:2]) * valid
    lo, hi = np.percentile(up[valid], [5, 99.5])
    u = np.clip((up - lo) / max(hi - lo, 1e-12), 0, 1) * valid
    a = (0.8 * u ** 0.7)[..., None]
    out = img.astype(np.float32) / 255.0 * (1 - a) + plt.get_cmap("turbo")(u)[..., :3] * a
    r, c = _bbox(img)
    return out[r, c]


def _show(ax, img, cam=None, title=None, contour=None):
    r, c = _bbox(img)
    ax.imshow(_overlay(img, cam) if cam is not None else img[r, c])
    if contour is not None and contour[r, c].any():
        ax.contour(contour[r, c], levels=[0.5], colors="white", linewidths=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight="normal")


def _cam_colorbar(fig, ax):
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Grad-CAM (relative)", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def _short(meta):
    names = {"pi05_g1_pickplace": "pi05", "cosmos2_8b_g1_pickplace": "Cosmos stage 1",
             "cosmos2_8b_lora_g1_pickplace": "Cosmos stage 2 (LoRA)", "cosmos2_8b_last2_g1_pickplace": "Cosmos last-N"}
    return f"{names.get(meta['config'], meta['config'])} @ {Path(meta['checkpoint']).name}"


def _rel_row(r, v):
    return 100 * (r[f"loss[{v}]"] / r["loss[base]"] - 1)


def sample_figure(c, path, tag):
    s, r, imgs, maps = c["sample"], c["row"], c["images"], c["maps"]
    obj = {0: "pink", 1: "blue", 2: "pink", 3: "blue"}[s["task_index"]]
    contour = octopus_masks(imgs["ego_view"])[obj]
    fig = plt.figure(figsize=(17.5, 7.6))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 0.05, 0.62, 0.95], wspace=0.06, hspace=0.18)
    _show(fig.add_subplot(gs[0, 0]), imgs["ego_view"], title=f"ego_view input ({obj} octopus outlined)", contour=contour)
    _show(fig.add_subplot(gs[0, 1]), imgs["ego_view"], maps["all"]["ego_view"], "ego_view · all joints", contour)
    _show(fig.add_subplot(gs[0, 2]), imgs["right_wrist"], maps["all"]["right_wrist"], "right_wrist · all joints")
    _show(fig.add_subplot(gs[1, 0]), imgs["ego_view"], maps["right_arm"]["ego_view"], "ego_view · right arm", contour)
    _show(fig.add_subplot(gs[1, 1]), imgs["ego_view"], maps["right_hand"]["ego_view"], "ego_view · right hand (grasp)", contour)
    _show(fig.add_subplot(gs[1, 2]), imgs["left_wrist"], maps["all"]["left_wrist"], "left_wrist · all joints")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=mcolors.Normalize(0, 1)), cax=fig.add_subplot(gs[:, 3]))
    cb.set_label("Grad-CAM (relative, per image)")

    ax = fig.add_subplot(gs[0, 5])
    vals = [r[f"share[all]_{k}"] for k in INPUT_GROUPS]
    ax.barh(INPUT_GROUPS[::-1], vals[::-1], color=[PALETTE[k] for k in INPUT_GROUPS[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{100 * v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title("Gradient share by input")
    ax = fig.add_subplot(gs[1, 5])
    names = ["-all_cameras", "scene_swap", "-state", "swap_prompt"]
    labels = ["no cameras", "other scene", "no state", "other instruction"]
    d = [_rel_row(r, v) for v in names]
    ax.barh(labels[::-1], d[::-1], color=["#C44E52", "#DD8452", "#55A868", "#4C72B0"])
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.9", zorder=0)
    for i, v in enumerate(d[::-1]):
        ax.text(v, i, f" {v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_title("Loss change when input removed")
    ax.set_xlabel("%")
    skill = r["mae_short_right_arm"] / max(r["hold_short_right_arm"], 1e-9)
    fig.suptitle(f"{tag}   |   task {s['task_index']} · ep {s['episode']} · frame {s['frame']} ({100 * s['progress']:.0f}% of episode) · "
                 f"{s['phase']}\n\"{s['task']}\"   loss {r['loss[base]']:.4f} · right-arm skill {skill:.2f} (vs hold still)",
                 fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_figures(df, cams_to_plot, out_dir, meta):
    tag = _short(meta)
    phases = [p for p in ("approach", "holding", "open (after grasp)") if p in set(df["phase"])]

    # --- dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    ax = axes[0, 0]
    left = np.zeros(3)
    for k in INPUT_GROUPS:
        vals = np.array([df[f"share[{t}]_{k}"].mean() for t in ("all", "right_arm", "right_hand")])
        ax.barh(["all joints", "right arm", "right hand"], vals, left=left, color=PALETTE[k], label=k, edgecolor="white")
        for i, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                ax.text(l + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Gradient attribution: what the loss is sensitive to")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)

    ax = axes[0, 1]
    abl = [("-ego_view", "no ego_view"), ("-left_wrist", "no left_wrist"), ("-right_wrist", "no right_wrist"),
           ("-all_cameras", "no cameras"), ("scene_swap", "other episode's scene"), ("-state", "no state"),
           ("swap_prompt", "other instruction")]
    d = np.array([100 * _rel(df, v) for v, _ in abl])
    colors = ["#4C72B0"] * 5 + ["#DD8452", "#C44E52"]
    ax.barh([lab for _, lab in abl], d, color=colors)
    ax.axvspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.92", zorder=0, label=f"negligible (±{100 * NEGLIGIBLE:.0f}%)")
    ax.axvline(0, color="k", lw=0.8)
    for i, v in enumerate(d):
        ax.text(v, i, f" {v:+.1f}% ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("flow-matching loss change (%)")
    ax.set_title("Causal test: what the model actually needs")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_xlim(min(d.min() * 1.3, -10), max(d.max() * 1.25, 10))

    ax = axes[0, 2]
    groups = ["right_arm", "right_hand", "left_arm"]
    x = np.arange(len(groups))
    for j, (h, lab, col) in enumerate([("short", f"first {SHORT_HORIZON} steps (0.5 s)", "#4C72B0"), ("full", "full 50-step chunk", "#8DA0CB")]):
        vals = [df[f"mae_{h}_{g}"].mean() / max(df[f"hold_{h}_{g}"].mean(), 1e-9) for g in groups]
        b = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.axhline(1, ls="--", color="#C44E52", lw=1, label="hold still (=1)")
    ax.axhspan(0, SKILL_GOOD, color="#55A868", alpha=0.08, label=f"good (<{SKILL_GOOD})")
    ax.set_xticks(x, groups)
    ax.set_ylabel("skill = MAE(model) / MAE(hold still)")
    ax.set_title("Action accuracy vs not moving (lower is better)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    x = np.arange(len(phases))
    for j, (g, col) in enumerate([("right_arm", "#4C72B0"), ("right_hand", "#55A868")]):
        vals = [df[df.phase == p][f"mae_short_{g}"].mean() / max(df[df.phase == p][f"hold_short_{g}"].mean(), 1e-9) for p in phases]
        b = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, label=g, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.axhline(1, ls="--", color="#C44E52", lw=1)
    ax.set_xticks(x, [f"{p}\n(n={int((df.phase == p).sum())})" for p in phases])
    ax.set_ylabel("skill (first 10 steps)")
    ax.set_title("Accuracy by task phase")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    bottom = np.zeros(len(phases))
    for k in INPUT_GROUPS:
        vals = np.array([df[df.phase == p][f"share[all]_{k}"].mean() for p in phases])
        ax.bar(phases, vals, bottom=bottom, color=PALETTE[k], label=k, edgecolor="white", width=0.6)
        for i, (b0, v) in enumerate(zip(bottom, vals)):
            if v > 0.06:
                ax.text(i, b0 + v / 2, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_title("Gradient attribution by task phase")

    ax = axes[1, 2]
    words = [c for c in df.columns if c.startswith("word_")]
    order = sorted(words, key=lambda w: -df[w].mean())
    tcols = ["#C44E52", "#4C72B0", "#DD8452", "#55A868"]
    for j, (ti, dd) in enumerate(df.groupby("task_index")):
        ax.bar(np.arange(len(order)) + (j - 1.5) * 0.2, [dd[w].mean() for w in order], 0.2, label=f"task {ti}", color=tcols[j % 4])
    ax.set_xticks(np.arange(len(order)), [w[5:] for w in order], rotation=30, ha="right")
    ax.set_ylabel("share among task words")
    ax.set_title("Which instruction words carry gradient")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"{tag}  ·  {meta['n_samples']} val frames  ·  {meta['model']}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_summary.png", dpi=130)
    plt.close(fig)

    # --- per-sample panels
    for c in cams_to_plot:
        s = c["sample"]
        sample_figure(c, out_dir / "samples" / f"task{s['task_index']}_ep{s['episode']:03d}_f{s['frame']:04d}_{s['phase'].split()[0]}.png", tag)

    # --- overview: one episode per task, ego view + right wrist over time
    first_eps = {}
    for c in cams_to_plot:
        first_eps.setdefault(c["sample"]["task_index"], c["sample"]["episode"])
    rows_ = [[c for c in cams_to_plot if c["sample"]["episode"] == ep] for _, ep in sorted(first_eps.items())]
    ncol = max(len(r) for r in rows_)
    fig, axes = plt.subplots(2 * len(rows_), ncol, figsize=(2.5 * ncol, 2.1 * 2 * len(rows_)), squeeze=False,
                             gridspec_kw={"wspace": 0.04, "hspace": 0.28})
    for i, rr in enumerate(rows_):
        for j in range(ncol):
            for k, cam in enumerate(("ego_view", "right_wrist")):
                ax = axes[2 * i + k, j]
                ax.axis("off")
                if j < len(rr):
                    c = rr[j]
                    obj = {0: "pink", 1: "blue", 2: "pink", 3: "blue"}[c["sample"]["task_index"]]
                    contour = octopus_masks(c["images"]["ego_view"])[obj] if cam == "ego_view" else None
                    _show(ax, c["images"][cam], c["maps"]["all"][cam],
                          f"{100 * c['sample']['progress']:.0f}% · {c['sample']['phase']}" if k == 0 else None, contour)
            axes[2 * i, 0].axis("on")
            axes[2 * i, 0].set_ylabel(f"task {rr[0]['sample']['task_index']}\nego_view", fontsize=9)
            axes[2 * i + 1, 0].axis("on")
            axes[2 * i + 1, 0].set_ylabel("right_wrist", fontsize=9)
            for a in (axes[2 * i, 0], axes[2 * i + 1, 0]):
                a.set_xticks([])
                a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
    fig.suptitle(f"Grad-CAM through one episode per task (score: all joints; outline = instructed octopus)  ·  {tag}",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "fig_gradcam_overview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_cams(cams_to_plot, out_dir):
    """Images + maps for `--compare` (side-by-side Grad-CAM of several checkpoints on the same frames)."""
    arrays, index = {}, []
    for i, c in enumerate(cams_to_plot):
        s = c["sample"]
        index.append({k: s[k] for k in ("episode", "frame", "progress", "task_index", "phase", "task")})
        for cam in CAMERA_NAMES:
            arrays[f"img_{i}_{cam}"] = c["images"][cam]
            for t in ("all", "right_arm", "right_hand"):
                arrays[f"map_{i}_{t}_{cam}"] = c["maps"][t][cam].astype(np.float32)
    np.savez_compressed(out_dir / "cams.npz", index=json.dumps(index), **arrays)


# ============================================================================
# Report
# ============================================================================


def write_report(s, df, out_dir):
    m = s["meta"]
    L = [f"# Grad-CAM / input analysis — {m['config']} @ {Path(m['checkpoint']).name}", "",
         f"- Checkpoint: `{m['checkpoint']}`", f"- Model: {m['model']}",
         f"- Val samples: {m['n_samples']} frames ({df['episode'].nunique()} episodes, tasks "
         f"{sorted(df['task_index'].unique().tolist())}), {m['n_noise']} noise/time draws per loss, {m['num_steps']} denoising steps",
         f"- Val flow-matching loss on these samples: **{s['val_loss']:.4f}** (right arm {s['val_loss_by_group']['right_arm']:.4f}, "
         f"right hand {s['val_loss_by_group']['right_hand']:.4f})", "",
         "## Verdicts", ""]
    for area, st, text in s["verdicts"]:
        L.append(f"- **[{st}] {area}** — {text}")
    L += ["", "## Recommended actions", ""] + [f"{i + 1}. {a}" for i, a in enumerate(s["actions"])]
    L += ["", "## Data check (idle episodes: right hand never moves)", ""]
    for k, d in m["data_check"].items():
        if d:
            L.append(f"- {k}: {d['n_idle']}/{d['n_episodes']} episodes, {100 * d['idle_frame_frac']:.1f}% of frames; by task: "
                     + ", ".join(f"'{t[:40]}…' {n}" for t, n in d["idle_by_task"].items()))
    L.append(f"- Idle val episodes are {'included in' if m['include_idle'] else 'excluded from'} the analysis below.")
    L += ["", "## Action accuracy (robot units, rad)", "",
          "| joint group | MAE first 10 | hold-still | skill | MAE full 50 | hold-still | skill |", "|---|---|---|---|---|---|---|"]
    for g in ACTION_GROUPS:
        a, b = s["accuracy"][f"{g}_short"], s["accuracy"][f"{g}_full"]
        L.append(f"| {g} | {a['mae']:.4f} | {a['hold_mae']:.4f} | {a['skill']:.2f} | {b['mae']:.4f} | {b['hold_mae']:.4f} | {b['skill']:.2f} |")
    L += ["", "Skill = MAE(model) / MAE(hold still). The left hand never moves in this data, so its skill is not meaningful.", "",
          "## What the model relies on", "",
          "| input | gradient share (all) | gradient share (right arm) | gradient share (right hand) | loss change when removed |",
          "|---|---|---|---|---|"]
    abl_name = {"ego_view": "-ego_view", "left_wrist": "-left_wrist", "right_wrist": "-right_wrist", "prompt:state": "-state",
                "prompt:task": "swap_prompt"}
    for k in INPUT_GROUPS:
        ab = s["ablation"].get(abl_name.get(k, ""), {}).get("all") if abl_name.get(k) != "swap_prompt" else \
            _rel(df, "swap_prompt")
        L.append(f"| {k} | {100 * s['grad_shares']['all'][k]:.1f}% | {100 * s['grad_shares']['right_arm'][k]:.1f}% | "
                 f"{100 * s['grad_shares']['right_hand'][k]:.1f}% | {_fmt(ab, True) if abl_name.get(k) else '—'} |")
    L += ["", f"All three cameras removed: **{_fmt(s['ablation']['-all_cameras']['all'], True)}**; cameras showing "
          f"another episode's scene: **{_fmt(s['ablation']['scene_swap']['all'], True)}** (right arm "
          f"{_fmt(s['ablation']['scene_swap']['right_arm'], True)}, right hand {_fmt(s['ablation']['scene_swap']['right_hand'], True)}).",
          "", "Prompt:task 'removed' = instruction swapped for the other task. Gradient shares show sensitivity; the loss "
          "change shows what the model actually needs. Trust a conclusion when both agree.", "",
          "### By phase (gradient share, all joints)", "", "| phase | " + " | ".join(INPUT_GROUPS) + " |",
          "|---|" + "---|" * len(INPUT_GROUPS)]
    for p, d in s["grad_shares_by_phase"].items():
        L.append(f"| {p} | " + " | ".join(f"{100 * d[k]:.0f}%" for k in INPUT_GROUPS) + " |")
    L += ["", "### Task words (share of gradient among task words)", "",
          ", ".join(f"{w}: {100 * v:.0f}%" for w, v in sorted(s["word_shares"].items(), key=lambda kv: -kv[1])), "",
          "## Figures", "", "![summary](fig_summary.png)", "", "![overview](fig_gradcam_overview.png)", "",
          "- `samples/` — every analysed frame: Grad-CAM on all 3 cameras, right-arm / right-hand scores, input reliance",
          "- `per_sample.csv`, `summary.json`, `cams.npz` — raw numbers and maps (`--compare` uses them)", "",
          "## Method and caveats", "",
          "- Score = −flow-matching loss on the ground-truth chunk (the training objective), averaged over fixed noise draws "
          "and stratified times t ~ Beta(1.5, 1). Grad-CAM is taken at the LLM input (vision-tower output; for Qwen3-VL "
          "also the DeepStack features), the closest analogue of the last conv layer. Map resolution: "
          f"{m['grid']}x{m['grid']} per camera.",
          "- Camera removal masks the camera's tokens (never seen in training, so it is an upper bound on reliance); "
          "state removal puts the mid-range state into the prompt; the GT target is unchanged in all variants.",
          "- Object masks come from colour thresholds (pink / cyan octopus) on the ego view; frames where the object is "
          f"not visible (area < {MIN_OBJ_AREA:.1%}) are skipped. Baskets are not segmented (their colours overlap the table).",
          "- Open-loop, single-frame analysis. It explains what the checkpoint uses; closed-loop success still needs a robot run."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


def compare(dirs, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = [json.load(open(Path(d) / "summary.json")) for d in dirs]
    names = [_short(r["meta"]) for r in runs]
    L = ["# Checkpoint comparison", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(runs)]

    def row(label, f):
        vals = []
        for r in runs:
            try:
                vals.append(f(r))
            except (KeyError, TypeError):
                vals.append("n/a")
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    row("val loss", lambda r: f"{r['val_loss']:.4f}")
    for g in ("right_arm", "right_hand"):
        row(f"skill {g} (first 10, lower better)", lambda r, g=g: f"{r['accuracy'][f'{g}_short']['skill']:.2f}")
    for v in ("-all_cameras", "scene_swap", "-state"):
        row(f"loss change {v}", lambda r, v=v: _fmt(r["ablation"][v]["all"], True))
    row("instruction swap, both octopi visible", lambda r: _fmt(r["language"]["swap_two_object"], True))
    row("target-object CAM lift (approach)", lambda r: _fmt(r["grounding"]["all"]["approach_target_lift_median"]))
    for k in INPUT_GROUPS:
        row(f"grad share {k}", lambda r, k=k: f"{100 * r['grad_shares']['all'][k]:.0f}%")
    L += ["", "![metrics](compare_metrics.png)", "", "![gradcam](compare_gradcam.png)", ""]
    L += ["## Verdicts per checkpoint", ""]
    for n, r in zip(names, runs):
        L += [f"### {n}", ""] + [f"- [{st}] {a}: {t}" for a, st, t in r["verdicts"]] + [""]
    out.write_text("\n".join(L) + "\n")

    # --- metrics figure
    rc = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [0.8, 1.3, 1]})
    n = len(runs)
    w = 0.8 / n
    cats = [("right_arm", "right arm"), ("right_hand", "right hand")]
    for j, r in enumerate(runs):
        b = axes[0].bar(np.arange(2) + (j - (n - 1) / 2) * w, [r["accuracy"][f"{g}_short"]["skill"] for g, _ in cats], w,
                        color=rc[j % 5], label=names[j])
        axes[0].bar_label(b, fmt="%.2f", fontsize=7)
    axes[0].axhline(1, ls="--", color="k", lw=0.8)
    axes[0].set_xticks(np.arange(2), [c for _, c in cats])
    axes[0].set_title("Skill vs hold-still (lower better)")
    abl = [("-all_cameras", "no cameras"), ("scene_swap", "other scene"), ("-state", "no state")]
    for j, r in enumerate(runs):
        vals = [100 * r["ablation"][v]["all"] for v, _ in abl] + [100 * r["language"]["swap_two_object"]]
        b = axes[1].bar(np.arange(4) + (j - (n - 1) / 2) * w, vals, w, color=rc[j % 5], label=names[j])
        axes[1].bar_label(b, fmt="%+.0f%%", fontsize=7)
    axes[1].axhspan(-100 * NEGLIGIBLE, 100 * NEGLIGIBLE, color="0.92", zorder=0)
    axes[1].set_xticks(np.arange(4), [lab for _, lab in abl] + ["other instruction\n(both octopi visible)"])
    axes[1].set_ylabel("loss change (%)")
    axes[1].set_title("Causal reliance on inputs")
    axes[1].legend(fontsize=8, frameon=False)
    left = np.zeros(n)
    for k in INPUT_GROUPS:
        vals = np.array([r["grad_shares"]["all"][k] for r in runs])
        axes[2].barh(names, vals, left=left, color=PALETTE[k], label=k, edgecolor="white")
        for i, (l, v) in enumerate(zip(left, vals)):
            if v > 0.06:
                axes[2].text(l + v / 2, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8)
        left += vals
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Gradient attribution share")
    axes[2].legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(out.parent / "compare_metrics.png", dpi=130)
    plt.close(fig)

    # --- Grad-CAM side by side on the same val frames
    cams = []
    for d in dirs:
        z = np.load(Path(d) / "cams.npz")
        cams.append((json.loads(str(z["index"])), z))
    keys = [{(e["episode"], e["frame"]) for e in idx} for idx, _ in cams]
    common = set.intersection(*keys)
    picks = []
    for t in sorted({e["task_index"] for e in cams[0][0]}):
        cands = [e for e in cams[0][0] if e["task_index"] == t and (e["episode"], e["frame"]) in common]
        for ph in ("approach", "holding"):
            hit = [e for e in cands if e["phase"] == ph]
            if hit:
                picks.append(hit[len(hit) // 2])
    if not picks:
        print("No common frames between runs; skipping compare_gradcam.png")
    else:
        fig, axes = plt.subplots(len(picks), 2 * n, figsize=(2.6 * 2 * n, 2.2 * len(picks)), squeeze=False,
                                 gridspec_kw={"wspace": 0.03, "hspace": 0.12})
        for i, e in enumerate(picks):
            obj = {0: "pink", 1: "blue", 2: "pink", 3: "blue"}[e["task_index"]]
            for j, (idx, z) in enumerate(cams):
                k = next(q for q, x in enumerate(idx) if (x["episode"], x["frame"]) == (e["episode"], e["frame"]))
                for m, cam in enumerate(("ego_view", "right_wrist")):
                    img = z[f"img_{k}_{cam}"]
                    contour = octopus_masks(img)[obj] if cam == "ego_view" else None
                    ax = axes[i, 2 * j + m]
                    _show(ax, img, z[f"map_{k}_all_{cam}"], f"{names[j]}\n{cam}" if i == 0 else None, contour)
            axes[i, 0].set_ylabel(f"task {e['task_index']}\n{e['phase']}\nep {e['episode']} f{e['frame']}", fontsize=8)
        fig.suptitle("Grad-CAM on identical val frames (score: all joints; outline = instructed octopus)",
                     fontsize=11, fontweight="bold")
        fig.savefig(out.parent / "compare_gradcam.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out} (+ compare_metrics.png, compare_gradcam.png)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", help="step dir: params/ (pi05) or model.safetensors (cosmos)")
    p.add_argument("--config_name", default=None, help="default: auto from the checkpoint")
    p.add_argument("--val_dir", default="val")
    p.add_argument("--asset", default="/mnt/drive2/vla_traning_ws/pravink_pi0/openpi/95000/95000/assets/simpk/single_pickplace/")
    p.add_argument("--train_dir", default="train", help="only scanned for idle episodes (data check)")
    p.add_argument("--include_idle", action="store_true", help="also analyse idle (no-grasp) val episodes")
    p.add_argument("--episodes_per_task", type=int, default=3)
    p.add_argument("--frames_per_episode", type=int, default=8)
    p.add_argument("--n_noise", type=int, default=8, help="noise/time draws averaged in every loss")
    p.add_argument("--num_steps", type=int, default=10, help="denoising steps when sampling actions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--compare", nargs="+", default=None, help="analysis dirs with summary.json to compare")
    p.add_argument("--compare_out", default="plots/gradcam/compare/compare.md")
    args = p.parse_args()
    if args.compare:
        compare(args.compare, args.compare_out)
    elif args.checkpoint:
        analyse(args)
    else:
        p.error("--checkpoint or --compare is required")


if __name__ == "__main__":
    main()
