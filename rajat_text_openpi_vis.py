"""Attention visualization for the openpi pi05 (PaliGemma, JAX) policy, with the same outputs as
rajat_cosmos_visualization.py: VLM attention AND action-expert attention.

Both attentions are REAL softmax attention (mean over heads and the chosen layers), not the cosine-similarity proxy of
rajat_text_visualization.py. The model's transformer runs inside a remat'ed nn.scan, so the probabilities cannot be
read out of it; this script re-runs the same layers functionally from the loaded weights (vision tower, embeddings,
action projections are the model's own modules) and keeps the probabilities. --verify checks that the actions it
samples match model.sample_actions.

  vlm     PaliGemma: prompt-text queries -> [image patches | text]. pi05's prefix is bidirectional, so image patches
          also attend to text ("text attention received from image patches" is meaningful here, unlike Cosmos).
  expert  Action expert: the 50 action tokens (averaged over denoising steps) -> [image | prompt words | state].
          Unlike the Cosmos stage-1 setup, pi05 fine-tunes the VLM too, so BOTH change between checkpoints.

Inputs: the CSV video fills base_0_rgb; wrist cameras are masked unless the CSV has left_wrist_path /
right_wrist_path. pi05 puts the discretized state in the prompt; there is no robot state for a plain video, so
--state zero uses the mid-range state (normalized 0); --state none drops it.

Usage:
  .venv/bin/python rajat_text_openpi_vis.py \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/pi05_g1_pickplace/run3/2000 \
      --csv /mnt/nas/rajat_ws/openpi-et-robotics-implementation/data/meta.csv
"""

import argparse
import os
from pathlib import Path
import re
from typing import Dict, List, Optional

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from openpi.models import model as _model
from openpi.models.gemma import _apply_rope
from openpi.models.gemma import _gated_residual
from openpi.models.pi0 import make_attn_mask
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as _config
from vis_utils import STOPWORDS
from vis_utils import generate_and_save_vlm_attention
from vis_utils import load_video_frames
from vis_utils import print_params_fingerprint
from vis_utils import save_text_query_attention

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")  # prefix order (model.embed_prefix iterates these)
BIG_NEG = -2.3819763e38  # same as gemma.Attention


# ============================================================================
# Prompt tokens
# ============================================================================


def tokenize_prompt(prompt: str, tokenizer: PaligemmaTokenizer, state: Optional[np.ndarray]):
    """Tokenizes exactly like training and maps prompt words / token groups to token indices.

    Returns ids [T] (padding removed, BOS included), pieces [T], word_map {word: [token idx]} for non-stopwords of the
    task text, and groups {"task", "state", "other"} of token indices.
    """
    ids, mask = tokenizer.tokenize(prompt, state)
    ids = np.asarray(ids)[np.asarray(mask).astype(bool)]
    sp = tokenizer._tokenizer  # noqa: SLF001

    def encode(text):  # (ids, char offsets); copied out while the proto is alive (its pieces are views into it)
        # proto = sp.encode_as_immutable_proto(text)
        proto = sp.encode(text, return_type="proto")
        return [p.id for p in proto.pieces], [(p.begin, p.end) for p in proto.pieces]

    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        full = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        tok_ids, tok_offsets = encode(full)
    else:
        full = cleaned
        tok_ids, tok_offsets = encode(cleaned)
        nl_ids, _ = encode("\n")  # encoded separately: its offsets are not into `full`
        tok_ids, tok_offsets = tok_ids + nl_ids, tok_offsets + [(-1, -1)] * len(nl_ids)
    # Token 0 is BOS (no characters); the rest carry character offsets into `full`.
    offsets = [(-1, -1)] + tok_offsets
    if [sp.bos_id()] + tok_ids != ids.tolist():
        raise RuntimeError("Prompt reconstruction does not match PaligemmaTokenizer; update tokenize_prompt.")

    task_start = full.index(cleaned)
    task_end = task_start + len(cleaned)

    def tokens_in(lo, hi):
        return [i for i, (s, e) in enumerate(offsets) if s >= 0 and s < hi and e > lo]

    task_idx = tokens_in(task_start, task_end)
    state_idx = []
    if state is not None:
        s0 = full.find("State: ") + len("State: ")
        state_idx = tokens_in(s0, full.find(";", s0))
    other_idx = [i for i in range(len(ids)) if i not in set(task_idx) | set(state_idx)]

    word_map: Dict[str, List[int]] = {}
    for m in re.finditer(r"[A-Za-z]+", cleaned):
        w = m.group().lower()
        if w in STOPWORDS or len(w) < 2:
            continue
        idxs = tokens_in(task_start + m.start(), task_start + m.end())
        if not idxs:
            continue
        key, k = w, 2
        while key in word_map:
            key, k = f"{w}#{k}", k + 1
        word_map[key] = idxs

    pieces = []
    for i in ids.tolist():
        p = sp.id_to_piece(int(i))
        pieces.append("\\n" if p in ("\n", "<0x0A>") else p.replace("▁", "_"))
    return ids, pieces, word_map, {"task": task_idx, "state": state_idx, "other": other_idx}


# ============================================================================
# Functional re-implementation of gemma.Module's layers (two experts), returning attention probabilities
# ============================================================================


def _rms_norm(x, p, cond):
    """gemma.RMSNorm: plain (p['scale']) or adaptive (p['Dense_0'], conditioned on cond)."""
    dtype = x.dtype
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
    if cond is None:
        return (normed * (1 + p["scale"])).astype(dtype), None
    mod = jnp.dot(cond.astype(dtype), p["Dense_0"]["kernel"].astype(dtype)) + p["Dense_0"]["bias"].astype(dtype)
    scale, shift, gate = jnp.split(mod[:, None, :], 3, axis=-1)
    return (normed * (1 + scale) + shift).astype(dtype), gate


def _sfx(i):
    return "" if i == 0 else f"_{i}"


@jax.jit
def _block(lp, xs, conds, positions, mask, kv_cache):
    """One gemma.Block for experts xs = [prefix | None, suffix | None]. Returns new xs, this call's (k, v) and the
    attention probabilities averaged over heads [B, T, S]."""
    dtype = next(x.dtype for x in xs if x is not None)
    pre, gates = [], []
    for i, x in enumerate(xs):
        if x is None:
            pre.append(None)
            gates.append(None)
            continue
        n, g = _rms_norm(x, lp[f"pre_attention_norm{_sfx(i)}"], conds[i])
        pre.append(n)
        gates.append(g)

    # --- gemma.Attention ---
    qs, ks, vs = [], [], []
    for i, x in enumerate(pre):
        if x is None:
            continue
        a = lp["attn"]
        qs.append(jnp.einsum("BTD,NDH->BTNH", x, a[f"q_einsum{_sfx(i)}"]["w"].astype(dtype)))
        k, v = jnp.einsum("BSD,2KDH->2BSKH", x, a[f"kv_einsum{_sfx(i)}"]["w"].astype(dtype))
        ks.append(k)
        vs.append(v)
    q, k, v = (jnp.concatenate(y, axis=1) for y in (qs, ks, vs))
    head_dim, num_kv = q.shape[-1], k.shape[2]
    q = _apply_rope(q, positions=positions)
    q *= head_dim**-0.5
    k = _apply_rope(k, positions=positions)
    if kv_cache is not None:
        k = jnp.concatenate([kv_cache[0], k], axis=1)
        v = jnp.concatenate([kv_cache[1], v], axis=1)
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
    probs_f32 = jax.nn.softmax(jnp.where(mask[:, :, None, :, :], logits, BIG_NEG), axis=-1)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_f32.astype(dtype), v)
    encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
    post, start = [], 0
    for i, x in enumerate(pre):
        if x is None:
            post.append(None)
            continue
        end = start + x.shape[1]
        w = lp["attn"][f"attn_vec_einsum{_sfx(i)}"]["w"].astype(dtype)
        post.append(jnp.einsum("BTNH,NHD->BTD", encoded[:, start:end], w))
        start = end

    xs = [_gated_residual(x, y, g) for x, y, g in zip(xs, post, gates, strict=True)]

    # --- FFN (lora.FeedForward without LoRA) ---
    out, gates = [], []
    for i, x in enumerate(xs):
        if x is None:
            out.append(None)
            gates.append(None)
            continue
        n, g = _rms_norm(x, lp[f"pre_ffw_norm{_sfx(i)}"], conds[i])
        mlp = lp[f"mlp{_sfx(i)}"]
        wg = mlp["gating_einsum"].astype(dtype)
        h = jax.nn.gelu(jnp.dot(n, wg[0]), approximate=True) * jnp.dot(n, wg[1])
        out.append(jnp.dot(h, mlp["linear"].astype(dtype)))
        gates.append(g)
    xs = [_gated_residual(x, y, g) for x, y, g in zip(xs, out, gates, strict=True)]
    return xs, (k, v), probs_f32.mean(axis=(1, 2))


def run_with_attention(model, llm, obs, noise, vlm_layers, expert_layers, num_steps):
    """Mirrors Pi0.sample_actions. Returns
    vlm_attn    [B, P, P] prefix attention (mean over heads and vlm_layers)
    expert_attn [B, P]    action-token -> prefix attention (mean over heads, action tokens, steps, expert_layers)
    actions     [B, H, D] sampled actions (for --verify)
    """
    obs = _model.preprocess_observation(None, obs, train=False)
    depth = llm["layers"]["attn"]["q_einsum"]["w"].shape[0]
    layer = [jax.tree.map(lambda a, i=i: a[i], llm["layers"]) for i in range(depth)]
    dtype = jnp.bfloat16

    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(obs)
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    x0, caches, vlm_sum = prefix_tokens.astype(dtype), [], 0.0
    for i in range(depth):
        (x0, _), kv, probs = _block(layer[i], [x0, None], [None, None], positions, prefix_attn[:, None], None)
        caches.append(kv)
        if i in vlm_layers:
            vlm_sum = vlm_sum + probs

    bsize, prefix_len = prefix_mask.shape
    dt, time, x_t, exp_sum, n_exp = -1.0 / num_steps, 1.0, noise, 0.0, 0
    while time >= -dt / 2:
        suffix_tokens, suffix_mask, suffix_ar, adarms = model.embed_suffix(obs, x_t, jnp.broadcast_to(time, bsize))
        full_attn = jnp.concatenate(
            [einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]), make_attn_mask(suffix_mask, suffix_ar)],
            axis=-1,
        )
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        x1 = suffix_tokens.astype(dtype)
        for i in range(depth):
            (_, x1), _, probs = _block(layer[i], [None, x1], [None, adarms], pos, full_attn[:, None], caches[i])
            if i in expert_layers:
                exp_sum = exp_sum + probs[:, :, :prefix_len].mean(axis=1)
                n_exp += 1
        x1, _ = _rms_norm(x1, llm["final_norm_1"], adarms)
        v_t = model.action_out_proj(x1[:, -model.action_horizon :])
        x_t = x_t + dt * v_t
        time += dt
    return (
        np.asarray(vlm_sum / len(vlm_layers), dtype=np.float32),
        np.asarray(exp_sum / n_exp, dtype=np.float32),
        np.asarray(x_t, dtype=np.float32),
    )


# ============================================================================
# Plots
# ============================================================================


def word_shares(token_scores: np.ndarray, word_map: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """[N, T] attention per token -> per-word share (sum over a word's tokens, words sum to 1 per frame)."""
    raw = {w: token_scores[:, idxs].sum(axis=1) for w, idxs in word_map.items()}
    if not raw:
        return {}
    total = np.sum(np.stack(list(raw.values())), axis=0) + 1e-12
    return {w: (v / total).astype(np.float32) for w, v in raw.items()}


def save_word_attention_plots(sources: Dict[str, Dict[str, np.ndarray]], output_path: str, title: str):
    """(1) mean share per word for every source, (2..) one word x frame heatmap per source."""
    names = list(sources)
    words = list(sources[names[0]])
    if not words:
        print("No words mapped to tokens; skipping word attention plot.")
        return
    fig, axes = plt.subplots(1 + len(names), 1, figsize=(max(9, 0.7 * len(words) + 5), 4 + 3.3 * len(names)),
                             gridspec_kw={"height_ratios": [1.3] + [1] * len(names)})
    x, width = np.arange(len(words)), 0.8 / len(names)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for j, name in enumerate(names):
        data = sources[name]
        axes[0].bar(x + (j - (len(names) - 1) / 2) * width, [data[w].mean() for w in words], width,
                    yerr=[data[w].std() for w in words], capsize=3, label=name, color=colors[j % 4], edgecolor="black")
    axes[0].axhline(1.0 / len(words), ls="--", color="gray", lw=1, label="uniform share")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(words, rotation=30, ha="right")
    axes[0].set_ylabel("Attention share (mean ± std over frames)")
    axes[0].set_title("Attention received by each prompt word")
    axes[0].legend(fontsize=8)
    for ax, name in zip(axes[1:], names):
        im = ax.imshow(np.stack([sources[name][w] for w in words]), aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel("Frame")
        ax.set_title(f"{name} over time")
        fig.colorbar(im, ax=ax, label="share")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved word attention plot to: {output_path}")


def save_expert_group_plot(group_shares: Dict[str, np.ndarray], output_path: str, title: str):
    """Stacked area: where the action expert's attention goes per frame (images / task words / state / template)."""
    names = list(group_shares)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(np.arange(len(group_shares[names[0]])), *[group_shares[n] for n in names], labels=names, alpha=0.85)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Share of action-expert attention")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved action-expert attention split to: {output_path}")


# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="openpi pi05 attention visualization (VLM + action expert)")
    parser.add_argument("--checkpoint", default="/mnt/nas/rajat_ws/checkpoints/unfrozen_vlm/pi05_g1_pickplace/830_episode_train_20000/15770/",
                        help="Checkpoint step dir containing params/")
    parser.add_argument("--config_name", default="pi05_g1_pickplace")
    parser.add_argument("--csv", default="data/meta.csv", help="CSV with video_path, task, prompt "
                        "(optional left_wrist_path, right_wrist_path)")
    parser.add_argument("--vlm_layers", default="all", help="'all' or comma list of layers to average, e.g. 12,17")
    parser.add_argument("--expert_layers", default="all", help="'all' or comma list of action-expert layers")
    parser.add_argument("--state", choices=["zero", "none"], default="zero",
                        help="zero: mid-range state in the prompt (as in training); none: prompt without state")
    parser.add_argument("--state_dim", default=28, type=int, help="State dims put in the prompt (28 for the G1)")
    parser.add_argument("--num_steps", default=10, type=int, help="Denoising steps (expert attention is averaged)")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--frame_stride", default=1, type=int, help="Use every n-th video frame")
    parser.add_argument("--max_frames", default=None, type=int)
    parser.add_argument("--fps", default=4, type=int, help="Output video fps")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--verify", action="store_true", help="Check sampled actions against model.sample_actions")
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    model_cfg = config.model
    step_number = Path(args.checkpoint.rstrip("/")).name
    run_tag = f"{args.config_name} @ {step_number}"
    out_dir = os.path.join("./plots", args.config_name, step_number, Path(args.csv).stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Files will be saved in: {out_dir}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    params = _model.restore_params(os.path.join(args.checkpoint, "params"))
    print_params_fingerprint(params)
    model = model_cfg.load(params)
    del params
    llm = nnx.state(model, nnx.Param).to_pure_dict()["PaliGemma"]["llm"]
    depth = llm["layers"]["attn"]["q_einsum"]["w"].shape[0]

    def parse_layers(spec):
        return set(range(depth)) if spec == "all" else {int(x) for x in spec.split(",")}

    vlm_layers, expert_layers = parse_layers(args.vlm_layers), parse_layers(args.expert_layers)
    tokenizer = PaligemmaTokenizer(model_cfg.max_token_len)
    res = _model.IMAGE_RESOLUTION
    df = pd.read_csv(args.csv)
    csv_dir = Path(args.csv).resolve().parent
    state = np.zeros(args.state_dim, dtype=np.float32) if args.state == "zero" else None

    def resolve(p):
        p = Path(p)
        if p.is_absolute() or p.exists():
            return p
        return csv_dir.parent / p if (csv_dir.parent / p).exists() else csv_dir / p

    for _, row in df.iterrows():
        prompt, task = row["prompt"], row["task"]
        print(f"\n================ {task}: {prompt} ================")
        ids, pieces, word_map, groups = tokenize_prompt(prompt, tokenizer, state)
        n_txt = len(ids)
        print(f"Tokens ({n_txt}): {pieces}")
        print(f"Words -> token indices: {word_map}")

        cams = [load_video_frames(resolve(row["video_path"]), args.max_frames, args.frame_stride, res)]
        for col in ("left_wrist_path", "right_wrist_path"):
            has = col in row and isinstance(row[col], str) and row[col]
            cams.append(load_video_frames(resolve(row[col]), args.max_frames, args.frame_stride, res) if has else None)
        n_frames = min(len(c) for c in cams if c is not None)
        video = np.stack(cams[0][:n_frames])

        tpi = None  # tokens per image, known after the first forward pass
        lang = np.zeros(model_cfg.max_token_len, dtype=np.int32)
        lang[:n_txt] = ids
        lang_mask = np.arange(model_cfg.max_token_len) < n_txt

        mats, vlm_img, word_img, from_text, from_vision, exp_full = [], [], [], [], [], []
        rng = jax.random.key(args.seed)
        for s in tqdm(range(0, n_frames, args.batch_size), desc=task):
            e = min(s + args.batch_size, n_frames)
            b = e - s
            images, masks = {}, {}
            for name, c in zip(CAMERAS, cams):
                if c is None:
                    images[name] = jnp.zeros((b, *res, 3), jnp.float32)
                    masks[name] = jnp.zeros((b,), jnp.bool_)
                else:
                    images[name] = jnp.asarray(np.stack(c[s:e]), jnp.float32) / 127.5 - 1.0
                    masks[name] = jnp.ones((b,), jnp.bool_)
            obs = _model.Observation(
                images=images, image_masks=masks, state=jnp.zeros((b, model_cfg.action_dim), jnp.float32),
                tokenized_prompt=jnp.asarray(np.tile(lang, (b, 1))),
                tokenized_prompt_mask=jnp.asarray(np.tile(lang_mask, (b, 1))),
            )
            noise = jax.random.normal(jax.random.fold_in(rng, s), (b, model_cfg.action_horizon, model_cfg.action_dim))
            vlm_attn, exp_attn, actions = run_with_attention(model, llm, obs, noise, vlm_layers, expert_layers,
                                                             args.num_steps)
            if args.verify and s == 0:
                ref = np.asarray(model.sample_actions(rng, obs, num_steps=args.num_steps, noise=noise), np.float32)
                print(f"[verify] max |actions - model.sample_actions| = {np.abs(actions - ref).max():.3e} "
                      f"(action scale {np.abs(ref).max():.3f})")

            if tpi is None:
                tpi = (vlm_attn.shape[-1] - model_cfg.max_token_len) // len(CAMERAS)
                grid = int(np.sqrt(tpi))
                text_start = len(CAMERAS) * tpi
                keys = np.concatenate([np.arange(tpi), text_start + np.arange(n_txt)])  # [base patches | prompt]

            # VLM: [base | text] x [base | text], rows renormalized (drops masked cameras / padding)
            sub = vlm_attn[:, keys][:, :, keys]
            sub = sub / (sub.sum(-1, keepdims=True) + 1e-12)
            mats.append(sub)
            text_rows = sub[:, tpi:, :]
            vlm_img.append(text_rows[:, groups["task"], :tpi].mean(1))
            word_img.extend(
                {w: text_rows[f, idxs, :tpi].mean(0).reshape(grid, grid) for w, idxs in word_map.items()}
                for f in range(b)
            )
            from_text.append((text_rows[:, :, tpi:] * (1 - np.eye(n_txt))[None]).sum(1) / max(n_txt - 1, 1))
            from_vision.append(sub[:, :tpi, tpi:].mean(1))
            exp_full.append(exp_attn)

        mats = np.concatenate(mats)
        vlm_img = np.concatenate(vlm_img)
        from_text, from_vision = np.concatenate(from_text), np.concatenate(from_vision)
        exp_full = np.concatenate(exp_full)

        # Action expert: share of the whole prefix per group, and over [base | text] for the overlays
        exp_groups = {"base image": exp_full[:, :tpi].sum(1)}
        for ci, name in [(1, "left wrist image"), (2, "right wrist image")]:
            if cams[ci] is not None:
                exp_groups[name] = exp_full[:, ci * tpi : (ci + 1) * tpi].sum(1)
        for gname, label in [("task", "prompt: task words"), ("state", "prompt: state"), ("other", "prompt: BOS/template")]:
            if groups[gname]:
                exp_groups[label] = exp_full[:, text_start + np.asarray(groups[gname])].sum(1)
        # exp_full only covers the prefix; the rest of each action token's attention goes to the action chunk itself
        exp_groups["action tokens (self)"] = np.clip(1.0 - exp_full.sum(1), 0.0, 1.0)
        exp_sub = exp_full[:, keys]
        exp_sub = exp_sub / (exp_sub.sum(-1, keepdims=True) + 1e-12)
        exp_img, exp_txt = exp_sub[:, :tpi], exp_sub[:, tpi:]

        vlm_text_shares = word_shares(from_text, word_map)
        vlm_vision_shares = word_shares(from_vision, word_map)
        expert_shares = word_shares(exp_txt, word_map)

        def per_frame(shares):
            return [{w: float(v[i]) for w, v in shares.items()} for i in range(n_frames)]

        # 1) VLM video: task words -> image heatmap, per-word boxes, panel = word shares received from image patches
        generate_and_save_vlm_attention(
            video_frames=video, vlm_patch_attention=vlm_img, word_attentions_per_frame=word_img,
            output_video_path=os.path.join(out_dir, f"vlm_prompt_attention_{task}.mp4"),
            output_grid_path=os.path.join(out_dir, f"vlm_prompt_attention_grid_{task}.png"),
            fps=args.fps, text_prompt=f"{prompt}  [{run_tag} | VLM]",
            word_text_shares_per_frame=per_frame(vlm_vision_shares),
            image_share_per_frame=mats[:, tpi:, :tpi].sum(-1).mean(-1),
            num_img_tokens=tpi, num_text_tokens=n_txt,
        )
        # 2) Action-expert video: action tokens -> image heatmap, panel = expert word shares
        generate_and_save_vlm_attention(
            video_frames=video, vlm_patch_attention=exp_img, word_attentions_per_frame=[{} for _ in range(n_frames)],
            output_video_path=os.path.join(out_dir, f"expert_attention_{task}.mp4"),
            output_grid_path=os.path.join(out_dir, f"expert_attention_grid_{task}.png"),
            fps=args.fps, text_prompt=f"{prompt}  [{run_tag} | action expert]",
            word_text_shares_per_frame=per_frame(expert_shares),
            image_share_per_frame=exp_img.sum(-1),
            num_img_tokens=tpi, num_text_tokens=n_txt,
        )
        # 3) VLM text-query matrices (one frame and mean over frames)
        mid = n_frames // 2
        for mat, name, sub_title in [(mats[mid], f"frame{mid}", f"Frame {mid}"),
                                     (mats.mean(0), "mean", f"Mean over {n_frames} frames")]:
            save_text_query_attention(
                attn_matrix=mat, num_img_tokens=tpi, word_token_map=word_map, token_pieces=pieces,
                output_path=os.path.join(out_dir, f"text_query_attention_{name}_{task}.png"),
                title=f"{run_tag} | VLM | {sub_title} | Prompt: '{prompt}'",
            )
        # 4) Word shares, expert attention split, CSV
        save_word_attention_plots(
            {"VLM: from image patches": vlm_vision_shares, "VLM: from other prompt tokens": vlm_text_shares,
             "action expert: from action tokens": expert_shares},
            os.path.join(out_dir, f"word_attention_{task}.png"), f"Prompt: '{prompt}'  [{run_tag}]",
        )
        save_expert_group_plot(exp_groups, os.path.join(out_dir, f"expert_attention_split_{task}.png"),
                               f"Action-expert attention by prefix part  [{run_tag}]")
        rows = []
        for i in range(n_frames):
            r = {"frame": i, **{f"expert__{g}": float(v[i]) for g, v in exp_groups.items()}}
            for w in word_map:
                r[f"{w}__vlm_from_vision"] = float(vlm_vision_shares[w][i])
                r[f"{w}__vlm_from_text"] = float(vlm_text_shares[w][i])
                r[f"{w}__expert"] = float(expert_shares[w][i])
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"attention_{task}.csv"), index=False)
        print(f"Saved per-frame CSV to: {os.path.join(out_dir, f'attention_{task}.csv')}")


if __name__ == "__main__":
    main()
