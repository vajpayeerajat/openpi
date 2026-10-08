"""Attention visualization for the Cosmos-Reason2-8B (Qwen3-VL) pi05 policy -- PyTorch port of
rajat_text_visualization.py (which targets the JAX PaliGemma model).

Two attention sources, both REAL softmax attention (mean over heads and the chosen layers), not the cosine-similarity
proxy the PaliGemma script uses:

  vlm     Qwen3-VL decoder: prompt-text queries -> [image patches | text]. In stage 1 the VLM is frozen, so this is
          IDENTICAL for every stage-1 checkpoint (it is the pretrained Cosmos model).
  expert  Action expert: the 50 action tokens (averaged over denoising steps) -> [image patches | prompt | state].
          This is the part stage-1 training changes, so compare it across checkpoints.

Differences from the PaliGemma script that shape the outputs:
  * 3 camera slots (ego_view, left/right wrist), each 8x8 = 64 tokens at 256 px. The CSV video fills ego_view; the
    wrist slots are masked out unless the CSV has `left_wrist_path` / `right_wrist_path` columns.
  * The prefix is causal (images, then text), so image tokens can never attend to text: "text attention received
    from image patches" does not exist here. The word panels use the action expert instead.
  * Qwen puts a large "attention sink" on the first token (<|vision_start|>). Keys are restricted to the ego_view
    patches + prompt tokens and rows are renormalized, so maps show where attention goes among real content.
  * The model was trained with the discretized state in the prompt ("Task: ..., State: ...;\\nAction: "). Without a
    robot state for the video frames, --state zero uses the mid-range state (normalized 0); --state none drops it.

Usage:
  python rajat_cosmos_visualization.py \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/cosmos_as_vlm/cosmos2_8b_g1_pickplace/stage1/294 \
      --csv /mnt/nas/rajat_ws/openpi-et-robotics-implementation/data/meta.csv
"""

import argparse
import math
import os
from pathlib import Path
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from openpi.models.tokenizer import CosmosTokenizer
from openpi.models_pytorch.cosmos_pytorch import CosmosPi05Pytorch
from openpi.models_pytorch.cosmos_pytorch import _apply_rope
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.training import config as _config
from rajat_text_visualization import STOPWORDS
from rajat_text_visualization import generate_and_save_vlm_attention
from rajat_text_visualization import load_video_frames
from rajat_text_visualization import save_text_query_attention

NUM_CAMERAS = 3  # ego_view, left wrist, right wrist (order of the camera slots in the prefix)


# ============================================================================
# Prompt tokens
# ============================================================================


def tokenize_prompt(prompt: str, tokenizer: CosmosTokenizer, state: Optional[np.ndarray]):
    """Tokenizes exactly like training (CosmosTokenizer) and maps prompt words / token groups to token indices.

    Returns:
        ids:      [T] token ids (padding removed)
        pieces:   [T] readable token strings
        word_map: {word: [token indices]} for non-stopwords of the task text (state digits are excluded)
        groups:   {"task": [...], "state": [...], "other": [...]} token indices by role
    """
    ids, mask = tokenizer.tokenize(prompt, state)
    ids = np.asarray(ids)[np.asarray(mask).astype(bool)]
    hf_tok = tokenizer._tokenizer  # noqa: SLF001

    # Rebuild the same string CosmosTokenizer encoded, to get character offsets per token.
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        full = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
    else:
        full = f"{cleaned}\n"
    enc = hf_tok(full, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != ids.tolist():
        raise RuntimeError("Prompt reconstruction does not match CosmosTokenizer; update tokenize_prompt.")
    offsets = enc["offset_mapping"]

    task_start = full.index(cleaned)
    task_end = task_start + len(cleaned)
    state_start = full.find("State: ") + len("State: ") if state is not None else -1
    state_end = full.find(";", state_start) if state is not None else -1

    def tokens_in(lo, hi):
        return [i for i, (s, e) in enumerate(offsets) if s < hi and e > lo]

    task_idx = tokens_in(task_start, task_end)
    state_idx = tokens_in(state_start, state_end) if state is not None else []
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
        while key in word_map:  # repeated word -> "word#2"
            key, k = f"{w}#{k}", k + 1
        word_map[key] = idxs

    pieces = [p.replace("Ġ", "_").replace("Ċ", "\\n") for p in hf_tok.convert_ids_to_tokens(ids.tolist())]
    return ids, pieces, word_map, {"task": task_idx, "state": state_idx, "other": other_idx}


# ============================================================================
# Attention capture (mirrors CosmosPi05Pytorch.compute_prefix_kv / run_action_expert)
# ============================================================================


def _attn_probs(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax attention probabilities averaged over heads. q [B,H,Lq,D], k [B,Hkv,Lk,D], mask [B,1,Lq,Lk]."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)  # GQA
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1).mean(dim=1)  # [B, Lq, Lk]


def _vlm_layer_probs(model, i, h, cos, sin, attn_mask):
    layer = model.vlm.language_model.layers[i]
    attn = layer.self_attn
    b, n, _ = h.shape
    x = layer.input_layernorm(h)
    q = attn.q_norm(attn.q_proj(x).view(b, n, -1, model.head_dim)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(b, n, -1, model.head_dim)).transpose(1, 2)
    q, k = _apply_rope(q, k, cos, sin)
    return _attn_probs(q, k, attn_mask)


def _expert_layer_probs(layer, h, cond, cos, sin, prefix_k, attn_mask):
    b, n, _ = h.shape
    x, _ = layer.input_layernorm(h, cond)
    q = layer.q_norm(layer.q_proj(x).view(b, n, -1, layer.head_dim)).transpose(1, 2)
    k = layer.k_norm(layer.k_proj(x).view(b, n, -1, layer.head_dim)).transpose(1, 2)
    q, k = _apply_rope(q, k, cos, sin)
    k = torch.cat([prefix_k.to(k.dtype), k], dim=2)
    return _attn_probs(q, k, attn_mask)


@torch.no_grad()
def run_with_attention(model, images, img_masks, lang_tokens, lang_masks, vlm_layers, expert_layers, num_steps, seed):
    """Runs the prefix and the full denoising loop, returning
    vlm_attn    [B, P, P] prefix attention (mean over heads and vlm_layers)
    expert_attn [B, P]    action-token -> prefix attention (mean over heads, action tokens, steps, expert_layers)
    """
    device = lang_tokens.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        embs, pad_masks, att_masks, positions, next_pos, visual_index, deepstack = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        att_2d = att_2d | torch.eye(att_2d.shape[-1], dtype=torch.bool, device=device)[None]
        attn_mask = att_2d[:, None]
        cos, sin = model.vlm.language_model.rotary_emb(embs, positions)

        h, kv, vlm_sum = embs, [], None
        for i in range(model.num_llm_layers):
            if i in vlm_layers:
                p = _vlm_layer_probs(model, i, h, cos, sin, attn_mask)
                vlm_sum = p if vlm_sum is None else vlm_sum + p
            ds = deepstack[i] if i < len(deepstack) else None
            h, k, v = model._llm_layer(i, h, cos, sin, attn_mask, visual_index, ds)  # noqa: SLF001
            kv.append((k, v))

        # Denoising loop, as in sample_actions, recording the action expert's attention at every step.
        bsize, prefix_len = pad_masks.shape
        horizon, action_dim = model.config.action_horizon, model.config.action_dim
        gen = torch.Generator(device=device).manual_seed(seed)
        x_t = torch.randn(bsize, horizon, action_dim, device=device, generator=gen)
        prefix_part = pad_masks[:, None, :].expand(bsize, horizon, prefix_len)
        suffix_part = torch.ones(bsize, horizon, horizon, dtype=torch.bool, device=device)
        exp_mask = torch.cat([prefix_part, suffix_part], dim=2)[:, None]
        pos = next_pos[:, None] + torch.arange(horizon, device=device)[None, :]
        dt, time, exp_sum, n_exp = -1.0 / num_steps, 1.0, None, 0
        while time >= -dt / 2:
            t = torch.full((bsize,), time, dtype=torch.float32, device=device)
            tokens, cond = model.embed_suffix(x_t, t)
            ecos, esin = model.vlm.language_model.rotary_emb(tokens, pos[None].expand(3, -1, -1))
            hh = tokens
            for i, layer in enumerate(model.action_expert.layers):
                pk, pv = kv[i]
                if i in expert_layers:
                    p = _expert_layer_probs(layer, hh, cond, ecos, esin, pk, exp_mask)[:, :, :prefix_len].mean(1)
                    exp_sum = p if exp_sum is None else exp_sum + p
                    n_exp += 1
                hh = layer(hh, cond, ecos, esin, pk, pv, exp_mask)
            hh, _ = model.action_expert.norm(hh, cond)
            x_t = x_t + dt * model.action_out_proj(hh.float()).float()
            time += dt

    return (vlm_sum / len(vlm_layers)).float(), (exp_sum / n_exp).float()


# ============================================================================
# Reductions to [ego_view image | prompt text] and plots
# ============================================================================


def word_shares(token_scores: np.ndarray, word_map: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """[N, T] attention per token -> per-word share (sum over a word's tokens, words sum to 1 per frame)."""
    raw = {w: token_scores[:, idxs].sum(axis=1) for w, idxs in word_map.items()}
    if not raw:
        return {}
    total = np.sum(np.stack(list(raw.values())), axis=0) + 1e-12
    return {w: (v / total).astype(np.float32) for w, v in raw.items()}


def save_word_attention_plots(vlm_shares, expert_shares, output_path, title):
    """(1) mean share per word: VLM text->word vs action expert->word, (2)/(3) word x frame heatmaps."""
    words = list(vlm_shares)
    if not words:
        print("No words mapped to tokens; skipping word attention plot.")
        return
    fig, axes = plt.subplots(3, 1, figsize=(max(9, 0.7 * len(words) + 5), 11), gridspec_kw={"height_ratios": [1.3, 1, 1]})
    x, width = np.arange(len(words)), 0.38
    for off, data, label, color in [
        (-width / 2, vlm_shares, "VLM: from other prompt tokens", "tab:orange"),
        (width / 2, expert_shares, "action expert: from action tokens", "tab:green"),
    ]:
        axes[0].bar(x + off, [data[w].mean() for w in words], width, yerr=[data[w].std() for w in words],
                    capsize=3, label=label, color=color, edgecolor="black")
    axes[0].axhline(1.0 / len(words), ls="--", color="gray", lw=1, label="uniform share")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(words, rotation=30, ha="right")
    axes[0].set_ylabel("Attention share (mean ± std over frames)")
    axes[0].set_title("Attention received by each prompt word")
    axes[0].legend(fontsize=8)
    for ax, data, sub in [(axes[1], vlm_shares, "VLM text -> word"), (axes[2], expert_shares, "action expert -> word")]:
        im = ax.imshow(np.stack([data[w] for w in words]), aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel("Frame")
        ax.set_title(f"{sub} over time")
        fig.colorbar(im, ax=ax, label="share")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved word attention plot to: {output_path}")


def save_expert_group_plot(group_shares: Dict[str, np.ndarray], output_path: str, title: str):
    """Stacked area: where the action expert's attention goes per frame (image / task words / state / template)."""
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


def print_trainable_fingerprint(model):
    """Per-module checksum of the trained weights. Identical numbers between two checkpoints = module unchanged."""
    groups: Dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            g = n.split(".")[0]
            groups[g] = groups.get(g, 0.0) + float(p.detach().abs().float().reshape(-1)[:4096].sum())
    print("\n[fingerprint] trained weights (compare between checkpoints):")
    for g in sorted(groups):
        print(f"    {g:30s} {groups[g]:.10e}")
    print()


# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Cosmos pi05 attention visualization")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step dir containing model.safetensors")
    parser.add_argument("--config_name", default="cosmos2_8b_g1_pickplace")
    parser.add_argument("--csv", default="data/meta.csv", help="CSV with video_path, task, prompt "
                        "(optional left_wrist_path, right_wrist_path)")
    parser.add_argument("--vlm_layers", default="all", help="'all' or comma list of LLM layers to average, e.g. 20,27")
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config.get_config(args.config_name)
    model_cfg = config.model
    res = model_cfg.image_resolution

    step_number = Path(args.checkpoint.rstrip("/")).name
    run_tag = f"{args.config_name} @ {step_number}"
    out_dir = os.path.join("./plots", args.config_name, step_number, Path(args.csv).stem)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Files will be saved in: {out_dir}")

    print(f"Loading {model_cfg.vlm_path} + trained weights from {args.checkpoint}")
    model = CosmosPi05Pytorch(model_cfg)
    model.load_trainable_weights(os.path.join(args.checkpoint, "model.safetensors"))
    model.to(device).eval()
    print_trainable_fingerprint(model)

    def parse_layers(spec, n):
        return set(range(n)) if spec == "all" else {int(x) for x in spec.split(",")}

    vlm_layers = parse_layers(args.vlm_layers, model.num_llm_layers)
    expert_layers = parse_layers(args.expert_layers, len(model.action_expert.layers))
    tokenizer = CosmosTokenizer(model_cfg.max_token_len, model_cfg.vlm_path)
    tpi = model.tokens_per_image
    grid = model.llm_grid
    text_start = NUM_CAMERAS * (tpi + 2)  # prefix = [vs, img, ve] x 3 cameras, then the prompt
    ego_idx = np.arange(1, 1 + tpi)  # ego_view patches (camera slot 0)

    df = pd.read_csv(args.csv)
    csv_dir = Path(args.csv).resolve().parent
    state = np.zeros(args.state_dim, dtype=np.float32) if args.state == "zero" else None

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() or p.exists() else csv_dir.parent / p if (csv_dir.parent / p).exists() else csv_dir / p

    for _, row in df.iterrows():
        prompt, task = row["prompt"], row["task"]
        print(f"\n================ {task}: {prompt} ================")
        ids, pieces, word_map, groups = tokenize_prompt(prompt, tokenizer, state)
        n_txt = len(ids)
        print(f"Tokens ({n_txt}): {pieces}")
        print(f"Words -> token indices: {word_map}")

        cams = [load_video_frames(resolve(row["video_path"]), args.max_frames, args.frame_stride, (res, res))]
        for col in ("left_wrist_path", "right_wrist_path"):
            has = col in row and isinstance(row[col], str) and row[col]
            cams.append(load_video_frames(resolve(row[col]), args.max_frames, args.frame_stride, (res, res)) if has else None)
        n_frames = min(len(c) for c in cams if c is not None)
        video = np.stack(cams[0][:n_frames])

        lang = np.zeros(model_cfg.max_token_len, dtype=np.int64)
        lang[:n_txt] = ids
        lang_mask = np.arange(model_cfg.max_token_len) < n_txt
        txt_pos = text_start + np.arange(n_txt)
        keys_np = np.concatenate([ego_idx, txt_pos])  # [ego patches | prompt tokens]
        keys = torch.from_numpy(keys_np).to(device)

        vlm_img, word_img, vlm_txt_rows, exp_keys, txt_query_mats = [], [], [], [], []
        for s in tqdm(range(0, n_frames, args.batch_size), desc=task):
            e = min(s + args.batch_size, n_frames)
            b = e - s
            images, masks = [], []
            for c in cams:
                if c is None:
                    images.append(torch.zeros(b, 3, res, res, device=device))
                    masks.append(torch.zeros(b, dtype=torch.bool, device=device))
                else:
                    x = torch.from_numpy(np.stack(c[s:e])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
                    images.append(x)
                    masks.append(torch.ones(b, dtype=torch.bool, device=device))
            tok = torch.from_numpy(lang).to(device)[None].expand(b, -1)
            tok_mask = torch.from_numpy(lang_mask).to(device)[None].expand(b, -1)
            vlm_attn, exp_attn = run_with_attention(
                model, images, masks, tok, tok_mask, vlm_layers, expert_layers, args.num_steps, args.seed
            )

            # VLM: [ego | text] x [ego | text] sub-matrix, rows renormalized (drops sinks and masked cameras).
            sub = vlm_attn[:, keys][:, :, keys].cpu().numpy()
            sub = sub / (sub.sum(-1, keepdims=True) + 1e-12)
            txt_query_mats.append(sub)
            text_rows = sub[:, tpi:, :]  # [b, T, tpi + T]
            task_rows = text_rows[:, groups["task"]]
            vlm_img.append(task_rows[:, :, :tpi].mean(1))  # task words -> image, [b, tpi]
            word_img.append([
                {w: text_rows[f, idxs, :tpi].mean(0).reshape(grid, grid) for w, idxs in word_map.items()}
                for f in range(b)
            ])
            no_self = text_rows[:, :, tpi:] * (1 - np.eye(n_txt))[None]
            vlm_txt_rows.append(no_self.sum(1) / max(n_txt - 1, 1))  # [b, T] received from other text tokens
            exp_keys.append(exp_attn.cpu().numpy())  # [b, P] full prefix

        vlm_img = np.concatenate(vlm_img)
        word_img = [d for chunk in word_img for d in chunk]
        mats = np.concatenate(txt_query_mats)
        vlm_txt = np.concatenate(vlm_txt_rows)
        exp_full = np.concatenate(exp_keys)  # [N, P]

        # Action expert: share over the whole prefix by group, and over [ego | text] for the overlays.
        exp_groups = {"ego_view image": exp_full[:, ego_idx].sum(1)}
        for ci, name in [(1, "left wrist image"), (2, "right wrist image")]:
            if cams[ci] is not None:
                exp_groups[name] = exp_full[:, ci * (tpi + 2) + 1 : ci * (tpi + 2) + 1 + tpi].sum(1)
        for gname, label in [("task", "prompt: task words"), ("state", "prompt: state"), ("other", "prompt: template")]:
            if groups[gname]:
                exp_groups[label] = exp_full[:, text_start + np.asarray(groups[gname])].sum(1)
        accounted = np.sum(np.stack(list(exp_groups.values())), axis=0)
        exp_groups["special / sink tokens"] = np.clip(1.0 - accounted, 0.0, 1.0)
        exp_sub = exp_full[:, keys_np]
        exp_sub = exp_sub / (exp_sub.sum(-1, keepdims=True) + 1e-12)
        exp_img, exp_txt = exp_sub[:, :tpi], exp_sub[:, tpi:]

        vlm_shares = word_shares(vlm_txt, word_map)
        expert_shares = word_shares(exp_txt, word_map)

        # 1) VLM video: task-words -> image heatmap, per-word boxes, panel = expert word shares (what the policy reads)
        generate_and_save_vlm_attention(
            video_frames=video,
            vlm_patch_attention=vlm_img,
            word_attentions_per_frame=word_img,
            output_video_path=os.path.join(out_dir, f"vlm_prompt_attention_{task}.mp4"),
            output_grid_path=os.path.join(out_dir, f"vlm_prompt_attention_grid_{task}.png"),
            fps=args.fps,
            text_prompt=f"{prompt}  [{run_tag} | VLM]",
            word_text_shares_per_frame=[{w: float(v[i]) for w, v in vlm_shares.items()} for i in range(n_frames)],
            image_share_per_frame=mats[:, tpi:, :tpi].sum(-1).mean(-1),
            num_img_tokens=tpi,
            num_text_tokens=n_txt,
        )
        # 2) Action-expert video: action tokens -> image heatmap, panel = expert word shares
        generate_and_save_vlm_attention(
            video_frames=video,
            vlm_patch_attention=exp_img,
            word_attentions_per_frame=[{} for _ in range(n_frames)],
            output_video_path=os.path.join(out_dir, f"expert_attention_{task}.mp4"),
            output_grid_path=os.path.join(out_dir, f"expert_attention_grid_{task}.png"),
            fps=args.fps,
            text_prompt=f"{prompt}  [{run_tag} | action expert]",
            word_text_shares_per_frame=[{w: float(v[i]) for w, v in expert_shares.items()} for i in range(n_frames)],
            image_share_per_frame=exp_img.sum(-1),
            num_img_tokens=tpi,
            num_text_tokens=n_txt,
        )
        # 3) VLM text-query matrices (one frame and mean over frames)
        mid = n_frames // 2
        for mat, name, sub_title in [(mats[mid], f"frame{mid}", f"Frame {mid}"), (mats.mean(0), "mean", f"Mean over {n_frames} frames")]:
            save_text_query_attention(
                attn_matrix=mat, num_img_tokens=tpi, word_token_map=word_map, token_pieces=pieces,
                output_path=os.path.join(out_dir, f"text_query_attention_{name}_{task}.png"),
                title=f"{run_tag} | VLM | {sub_title} | Prompt: '{prompt}'",
            )
        # 4) Word shares, expert attention split, CSV
        save_word_attention_plots(vlm_shares, expert_shares, os.path.join(out_dir, f"word_attention_{task}.png"),
                                  f"Prompt: '{prompt}'  [{run_tag}]")
        save_expert_group_plot(exp_groups, os.path.join(out_dir, f"expert_attention_split_{task}.png"),
                               f"Action-expert attention by prefix part  [{run_tag}]")
        rows = []
        for i in range(n_frames):
            r = {"frame": i, **{f"expert__{g}": float(v[i]) for g, v in exp_groups.items()}}
            for w in word_map:
                r[f"{w}__vlm"] = float(vlm_shares[w][i])
                r[f"{w}__expert"] = float(expert_shares[w][i])
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"attention_{task}.csv"), index=False)
        print(f"Saved per-frame CSV to: {os.path.join(out_dir, f'attention_{task}.csv')}")


if __name__ == "__main__":
    main()
