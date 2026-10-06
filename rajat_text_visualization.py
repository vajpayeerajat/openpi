import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import flax.nnx as nnx
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from openpi.models import model as _model
from openpi.training import config as _config
import pandas as pd
from openpi.shared import download

STOPWORDS = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "it", "is", "of", "up", "with", "for"}


def tokenize_prompt(prompt: str, pg_tokenizer) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """Tokenizes the prompt with openpi's PaligemmaTokenizer (SentencePiece).

    Returns:
        ids:      [T] token ids (padding removed), same tokens the model sees
        pieces:   [T] readable token strings, used as axis labels
        word_map: {word: [token indices]} for non-stopwords; a word split into several
                  SentencePiece pieces maps to all of them
    """
    ids, mask = pg_tokenizer.tokenize(prompt)
    ids = np.asarray(ids)[np.asarray(mask).astype(bool)]
    sp = pg_tokenizer._tokenizer
    raw_pieces = [sp.id_to_piece(int(i)) for i in ids]

    pieces = []
    for p in raw_pieces:
        if p in ("\n", "<0x0A>"):
            pieces.append("\\n")
        elif p.startswith("<") and p.endswith(">"):
            pieces.append(p)                 # e.g. <bos>
        else:
            pieces.append(p.replace("\u2581", "_"))  # SentencePiece word-start marker -> "_"

    # group pieces into words: a piece starting with the word-start marker opens a new word
    segments: List[Tuple[str, List[int]]] = []
    for i, p in enumerate(raw_pieces):
        if (p.startswith("<") and p.endswith(">")) or p in ("\n", "<0x0A>"):
            segments.append(("", []))        # special token -> breaks words
            continue
        if p.startswith("\u2581") or not segments or not segments[-1][1]:
            segments.append((p.replace("\u2581", ""), [i]))
        else:
            w, idxs = segments[-1]
            segments[-1] = (w + p, idxs + [i])

    word_map: Dict[str, List[int]] = {}
    for w, idxs in segments:
        w = w.strip(".,!?;:").lower()
        if not idxs or not w or w in STOPWORDS or len(w) < 2:
            continue
        key, k = w, 2
        while key in word_map:               # repeated word -> "word#2"
            key, k = f"{w}#{k}", k + 1
        word_map[key] = idxs

    return ids, pieces, word_map


def map_words_to_token_indices(prompt: str, tokenizer) -> Dict[str, List[int]]:
    """Dynamically maps each non-stopword in the prompt to its token indices."""
    word_to_indices = {}

    if tokenizer is not None:
        token_ids = tokenizer(prompt)["input_ids"]
        tokens = [tokenizer.decode([tid]).strip().lower() for tid in token_ids]
        clean_tokens = [t.lstrip(" ").lstrip("Ġ").lstrip(" ") for t in tokens]

        words = prompt.lower().split()
        for word in words:
            clean_word = word.strip(".,!?")
            if clean_word in STOPWORDS or len(clean_word) < 2:
                continue

            matching_indices = [
                idx for idx, t in enumerate(clean_tokens) if t and (t in clean_word or clean_word in t)
            ]
            if matching_indices:
                word_to_indices[clean_word] = matching_indices
    else:
        words = [w.strip(".,!?").lower() for w in prompt.split()]
        valid_words = [w for w in words if w not in STOPWORDS and len(w) >= 2]
        for idx, word in enumerate(valid_words):
            word_to_indices[word] = [idx + 1]

    return word_to_indices


def generate_distinct_color(index: int) -> Tuple[int, int, int]:
    """Generates distinct bright RGB colors for dynamic word labels."""
    colors = [
        (255, 105, 180),  # Pink
        (50, 205, 50),    # Lime Green
        (30, 144, 255),   # Dodger Blue
        (255, 165, 0),    # Orange
        (147, 112, 219),  # Purple
        (0, 255, 255),    # Cyan
        (255, 215, 0),    # Gold
    ]
    return colors[index % len(colors)]


def load_video_frames(
    video_path: Union[str, Path],
    max_frames: Optional[int] = None,
    fps_sample_rate: Optional[int] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> List[np.ndarray]:
    path = Path(video_path)
    if not path.exists():
        print(f"Warning: {path} not found. Generating dummy video frames...")
        target = target_size or (224, 224)
        return [np.random.randint(0, 255, size=(target[0], target[1], 3), dtype=np.uint8) for _ in range(10)]

    frames_iter = iio.imiter(path, plugin="pyav")
    frames = []
    for idx, frame in enumerate(frames_iter):
        if fps_sample_rate and idx % fps_sample_rate != 0:
            continue
        if target_size is not None:
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            frame = np.array(img)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break

    print(f"Loaded {len(frames)} frames from {path.name} (Frame shape: {frames[0].shape})")
    return frames


def process_smooth_heatmap(grid_16x16: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Creates a smooth spatial heatmap avoiding bfloat16 errors and dark blob artifacts."""
    orig_h, orig_w = target_size

    # Ensure float32 conversion to prevent OpenCV bfloat16 crashes
    grid_float32 = np.array(grid_16x16, dtype=np.float32)

    # Resample using bicubic interpolation
    heatmap = cv2.resize(grid_float32, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Smooth out sharp patch boundaries
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Percentile normalization to prevent single pixels from dominating the color map
    p_min, p_max = float(np.percentile(heatmap, 5)), float(np.percentile(heatmap, 98))
    if p_max > p_min:
        heatmap = (heatmap - p_min) / (p_max - p_min)
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap.astype(np.float32)


def locate_and_annotate_keywords(
    frame_rgb: np.ndarray,
    word_attentions: Dict[str, np.ndarray],
    confidence_threshold: float = 0.45,
) -> np.ndarray:
    """Dynamically annotates peak spatial attention for all words in word_attentions."""
    annotated = frame_rgb.copy()
    orig_h, orig_w, _ = annotated.shape

    for word_idx, (word, attn_grid) in enumerate(word_attentions.items()):
        smooth_map = process_smooth_heatmap(attn_grid, (orig_h, orig_w))
        _, max_val, _, max_loc = cv2.minMaxLoc(smooth_map)

        if max_val >= confidence_threshold:
            center_x, center_y = max_loc

            box_w, box_h = int(orig_w * 0.18), int(orig_h * 0.18)
            x1 = max(0, center_x - box_w // 2)
            y1 = max(0, center_y - box_h // 2)
            x2 = min(orig_w, x1 + box_w)
            y2 = min(orig_h, y1 + box_h)

            color = generate_distinct_color(word_idx)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.40
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(word, font, font_scale, thickness)

            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 6, max(text_h + 6, y1)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                word,
                (x1 + 3, max(text_h + 2, y1 - 3)),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return annotated


def draw_colorbar_legend(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    bar_width = int(w * 0.08)
    margin = int(w * 0.02)

    total_w = w + bar_width + margin * 2
    if total_w % 2 != 0:
        total_w += 1
    total_h = h if h % 2 == 0 else h + 1

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:h, :w] = image_rgb
    canvas[:, w:] = (30, 30, 30)

    gradient_h = int(h * 0.7)
    start_y = int(h * 0.15)
    gradient = np.linspace(255, 0, gradient_h, dtype=np.uint8)[:, None]
    gradient_bar = np.tile(gradient, (1, max(1, bar_width // 2)))

    color_bar = cv2.applyColorMap(gradient_bar, cv2.COLORMAP_JET)
    color_bar_rgb = cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB)

    bar_x = w + margin
    actual_bar_w = color_bar_rgb.shape[1]
    canvas[start_y : start_y + gradient_h, bar_x : bar_x + actual_bar_w] = color_bar_rgb

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    text_color = (255, 255, 255)
    thickness = 1

    cv2.putText(canvas, "High", (bar_x, start_y - int(h * 0.03)), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(canvas, "Low", (bar_x, start_y + gradient_h + int(h * 0.05)), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return canvas


# ============================================================================
# NEW: text-attention helpers
# ============================================================================

def compute_word_attention_shares(
    token_scores: np.ndarray,
    word_token_map: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Turns per-token received attention [N_frames, T_text] into per-word shares.

    A word's score is the SUM of attention received by its sub-word tokens (it is
    attention mass, so pieces add up). Scores are then normalised so that, in every
    frame, the words of the prompt sum to 1 -> "share of attention" per word.
    """
    n_frames, n_tokens = token_scores.shape
    raw = {}
    for word, idxs in word_token_map.items():
        valid = [i for i in idxs if 0 <= i < n_tokens]
        raw[word] = token_scores[:, valid].sum(axis=1) if valid else np.zeros(n_frames, dtype=np.float32)

    if not raw:
        return {}

    total = np.sum(np.stack(list(raw.values()), axis=0), axis=0) + 1e-12
    return {word: (vals / total).astype(np.float32) for word, vals in raw.items()}


def draw_text_attention_panel(
    image_rgb: np.ndarray,
    word_shares: Dict[str, float],
    title: str = "Image -> word attention",
) -> np.ndarray:
    """Appends a horizontal bar panel under the frame showing each word's attention share.

    Bar colors match the bounding-box colors (same word order / same palette).
    Bars are scaled relative to the largest word so differences stay visible;
    the printed number is the actual share.
    """
    if not word_shares:
        return image_rgb

    h, w, _ = image_rgb.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.33
    thickness = 1
    pad = 6
    header_h = 16
    row_h = 14

    panel_h = header_h + row_h * len(word_shares) + pad
    total_h = h + panel_h
    if total_h % 2 != 0:  # keep even dims for h264
        total_h += 1

    canvas = np.full((total_h, w, 3), 20, dtype=np.uint8)
    canvas[:h] = image_rgb

    cv2.putText(canvas, title, (pad, h + 12), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    label_w = max(cv2.getTextSize(word, font, font_scale, thickness)[0][0] for word in word_shares) + 2 * pad
    value_w = cv2.getTextSize("100%", font, font_scale, thickness)[0][0] + pad
    bar_max_w = max(10, w - label_w - value_w - pad)
    max_share = max(max(word_shares.values()), 1e-8)

    for i, (word, share) in enumerate(word_shares.items()):
        y_top = h + header_h + i * row_h
        y_text = y_top + row_h // 2 + 4
        color = generate_distinct_color(i)

        cv2.putText(canvas, word, (pad, y_text), font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA)
        bar_len = int(bar_max_w * share / max_share)
        cv2.rectangle(canvas, (label_w, y_top + 3), (label_w + bar_len, y_top + row_h - 3), color, -1)
        cv2.putText(
            canvas, f"{share * 100:.0f}%", (label_w + bar_max_w + 4, y_text),
            font, font_scale, (230, 230, 230), thickness, cv2.LINE_AA,
        )

    return canvas


def save_text_attention_plots(
    shares_from_vision: Dict[str, np.ndarray],
    shares_from_text: Dict[str, np.ndarray],
    output_path: str,
    text_prompt: str,
):
    """Saves a summary figure:
       (1) mean attention share per word (from image patches vs. from other text tokens)
       (2) word x frame heatmap of image -> word attention
       (3) word x frame heatmap of text -> word attention
    """
    words = list(shares_from_vision.keys())
    if not words:
        print("No words mapped to tokens; skipping text attention plot.")
        return

    fig, axes = plt.subplots(
        3, 1,
        figsize=(max(9, 0.7 * len(words) + 5), 11),
        gridspec_kw={"height_ratios": [1.3, 1, 1]},
    )

    # (1) Bar chart
    x = np.arange(len(words))
    width = 0.38
    vis_mean = [shares_from_vision[w].mean() for w in words]
    vis_std = [shares_from_vision[w].std() for w in words]
    txt_mean = [shares_from_text[w].mean() for w in words]
    txt_std = [shares_from_text[w].std() for w in words]

    ax = axes[0]
    ax.bar(x - width / 2, vis_mean, width, yerr=vis_std, capsize=3,
           label="received from image patches", color="tab:blue", edgecolor="black")
    ax.bar(x + width / 2, txt_mean, width, yerr=txt_std, capsize=3,
           label="received from other text tokens", color="tab:orange", edgecolor="black")
    ax.axhline(1.0 / len(words), ls="--", color="gray", lw=1, label="uniform share")
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=30, ha="right")
    ax.set_ylabel("Attention share (mean ± std over frames)")
    ax.set_title("Attention received by each prompt word")
    ax.legend(fontsize=8)

    # (2)/(3) Temporal heatmaps
    for ax, data, title in zip(
        axes[1:],
        [shares_from_vision, shares_from_text],
        ["Image -> word attention over time", "Text -> word attention over time"],
    ):
        mat = np.stack([data[w] for w in words], axis=0)  # [W, N]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel("Frame")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="share")

    fig.suptitle(f"Prompt: '{text_prompt}'", fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved text attention plot to: {output_path}")


def save_text_attention_csv(
    shares_from_vision: Dict[str, np.ndarray],
    shares_from_text: Dict[str, np.ndarray],
    output_path: str,
):
    """Per-frame per-word shares, for your own analysis."""
    if not shares_from_vision:
        return
    n_frames = len(next(iter(shares_from_vision.values())))
    rows = []
    for f in range(n_frames):
        row = {"frame": f}
        for w in shares_from_vision:
            row[f"{w}__from_vision"] = float(shares_from_vision[w][f])
            row[f"{w}__from_text"] = float(shares_from_text[w][f])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved text attention CSV to: {output_path}")


# ============================================================================
# NEW: image vs. text (joint) attention helpers
# ============================================================================

def compute_image_text_stats(attn_weights: np.ndarray, num_img_tokens: int) -> Dict[str, np.ndarray]:
    """Summarises where attention goes across the full [image | text] sequence.

    attn_weights: [B, V+T, V+T] (rows = queries, cols = keys, rows sum to 1)

    Returns:
        received:   [B, V+T]  attention each token receives, averaged over ALL queries
                              (image + text). Sums to 1 per frame.
        block_mass: [B, 2, 2] rows = query type (0=image, 1=text),
                              cols = key type   (0=image, 1=text).
                              Each row sums to 1: e.g. block_mass[:, 1, 0] = fraction of
                              text-query attention that lands on image tokens.
    """
    V = num_img_tokens
    received = attn_weights.mean(axis=1)

    img_q = attn_weights[:, :V, :]
    txt_q = attn_weights[:, V:, :]
    block_mass = np.stack(
        [
            np.stack([img_q[..., :V].sum(-1).mean(-1), img_q[..., V:].sum(-1).mean(-1)], axis=-1),
            np.stack([txt_q[..., :V].sum(-1).mean(-1), txt_q[..., V:].sum(-1).mean(-1)], axis=-1),
        ],
        axis=1,
    )
    return {"received": received.astype(np.float32), "block_mass": block_mass.astype(np.float32)}


def build_text_token_labels(
    num_text_tokens: int,
    word_token_map: Dict[str, List[int]],
    token_pieces: Optional[List[str]] = None,
) -> List[str]:
    """Labels every text token. With real tokens: its SentencePiece string.
    Otherwise: its word, or 't<i>' for unmapped tokens."""
    if token_pieces is not None and len(token_pieces) == num_text_tokens:
        return list(token_pieces)
    labels = [f"t{i}" for i in range(num_text_tokens)]
    for word, idxs in word_token_map.items():
        for i in idxs:
            if 0 <= i < num_text_tokens:
                labels[i] = word
    return labels


def draw_image_text_split_panel(image_rgb: np.ndarray, image_share: float, num_img: int, num_txt: int) -> np.ndarray:
    """Appends a stacked bar: share of total attention on image tokens vs. text tokens,
    plus the per-token ratio (which corrects for there being far more image tokens)."""
    h, w, _ = image_rgb.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.33
    thickness = 1
    pad = 6
    panel_h = 40
    total_h = h + panel_h
    if total_h % 2 != 0:
        total_h += 1

    canvas = np.full((total_h, w, 3), 20, dtype=np.uint8)
    canvas[:h] = image_rgb

    text_share = 1.0 - image_share
    per_img = image_share / max(num_img, 1)
    per_txt = text_share / max(num_txt, 1)
    ratio = per_txt / max(per_img, 1e-12)

    cv2.putText(
        canvas, f"Img {image_share*100:.1f}% | Txt {text_share*100:.1f}% | per-tok txt/img x{ratio:.2f}",
        (pad, h + 12), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA,
    )

    bar_x0, bar_x1 = pad, w - pad
    bar_y0, bar_y1 = h + 18, h + 34
    split = bar_x0 + int((bar_x1 - bar_x0) * image_share)
    cv2.rectangle(canvas, (bar_x0, bar_y0), (split, bar_y1), (30, 144, 255), -1)   # image = blue
    cv2.rectangle(canvas, (split, bar_y0), (bar_x1, bar_y1), (255, 165, 0), -1)    # text  = orange

    # tick where the split would be if attention were uniform over tokens
    uniform_x = bar_x0 + int((bar_x1 - bar_x0) * num_img / max(num_img + num_txt, 1))
    cv2.line(canvas, (uniform_x, bar_y0 - 2), (uniform_x, bar_y1 + 2), (255, 255, 255), 1)

    return canvas


def save_text_query_attention(
    attn_matrix: np.ndarray,
    num_img_tokens: int,
    word_token_map: Dict[str, List[int]],
    output_path: str,
    title: str,
    token_pieces: Optional[List[str]] = None,
):
    """Horizontal view: text tokens (queries) on the y-axis, ALL tokens (keys) on the x-axis.

    Columns are split into [ IMAGE patches | TEXT tokens ] with the text block widened so
    it stays readable. Values are in "x uniform" units (weight * num_tokens):
    1.0 = fair share, red = above, blue = below.
    Right panel: per-token text/image ratio per row (>1 = this token prefers text tokens).
    """
    from matplotlib.colors import TwoSlopeNorm

    m = np.asarray(attn_matrix, dtype=np.float32)
    n = m.shape[0]
    V = num_img_tokens
    T = n - V
    labels = build_text_token_labels(T, word_token_map, token_pieces)

    rows = m[V:, :] * n            # [T, V+T] in x uniform
    img_part, txt_part = rows[:, :V], rows[:, V:]

    lo, hi = float(np.percentile(rows, 1)), float(np.percentile(rows, 99.5))
    norm = TwoSlopeNorm(vmin=min(lo, 0.999), vcenter=1.0, vmax=max(hi, 1.001))
    cmap = "coolwarm"

    fig = plt.figure(figsize=(22, 0.45 * T + 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[9, max(1.8, 0.3 * T), 2.4, 0.25], wspace=0.06)
    ax_i = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharey=ax_i)
    ax_s = fig.add_subplot(gs[2], sharey=ax_i)
    cax = fig.add_subplot(gs[3])

    # --- image keys ---
    im = ax_i.imshow(img_part, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    grid = int(np.sqrt(V))
    for x in range(grid, V, grid):
        ax_i.axvline(x - 0.5, color="black", lw=0.3, alpha=0.4)
    ax_i.set_xticks([r * grid + grid / 2 - 0.5 for r in range(grid)])
    ax_i.set_xticklabels([f"r{r}" for r in range(grid)], fontsize=7)
    ax_i.set_yticks(range(T))
    ax_i.set_yticklabels(labels, fontsize=9)
    ax_i.set_xlabel(f"IMAGE patches ({V}), grouped by patch row")
    ax_i.set_ylabel("Text token (query)")
    ax_i.set_title("text -> image")

    # --- text keys ---
    ax_t.imshow(txt_part, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax_t.set_xticks(range(T))
    ax_t.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax_t.tick_params(axis="y", labelleft=False)
    ax_t.set_xlabel(f"TEXT ({T})")
    ax_t.set_title("text -> text")
    if T <= 24:
        for r in range(T):
            for c in range(T):
                ax_t.text(c, r, f"{txt_part[r, c]:.2f}", ha="center", va="center", fontsize=6,
                          fontweight="bold" if r == c else "normal")

    # --- per-row summary ---
    img_mass = m[V:, :V].sum(axis=1)
    txt_mass = m[V:, V:].sum(axis=1)
    ratio = (txt_mass / max(T, 1)) / np.maximum(img_mass / V, 1e-12)
    ax_s.barh(range(T), ratio, color=["tab:orange" if r > 1 else "tab:blue" for r in ratio], edgecolor="black")
    ax_s.axvline(1.0, ls="--", color="gray", lw=1)
    for r in range(T):
        ax_s.text(ratio[r], r, f" x{ratio[r]:.3f} | img {img_mass[r]*100:.1f}%", va="center", fontsize=7)
    ax_s.tick_params(axis="y", labelleft=False)
    span = max(abs(ratio - 1).max(), 1e-3)
    ax_s.set_xlim(1 - 1.5 * span if ratio.min() < 1 else 0.0, 1 + 3.5 * span)
    ax_s.set_xlabel("per-token text/image\n(>1 prefers text)")
    ax_s.set_title("image vs text")

    fig.colorbar(im, cax=cax, label="x uniform (1 = fair share)")
    fig.suptitle(title, fontsize=12)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved text-query attention plot to: {output_path}")


# ============================================================================


def extract_prompt_conditioned_attention(
    image_embeddings: jnp.ndarray,
    text_embeddings: jnp.ndarray,
    llm_backbone: nnx.Module,
    temperature: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes prompt-conditioned attention weights from LLM hidden states.

    Returns:
        prompt_conditioned_vis_attn: [B, V]   mean text -> vision attention (spatial heatmap)
        text_received_from_text:     [B, T]   attention each text token receives from the
                                              OTHER text tokens (self-attention excluded)
        text_received_from_vision:   [B, T]   attention each text token receives from the
                                              image patches (mean over patches)
        text_to_visual_attn:         [B, T, V] raw text -> vision attention
    """
    b, num_img_tokens, d = image_embeddings.shape
    _, num_text_tokens, _ = text_embeddings.shape
    total_tokens = num_img_tokens + num_text_tokens

    # Enforce float32 types
    image_embeddings = image_embeddings.astype(jnp.float32)
    text_embeddings = text_embeddings.astype(jnp.float32)

    full_sequence = jnp.concatenate([image_embeddings, text_embeddings], axis=1)

    embedded = [full_sequence, None]
    positions = jnp.tile(jnp.arange(total_tokens, dtype=jnp.int32)[None, :], (b, 1))
    mask = jnp.ones((b, total_tokens, total_tokens), dtype=jnp.bool_)
    adarms_cond = [None, None]

    llm_outputs = llm_backbone(
        embedded=embedded,
        positions=positions,
        mask=mask,
        adarms_cond=adarms_cond,
        deterministic=True,
    )

    hidden_states = llm_outputs[0][0] if isinstance(llm_outputs[0], list) else llm_outputs[0]
    hidden_states = hidden_states.astype(jnp.float32)

    norm_states = hidden_states / (jnp.linalg.norm(hidden_states, axis=-1, keepdims=True) + 1e-8)
    scale = temperature if temperature is not None else jnp.sqrt(float(d))  # None = original behaviour
    attn_matrix = jnp.matmul(norm_states, norm_states.transpose(0, 2, 1)) / scale

    attn_weights = jax.nn.softmax(attn_matrix, axis=-1)

    # [B, T_text, V] : rows = text queries, cols = image keys
    text_to_visual_attn = attn_weights[:, num_img_tokens:, :num_img_tokens]
    prompt_conditioned_vis_attn = jnp.mean(text_to_visual_attn, axis=1)

    # [B, T_text, T_text] : how much each text token is attended to by other text tokens.
    # The diagonal (token vs itself, cosine sim = 1) always dominates, so drop it.
    text_to_text_attn = attn_weights[:, num_img_tokens:, num_img_tokens:]
    eye = jnp.eye(num_text_tokens, dtype=jnp.bool_)[None]
    text_to_text_no_self = jnp.where(eye, 0.0, text_to_text_attn)
    text_received_from_text = text_to_text_no_self.sum(axis=1) / max(num_text_tokens - 1, 1)

    # [B, V, T_text] : how much each text token is attended to by the image patches.
    vision_to_text_attn = attn_weights[:, :num_img_tokens, num_img_tokens:]
    text_received_from_vision = jnp.mean(vision_to_text_attn, axis=1)

    # attn_weights (full [B, V+T, V+T] matrix) is returned so image-vs-text stats can be computed.
    return (
        prompt_conditioned_vis_attn,
        text_received_from_text,
        text_received_from_vision,
        text_to_visual_attn,
        attn_weights,
    )


def generate_and_save_vlm_attention(
    video_frames: np.ndarray,
    vlm_patch_attention: np.ndarray,
    word_attentions_per_frame: List[Dict[str, np.ndarray]],
    output_video_path: str = "vlm_prompt_attention.mp4",
    output_grid_path: str = "vlm_prompt_attention_grid.png",
    fps: int = 10,
    text_prompt: str = "Prompt not provided",
    word_text_shares_per_frame: Optional[List[Dict[str, float]]] = None,
    image_share_per_frame: Optional[np.ndarray] = None,
    num_img_tokens: int = 256,
    num_text_tokens: int = 1,
):
    total_samples = len(video_frames)
    orig_h, orig_w = video_frames.shape[1], video_frames.shape[2]
    overlay_frames = []

    for i in tqdm(range(total_samples)):
        importance = np.array(vlm_patch_attention[i], dtype=np.float32)
        grid_dim = int(np.sqrt(importance.shape[0]))  # 16x16
        grid_2d = importance.reshape((grid_dim, grid_dim))

        # Generate smooth heatmap
        heatmap_norm = process_smooth_heatmap(grid_2d, (orig_h, orig_w))

        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(video_frames[i], 0.55, heatmap_rgb, 0.45, 0)

        # Draw object bounding box overlays with word text tags
        annotated_frame = locate_and_annotate_keywords(blended, word_attentions_per_frame[i])

        frame_with_scale = draw_colorbar_legend(annotated_frame)

        # NEW: per-word text attention bars under the frame
        if word_text_shares_per_frame is not None:
            frame_with_scale = draw_text_attention_panel(frame_with_scale, word_text_shares_per_frame[i])

        # NEW: image vs text split bar
        if image_share_per_frame is not None:
            frame_with_scale = draw_image_text_split_panel(
                frame_with_scale, float(image_share_per_frame[i]), num_img_tokens, num_text_tokens
            )

        overlay_frames.append(frame_with_scale)

    overlay_array = np.ascontiguousarray(np.stack(overlay_frames), dtype=np.uint8)

    try:
        iio.imwrite(output_video_path, overlay_array, fps=fps, plugin="pyav", codec="h264")
        print(f"Saved prompt-conditioned attention video to: {output_video_path}")
    except Exception as err:
        print(f"PyAV fallback to OpenCV VideoWriter due to: {err}")
        out_h, out_w, _ = overlay_array[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
        for frame in overlay_array:
            out_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_writer.release()
        print(f"Saved attention video via OpenCV to: {output_video_path}")

    # Export Clean PNG Comparison Grid (Fixes Text Overlap)
    max_cols = min(total_samples, 5)
    fig, axes = plt.subplots(2, max_cols, figsize=(3.5 * max_cols, 6.8), squeeze=False)

    for i in range(max_cols):
        axes[0, i].imshow(video_frames[i])
        axes[0, i].set_title(f"Frame {i}", fontsize=11, pad=6)
        axes[0, i].axis("off")

        # crop both width (colorbar) and height (text panel) back to the original frame
        axes[1, i].imshow(overlay_frames[i][:orig_h, :orig_w])
        axes[1, i].set_title("Prompt Attention Overlay", fontsize=10, pad=6)
        axes[1, i].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Text -> Vision Attention Intensity", rotation=270, labelpad=15)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"])

    fig.suptitle(f"Prompt: '{text_prompt}'", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.05, wspace=0.12, hspace=0.20)
    plt.savefig(output_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison grid to: {output_grid_path}")


def print_params_fingerprint(params, sample: int = 4096):
    """Cheap per-module checksum of the loaded weights. Run it for two checkpoints:
    modules whose numbers are identical did NOT change between them."""
    from flax import traverse_util

    flat = traverse_util.flatten_dict(params, sep="/")
    groups: Dict[str, List[float]] = {}
    for k, v in flat.items():
        parts = k.split("/")
        if parts[0] == "PaliGemma" and len(parts) > 1 and parts[1] == "llm":
            # openpi names action-expert weights with a '_1' suffix (e.g. q_einsum_1)
            group = "PaliGemma/llm (action expert, *_1)" if any(p.endswith("_1") for p in parts) else "PaliGemma/llm (VLM backbone)"
        else:
            group = "/".join(parts[:2])
        arr = np.asarray(v).reshape(-1)[:sample].astype(np.float64)
        groups.setdefault(group, []).append(float(np.abs(arr).sum()))
    print("\n[fingerprint] loaded weights (compare these between checkpoints):")
    for g in sorted(groups):
        print(f"    {g:45s} {sum(groups[g]):.10e}   ({len(groups[g])} arrays)")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualization script")
    parser.add_argument("--checkpoint", default="/mnt/nas/rajat_ws/checkpoints/pi05_g1_pickplace/run3/800/", help="Checkpoint path", type=str)
    parser.add_argument("--config_name", default="pi05_g1_pickplace", help="config name", type=str)
    parser.add_argument("--csv", default="data/meta.csv", help="CSV file", type=str)
    parser.add_argument(
        "--temperature", default=None, type=float,
        help="Softmax temperature for the cosine-similarity attention. "
             "Default (None) keeps the original sqrt(d) scaling; try 0.05-0.1 for sharper maps.",
    )

    parser.add_argument("--max_token_len", default=200, type=int, help="Tokenizer max length (padding is stripped)")

    args = parser.parse_args()
    # Load model checkpoint ONCE outside the loop to improve performance
    config_name = args.config_name
    config = _config.get_config(config_name)

    ## save dir
    # FIX: old code `args.checkpoint.split('/')[-2]` returned the PARENT folder (e.g. 'run3')
    # when the path had no trailing slash -> every checkpoint wrote to the same folder and
    # overwrote the previous run's plots.
    step_number = Path(args.checkpoint.rstrip("/")).name
    run_tag = f"{config_name} @ {step_number}"
    save_dir = os.path.join('./plots', config_name, step_number)
    print(f"Files would be saved at : {save_dir}")
    save_prompt = args.csv.split('/')[-1].replace('.csv', '')
    os.makedirs(os.path.join(save_dir, save_prompt), exist_ok=True)

    metadata_df = pd.read_csv(args.csv)
    prompts = metadata_df['prompt'].tolist()
    video_paths = metadata_df['video_path'].tolist()
    tasks = metadata_df['task'].tolist()

    print(f"Loaded {len(prompts)} prompts and video paths from metadata CSV.")

    params_path = os.path.join(args.checkpoint, "params")

    print(f"Loading checkpoint from: {args.checkpoint}")
    params = _model.restore_params(params_path)
    full_model = config.model.load(params)
    print_params_fingerprint(params)

    paligemma_dict = full_model.PaliGemma
    vision_tower = paligemma_dict.img if hasattr(paligemma_dict, "img") else paligemma_dict["img"]
    llm_backbone = paligemma_dict.llm if hasattr(paligemma_dict, "llm") else paligemma_dict["llm"]

    # Real tokenizer: openpi's PaligemmaTokenizer (downloads the SentencePiece model once).
    # Text embedding is done by the LLM itself via method="embed" (same as pi0/pi05 embed_prefix).
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer
        pg_tokenizer = PaligemmaTokenizer(max_len=args.max_token_len)
    except Exception as err:
        print(f"WARNING: could not create PaligemmaTokenizer ({err}).")
        pg_tokenizer = None

    import torch

    print("GPU:", torch.cuda.get_device_name(0))
    print("Allocated:", torch.cuda.memory_allocated(0) / 1024**3, "GB")
    print("Reserved: ", torch.cuda.memory_reserved(0) / 1024**3, "GB")
    print("Max allocated:", torch.cuda.max_memory_allocated(0) / 1024**3, "GB")
    print("Max reserved: ", torch.cuda.max_memory_reserved(0) / 1024**3, "GB")

    for video_path, prompt, task in zip(video_paths, prompts, tasks):
        print(f"\n================ PROCESSING VIDEO ==================")
        print(f"Video: {video_path} | Task: {task} | Prompt: {prompt}")

        frames = load_video_frames(video_path, fps_sample_rate=1, target_size=(224, 224))
        total_frames = len(frames)
        infer_batch_size = 36

        token_pieces = None
        sample_text_embeds = None
        if pg_tokenizer is not None:
            text_ids_np, token_pieces, word_token_map = tokenize_prompt(prompt, pg_tokenizer)
            print(f"Tokens ({len(token_pieces)}): {token_pieces}")
            print(f"Mapped Words -> Token Indices: {word_token_map}")
            try:
                text_ids = jnp.asarray(text_ids_np, dtype=jnp.int32)[None, :]
                sample_text_embeds = llm_backbone(text_ids, method="embed")
                print(f"Text embeddings: {sample_text_embeds.shape} (real tokens)")
            except Exception as err:
                print(f"WARNING: llm_backbone(..., method='embed') failed ({err}).")
                sample_text_embeds = None

        if sample_text_embeds is None:
            token_pieces = None
            word_token_map = map_words_to_token_indices(prompt, None)
            print(
                "WARNING: tokenizer/embedder not found -> using constant dummy text embeddings. "
                "All text tokens are identical, so per-word attention is NOT meaningful in this mode."
            )
            text_seq_len = max([max(idx_list) for idx_list in word_token_map.values()], default=8) + 2
            embed_dim = 2048
            sample_text_embeds = jnp.ones((1, text_seq_len, embed_dim), dtype=jnp.float32)

        video_array = np.stack(frames, axis=0)
        all_prompt_attentions = []
        all_text_from_text = []    # NEW
        all_text_from_vision = []  # NEW
        all_received = []          # NEW: [N, V+T]
        sample_idx = total_frames // 2   # frame used for the single-frame attention matrix
        sample_attn_matrix = None
        attn_sum = None                  # running sum -> mean attention matrix over all frames
        word_attentions_per_frame = []

        for start_idx in tqdm(range(0, total_frames, infer_batch_size)):
            end_idx = min(start_idx + infer_batch_size, total_frames)
            batch_np = video_array[start_idx:end_idx]
            # openpi feeds SigLIP images scaled to [-1, 1] (not [0, 1])
            batch_jax = jnp.array(batch_np, dtype=jnp.float32) / 127.5 - 1.0

            vision_outputs = vision_tower(batch_jax, train=False)
            img_embeds = vision_outputs[0] if isinstance(vision_outputs, (tuple, list)) else vision_outputs
            if len(img_embeds.shape) == 4:
                b_size, h, w, c = img_embeds.shape
                img_embeds = img_embeds.reshape((b_size, h * w, c))

            curr_batch_size = img_embeds.shape[0]
            batch_text_embeds = jnp.tile(sample_text_embeds, (curr_batch_size, 1, 1))

            batch_vis_attn, text_from_text, text_from_vision, raw_cross_attn, full_attn = extract_prompt_conditioned_attention(
                image_embeddings=img_embeds,
                text_embeddings=batch_text_embeds,
                llm_backbone=llm_backbone,
                temperature=args.temperature,
            )

            all_prompt_attentions.append(np.array(batch_vis_attn, dtype=np.float32))
            all_text_from_text.append(np.array(text_from_text, dtype=np.float32))
            all_text_from_vision.append(np.array(text_from_vision, dtype=np.float32))

            # NEW: joint image + text stats
            full_attn_np = np.array(full_attn, dtype=np.float32)
            if start_idx == 0:
                print(
                    f"[fingerprint] {run_tag} | img_embeds mean|x| = {float(jnp.abs(img_embeds).mean()):.8f} "
                    f"| attn std = {float(full_attn_np.std()):.6e}"
                )
            num_img_tokens = img_embeds.shape[1]
            stats = compute_image_text_stats(full_attn_np, num_img_tokens)
            all_received.append(stats["received"])
            attn_sum = full_attn_np.sum(axis=0) if attn_sum is None else attn_sum + full_attn_np.sum(axis=0)
            if start_idx <= sample_idx < end_idx:
                sample_attn_matrix = full_attn_np[sample_idx - start_idx]

            # DYNAMIC WORD EXTRACTION PER FRAME
            for b_idx in range(curr_batch_size):
                frame_kw_attn = {}
                for word, token_indices in word_token_map.items():
                    word_attn_slice = np.array(
                        raw_cross_attn[b_idx, token_indices, :].mean(axis=0),
                        dtype=np.float32,
                    )
                    frame_kw_attn[word] = word_attn_slice.reshape((16, 16))

                word_attentions_per_frame.append(frame_kw_attn)

        full_prompt_attn_np = np.concatenate(all_prompt_attentions, axis=0)

        # NEW: per-word text attention shares
        text_from_text_np = np.concatenate(all_text_from_text, axis=0)      # [N, T]
        text_from_vision_np = np.concatenate(all_text_from_vision, axis=0)  # [N, T]
        shares_from_text = compute_word_attention_shares(text_from_text_np, word_token_map)
        shares_from_vision = compute_word_attention_shares(text_from_vision_np, word_token_map)
        word_text_shares_per_frame = [
            {w: float(shares_from_vision[w][i]) for w in shares_from_vision} for i in range(total_frames)
        ]

        out_dir = os.path.join(save_dir, save_prompt)

        # NEW: image vs text
        received_all = np.concatenate(all_received, axis=0)      # [N, V+T]
        num_text_tokens = received_all.shape[1] - num_img_tokens
        image_share_per_frame = received_all[:, :num_img_tokens].sum(axis=1)

        generate_and_save_vlm_attention(
            video_frames=video_array,
            vlm_patch_attention=full_prompt_attn_np,
            word_attentions_per_frame=word_attentions_per_frame,
            output_video_path=os.path.join(out_dir, f"vlm_prompt_attention_{task}.mp4"),
            output_grid_path=os.path.join(out_dir, f"vlm_prompt_attention_grid_{task}.png"),
            fps=4,
            text_prompt=f"{prompt}  [{run_tag}]",
            word_text_shares_per_frame=word_text_shares_per_frame,
            image_share_per_frame=image_share_per_frame,
            num_img_tokens=num_img_tokens,
            num_text_tokens=num_text_tokens,
        )

        save_text_query_attention(
            attn_matrix=sample_attn_matrix,
            num_img_tokens=num_img_tokens,
            word_token_map=word_token_map,
            token_pieces=token_pieces,
            output_path=os.path.join(out_dir, f"text_query_attention_frame{sample_idx}_{task}.png"),
            title=f"{run_tag} | Frame {sample_idx} | Prompt: '{prompt}'",
        )
        save_text_query_attention(
            attn_matrix=attn_sum / total_frames,
            num_img_tokens=num_img_tokens,
            word_token_map=word_token_map,
            token_pieces=token_pieces,
            output_path=os.path.join(out_dir, f"text_query_attention_mean_{task}.png"),
            title=f"{run_tag} | Mean over {total_frames} frames | Prompt: '{prompt}'",
        )

        save_text_attention_plots(
            shares_from_vision,
            shares_from_text,
            output_path=os.path.join(out_dir, f"text_attention_{task}.png"),
            text_prompt=f"{prompt}  [{run_tag}]",
        )
        save_text_attention_csv(
            shares_from_vision,
            shares_from_text,
            output_path=os.path.join(out_dir, f"text_attention_{task}.csv"),
        )


if __name__ == "__main__":
    main()
