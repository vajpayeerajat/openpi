"""Find idle episodes in one or more G1 LeRobot v2.1 datasets: recordings where the robot never grasps (empty table,
a person handling the object while the robot waits) but which still carry a "Pick up ..." task.

Rule (same as `drop_idle_episodes` in the openpi data loader, split_lerobot_datasets.py and rajat_gradcam_analysis.py):
an episode is idle when the mean right-hand action command never moves more than --threshold (default 0.05) over the
whole episode. Right-hand dims are found from the action width: 28-dim upper body -> 21:28, 43-dim whole body
-> 36:43 (the upper body is 15:43). Override with --hand_dims.

Read-only by default. Prints, per dataset: idle episodes / frames, per task, how many episodes sit just above the
threshold (to judge how sensitive the count is), and a ready-made `--exclude` argument for
lerobot-merge-kit/merge_datasets.py (source indices follow the order of the paths given here, which must match the
order of the --src flags).

Usage:
  python scripts/find_idle_episodes.py /mnt/nas/Datasets/A /mnt/nas/Datasets/B
  python scripts/find_idle_episodes.py /data/A /data/B --csv idle.csv --preview idle_preview.png
  python scripts/find_idle_episodes.py /data/A --write_cache        # writes <A>/meta/idle_episodes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UPPER_BODY_START = 15  # 43-dim G1 whole body: legs 0..11, waist 12..14, upper body 15..42
RIGHT_HAND_28 = (21, 28)
RIGHT_ARM_28 = (14, 21)
BORDERLINE = 3.0  # report episodes with threshold <= hand range < BORDERLINE * threshold


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def episode_paths(root, info, ep):
    chunk = ep // info["chunks_size"]
    data = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    video = None
    if info.get("video_path") and "observation.images.ego_view" in info["features"]:
        video = root / info["video_path"].format(episode_chunk=chunk, video_key="observation.images.ego_view",
                                                 episode_index=ep)
    return data, video


def dims_for(width, hand_dims):
    if hand_dims:
        a, b = (int(x) for x in hand_dims.split(":"))
        return (a, b), None
    offset = {28: 0, 43: UPPER_BODY_START}.get(width)
    if offset is None:
        raise ValueError(f"action width {width}: pass --hand_dims START:END for the right-hand dims")
    return ((RIGHT_HAND_28[0] + offset, RIGHT_HAND_28[1] + offset),
            (RIGHT_ARM_28[0] + offset, RIGHT_ARM_28[1] + offset))


def scan(root, threshold, hand_dims):
    root = Path(root)
    info = json.load(open(root / "meta" / "info.json"))
    tasks = {t["task_index"]: t["task"] for t in read_jsonl(root / "meta" / "tasks.jsonl")}
    width = info["features"]["action"]["shape"][0]
    hand, arm = dims_for(width, hand_dims)
    rows = []
    for e in read_jsonl(root / "meta" / "episodes.jsonl"):
        ep = e["episode_index"]
        data, video = episode_paths(root, info, ep)
        df = pd.read_parquet(data, columns=["action", "task_index"])
        a = np.stack(df["action"]).astype(np.float64)
        hand_range = float(np.ptp(a[:, hand[0]:hand[1]].mean(1)))
        arm_range = float(np.ptp(a[:, arm[0]:arm[1]], axis=0).max()) if arm else float("nan")
        rows.append({"dataset": root.name, "episode_index": ep, "task": tasks.get(int(df["task_index"].iloc[0]), "?"),
                     "length": len(df), "right_hand_range": hand_range, "right_arm_max_range": arm_range,
                     "idle": hand_range < threshold, "video": str(video) if video else ""})
    print(f"[scan] {root}: {len(rows)} episodes, action width {width}, right-hand dims {hand[0]}:{hand[1]}")
    return pd.DataFrame(rows)


def report(df, threshold):
    lines = []
    for name, d in df.groupby("dataset", sort=False):
        idle = d[d["idle"]]
        border = d[(d["right_hand_range"] >= threshold) & (d["right_hand_range"] < BORDERLINE * threshold)]
        lines += [f"\n== {name}",
                  f"   idle episodes: {len(idle)}/{len(d)} ({100 * len(idle) / max(len(d), 1):.1f}% of episodes, "
                  f"{100 * idle['length'].sum() / max(d['length'].sum(), 1):.1f}% of frames)",
                  f"   borderline (hand range {threshold:g}-{BORDERLINE * threshold:g}): {len(border)} episodes "
                  f"{border['episode_index'].tolist()[:20]}{' ...' if len(border) > 20 else ''}"]
        if len(idle):
            lines.append(f"   idle right-arm max range: median {idle['right_arm_max_range'].median():.3f} rad "
                         f"(active episodes: {d.loc[~d['idle'], 'right_arm_max_range'].median():.3f})")
        per_task = d.groupby("task")["idle"].agg(["sum", "count"])
        for task, r in per_task.sort_values("sum", ascending=False).iterrows():
            lines.append(f"   {int(r['sum']):4d}/{int(r['count']):<4d} idle | {task[:90]}")
    total = df["idle"].sum()
    lines.append(f"\nTotal: {total}/{len(df)} idle episodes, {100 * df.loc[df['idle'], 'length'].sum() / df['length'].sum():.1f}% of frames")
    print("\n".join(lines))


def exclude_args(df):
    parts = []
    for i, (_, d) in enumerate(df.groupby("dataset", sort=False)):
        ids = d.loc[d["idle"], "episode_index"].tolist()
        if ids:
            parts.append(f"--exclude {i}:{','.join(map(str, ids))}")
    return " ".join(parts)


def preview(df, path, n, seed):
    """Contact sheet: ego_view at 10 / 50 / 90 % of n random idle episodes, to confirm they really are idle."""
    import imageio.v3 as iio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idle = df[df["idle"] & (df["video"] != "")]
    if idle.empty:
        print("No idle episodes with an ego_view video; skipping preview.")
        return
    pick = idle.sample(min(n, len(idle)), random_state=seed)
    fig, axes = plt.subplots(len(pick), 3, figsize=(9, 2.3 * len(pick)), squeeze=False)
    for i, (_, r) in enumerate(pick.iterrows()):
        want = [int(r["length"] * q) for q in (0.1, 0.5, 0.9)]
        frames = {}
        for k, fr in enumerate(iio.imiter(r["video"], plugin="pyav")):
            if k in want:
                frames[k] = fr
            if k >= want[-1]:
                break
        for j, k in enumerate(want):
            ax = axes[i, j]
            ax.axis("off")
            if k in frames:
                ax.imshow(frames[k])
            if j == 0:
                ax.set_title(f"{r['dataset']} ep {r['episode_index']} · hand range {r['right_hand_range']:.3f}",
                             fontsize=8, loc="left")
    fig.suptitle("Idle episodes (ego_view at 10 / 50 / 90 %)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    print(f"Preview: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("datasets", nargs="+", help="LeRobot v2.1 dataset roots (order = merge_datasets.py --src order)")
    p.add_argument("--threshold", type=float, default=0.05, help="right-hand action range below which an episode is idle")
    p.add_argument("--hand_dims", default=None, help="START:END of the right-hand action dims (default: from the width)")
    p.add_argument("--csv", default=None, help="write every episode's metrics here")
    p.add_argument("--preview", default=None, help="write a PNG contact sheet of random idle episodes here")
    p.add_argument("--preview_n", type=int, default=8)
    p.add_argument("--write_cache", action="store_true",
                   help="write <dataset>/meta/idle_episodes.json (the openpi loader's drop_idle_episodes cache)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = [Path(d).name for d in args.datasets]
    if len(set(names)) != len(names):
        raise ValueError("Dataset folders must have distinct names.")
    df = pd.concat([scan(d, args.threshold, args.hand_dims) for d in args.datasets], ignore_index=True)
    report(df, args.threshold)
    ex = exclude_args(df)
    print("\nFor lerobot-merge-kit/merge_datasets.py (same source order as given here):")
    print(f"  {ex}" if ex else "  (no idle episodes)")

    if args.csv:
        df.drop(columns=["video"]).to_csv(args.csv, index=False)
        print(f"CSV: {args.csv}")
    if args.preview:
        preview(df, args.preview, args.preview_n, args.seed)
    if args.write_cache:
        for d in args.datasets:
            sub = df[df["dataset"] == Path(d).name]
            out = Path(d) / "meta" / "idle_episodes.json"
            out.write_text(json.dumps({"rule": f"ptp(mean right-hand action) < {args.threshold}",
                                       "idle_episodes": sub.loc[sub["idle"], "episode_index"].tolist()}))
            print(f"Cache: {out}")


if __name__ == "__main__":
    main()
