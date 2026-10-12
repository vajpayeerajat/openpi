"""Merge one or more G1 LeRobot v2.1 datasets, drop idle episodes, and split into train / val / test with matched
distributions.

Inputs must share the schema of the openpi `train/` / `val/` folders (28-dim G1 upper-body state/action, cameras
ego_view / ego_left / ego_right, same fps and video settings). The script refuses to merge folders whose features differ.

Steps:
  1. Idle episodes are dropped: the right hand (action dims 21:28) never moves more than --idle_range. These are
     recordings where the robot does nothing (empty table, a person handling the object) under a "Pick up ..." prompt.
     Same rule as `drop_idle_episodes` in the training data loader and rajat_gradcam_analysis.py. --keep_idle keeps them.
  2. Stratified split. A stratum is (source folder, task). Inside a stratum, episodes are ordered by length (random
     tie-break) and dealt one by one to the split furthest below its target share, so every split gets the same
     source mix, task mix and episode-length distribution.
  3. Each split is written as a self-contained LeRobot v2.1 dataset: episode indices and global frame indices are
     renumbered, tasks are merged into one table, per-episode stats are carried over, videos are copied (or
     symlinked with --videos symlink), and meta/provenance.jsonl maps every new episode back to its source.
  4. split_report.md / split_report.json compare the splits: episodes, frames, task and source shares, episode-length
     quantiles, and the largest standardized difference of per-dimension action/state means vs the whole dataset.

Usage:
  python scripts/split_lerobot_datasets.py /mnt/nas/Datasets/<ds1> /mnt/nas/Datasets/<ds2> \
      --out /mnt/nas/Datasets/g1_merged_split --ratios 0.8 0.1 0.1 --dry_run      # check the plan first
  python scripts/split_lerobot_datasets.py /mnt/nas/Datasets/<ds1> /mnt/nas/Datasets/<ds2> \
      --out /mnt/nas/Datasets/g1_merged_split --ratios 0.8 0.1 0.1
The output has <out>/train, <out>/val, <out>/test, usable as `local_root` in the openpi configs.
"""

import argparse
import collections
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
RIGHT_HAND = slice(21, 28)
VECTOR_KEYS = ("observation.state", "action")
# Keys whose value legitimately differs between sources and is not part of the data schema.
INFO_PER_DATASET = {"total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks", "splits",
                    "derived_from", "discarded_episode_indices", "merged_from"}


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ============================================================================
# Loading sources
# ============================================================================


class Source:
    def __init__(self, root):
        self.root = Path(root)
        meta = self.root / "meta"
        if not (meta / "info.json").exists():
            raise FileNotFoundError(f"{root}: no meta/info.json (not a LeRobot dataset)")
        self.info = json.load(open(meta / "info.json"))
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(f"{root}: codebase_version {self.info.get('codebase_version')}, expected v2.1")
        self.tasks = {t["task_index"]: t["task"] for t in read_jsonl(meta / "tasks.jsonl")}
        self.episodes = {e["episode_index"]: e for e in read_jsonl(meta / "episodes.jsonl")}
        stats_path = meta / "episodes_stats.jsonl"
        self.stats = {s["episode_index"]: s["stats"] for s in read_jsonl(stats_path)} if stats_path.exists() else {}
        self.video_keys = [k for k, v in self.info["features"].items() if v["dtype"] == "video"]
        self.name = self.root.name

    def data_path(self, ep):
        return self.root / self.info["data_path"].format(episode_chunk=ep // self.info["chunks_size"], episode_index=ep)

    def video_path(self, ep, key):
        return self.root / self.info["video_path"].format(episode_chunk=ep // self.info["chunks_size"], video_key=key,
                                                          episode_index=ep)


def check_schema(sources):
    ref = sources[0]
    for s in sources[1:]:
        problems = []
        if s.info["fps"] != ref.info["fps"]:
            problems.append(f"fps {s.info['fps']} vs {ref.info['fps']}")
        if set(s.info["features"]) != set(ref.info["features"]):
            problems.append(f"feature keys differ: {sorted(set(s.info['features']) ^ set(ref.info['features']))}")
        else:
            for k, v in ref.info["features"].items():
                w = s.info["features"][k]
                if (v["dtype"], v["shape"]) != (w["dtype"], w["shape"]):
                    problems.append(f"{k}: {w['dtype']}{w['shape']} vs {v['dtype']}{v['shape']}")
                elif v["dtype"] == "video" and v.get("info") != w.get("info"):
                    problems.append(f"{k}: video settings differ")
        if problems:
            raise ValueError(f"{s.root} does not match the schema of {ref.root}:\n  " + "\n  ".join(problems))
    for k in VECTOR_KEYS:
        if ref.info["features"][k]["shape"] != [28]:
            raise ValueError(f"{k} must be 28-dim (G1 upper body), got {ref.info['features'][k]['shape']}")


def scan_episodes(sources, idle_range):
    """One row per episode: source, index, task, length, idle flag, action/state sums for the distribution report."""
    rows = []
    for si, s in enumerate(sources):
        for ep, meta in sorted(s.episodes.items()):
            df = pd.read_parquet(s.data_path(ep), columns=["action", "observation.state", "task_index"])
            action = np.stack(df["action"]).astype(np.float64)
            state = np.stack(df["observation.state"]).astype(np.float64)
            tasks = df["task_index"].unique()
            if len(tasks) != 1:
                raise ValueError(f"{s.data_path(ep)}: {len(tasks)} tasks in one episode; not supported")
            rows.append({
                "source": si, "source_name": s.name, "src_episode": int(ep), "task": s.tasks[int(tasks[0])],
                "length": len(df), "idle": bool(np.ptp(action[:, RIGHT_HAND].mean(1)) < idle_range),
                "a_sum": action.sum(0), "a_sq": (action ** 2).sum(0), "s_sum": state.sum(0), "s_sq": (state ** 2).sum(0),
            })
        print(f"[scan] {s.root}: {len(s.episodes)} episodes")
    return pd.DataFrame(rows)


# ============================================================================
# Stratified split
# ============================================================================


RARE = "(rare tasks)"


def strata_keys(eps, ratios):
    """(source, task) per episode, except tasks with too few episodes to reach every split are pooled per source into
    one "(rare tasks)" stratum. Without pooling, one-episode tasks (e.g. hundreds of unique paraphrased prompts) would
    all land in train and leave val/test without them."""
    ratios = np.asarray(ratios, float) / np.sum(ratios)
    min_n = max(int(round(1 / ratios[ratios > 0].min())), 1)
    counts = eps.groupby(["source", "task"])["task"].transform("size")
    return eps["task"].where(counts >= min_n, RARE)


def stratified_split(eps, ratios, seed):
    """Per stratum ((source, task), rare tasks pooled per source): order by length, shuffle within blocks of similar
    length, then deal each episode to the split furthest below its target. Every split gets the same share of each
    stratum and of each length range; the seed decides which episodes of a block go where."""
    rng = np.random.default_rng(seed)
    ratios = np.asarray(ratios, float) / np.sum(ratios)
    block = max(int(round(1 / ratios[ratios > 0].min())), 1)
    assign = pd.Series(index=eps.index, dtype=object)
    small = []
    for key, grp in eps.assign(_stratum=strata_keys(eps, ratios)).groupby(["source", "_stratum"]):
        by_len = grp.sort_values("length").index.to_numpy()
        order = np.concatenate([rng.permutation(by_len[i:i + block]) for i in range(0, len(by_len), block)])
        counts = np.zeros(len(SPLITS))
        offset = rng.random(len(SPLITS)) * 1e-6  # random tie-break between splits with equal deficit
        for n, idx in enumerate(order, start=1):
            deficit = ratios * n - counts + offset
            k = int(np.argmax(deficit))
            assign[idx] = SPLITS[k]
            counts[k] += 1
        if len(grp) < round(1 / ratios[ratios > 0].min()):
            small.append((eps.loc[grp.index[0], "source_name"], key[1], len(grp)))
    return assign, small


# ============================================================================
# Report
# ============================================================================


def split_score(eps, assign):
    """Worst standardized difference of per-dimension action/state means, any held-out split vs the whole dataset."""
    tot = eps["length"].sum()
    worst = 0.0
    for col in ("a", "s"):
        sums, sq = np.stack(eps[f"{col}_sum"]), np.stack(eps[f"{col}_sq"])
        mean = sums.sum(0) / tot
        std = np.sqrt(np.maximum(sq.sum(0) / tot - mean ** 2, 1e-12))
        valid = std > 1e-6
        for sp in SPLITS[1:]:
            m = (assign == sp).to_numpy()
            if m.any():
                sm = sums[m].sum(0) / eps["length"].to_numpy()[m].sum()
                worst = max(worst, float(np.max(np.abs(sm - mean)[valid] / std[valid])))
    return worst


def distribution_report(eps, ratios):
    eps = eps.assign(_stratum=strata_keys(eps, ratios))
    rep = {"ratios": dict(zip(SPLITS, ratios)), "splits": {},
           "n_tasks": int(eps["task"].nunique()), "n_rare_tasks": int(eps.loc[eps["_stratum"] == RARE, "task"].nunique())}
    tot_frames = eps["length"].sum()
    a_mean = np.stack(eps["a_sum"]).sum(0) / tot_frames
    a_std = np.sqrt(np.maximum(np.stack(eps["a_sq"]).sum(0) / tot_frames - a_mean ** 2, 1e-12))
    s_mean = np.stack(eps["s_sum"]).sum(0) / tot_frames
    s_std = np.sqrt(np.maximum(np.stack(eps["s_sq"]).sum(0) / tot_frames - s_mean ** 2, 1e-12))
    for sp in SPLITS:
        d = eps[eps["split"] == sp]
        if len(d) == 0:
            rep["splits"][sp] = {"episodes": 0}
            continue
        f = d["length"].sum()
        am = np.stack(d["a_sum"]).sum(0) / f
        sm = np.stack(d["s_sum"]).sum(0) / f
        valid_a = a_std > 1e-6  # constant dims (e.g. the idle left hand) carry no information
        valid_s = s_std > 1e-6
        rep["splits"][sp] = {
            "episodes": int(len(d)), "frames": int(f), "episode_share": len(d) / len(eps), "frame_share": f / tot_frames,
            "task_share": d["_stratum"].value_counts(normalize=True).round(4).to_dict(),
            "n_distinct_tasks": int(d["task"].nunique()),
            "source_share": d["source_name"].value_counts(normalize=True).round(4).to_dict(),
            "length_q": {q: float(np.quantile(d["length"], q)) for q in (0.1, 0.5, 0.9)},
            "action_max_smd": float(np.max(np.abs(am - a_mean)[valid_a] / a_std[valid_a])),
            "state_max_smd": float(np.max(np.abs(sm - s_mean)[valid_s] / s_std[valid_s])),
        }
    return rep


def write_report(rep, idle, small, out, dry_run):
    tasks = sorted({t for s in rep["splits"].values() for t in s.get("task_share", {})} - {RARE})
    label = {t: f"T{i}" for i, t in enumerate(tasks)}
    label[RARE] = f"rare tasks pooled ({rep['n_rare_tasks']} tasks with too few episodes each)"
    if rep["n_rare_tasks"]:
        tasks.append(RARE)
    sources = sorted({t for s in rep["splits"].values() for t in s.get("source_share", {})})
    L = ["# Split report", "", f"Target ratios: " + ", ".join(f"{k} {v:.2f}" for k, v in rep["ratios"].items()), "",
         "## Idle episodes dropped", ""]
    for name, d in idle.items():
        L.append(f"- {name}: {d['n_idle']}/{d['n']} episodes ({100 * d['frame_frac']:.1f}% of frames)")
    L += ["", "## Splits", "", "| | " + " | ".join(SPLITS) + " |", "|---|" + "---|" * len(SPLITS)]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(rep["splits"][sp]) if rep["splits"][sp].get("episodes") else "—"
                                             for sp in SPLITS) + " |")

    row("episodes", lambda s: f"{s['episodes']} ({100 * s['episode_share']:.1f}%)")
    row("frames", lambda s: f"{s['frames']} ({100 * s['frame_share']:.1f}%)")
    row("episode length p10 / p50 / p90", lambda s: " / ".join(f"{v:.0f}" for v in s["length_q"].values()))
    for t in tasks:
        row(f"task {label[t]}", lambda s, t=t: f"{100 * s['task_share'].get(t, 0):.1f}%")
    row("distinct task strings", lambda s: str(s["n_distinct_tasks"]))
    for src in sources:
        row(f"source: {src}", lambda s, src=src: f"{100 * s['source_share'].get(src, 0):.1f}%")
    row("max |mean diff| / std, action", lambda s: f"{s['action_max_smd']:.3f}")
    row("max |mean diff| / std, state", lambda s: f"{s['state_max_smd']:.3f}")
    if "search" in rep:
        sr = rep["search"]
        L.append(f"| search | chosen seed {sr['seed']} of {sr['n']} candidates: worst held-out difference {sr['best']:.3f} "
                 f"(median candidate {sr['median']:.3f}) | | |")
    L += ["", "Tasks: " + "; ".join(f"**{label[t]}** = {t}" for t in tasks if t != RARE)]
    L += ["", "The last two rows compare each split's per-dimension mean with the whole dataset, in units of the whole "
          "dataset's std (worst dimension). Below ~0.1 the splits are statistically very similar; above ~0.25 look at "
          "which source/task is unbalanced."]
    if small:
        L += ["", "## Small strata", "", "These strata are too small to appear in every split:", ""]
        L += [f"- {s} / {t}: {n} episodes" for s, t, n in small]
    if dry_run:
        L += ["", "_Dry run: nothing was written._"]
    text = "\n".join(L) + "\n"
    print("\n" + text)
    if not dry_run:
        (out / "split_report.md").write_text(text)
        json.dump(rep, open(out / "split_report.json", "w"), indent=2)


# ============================================================================
# Writing
# ============================================================================


def _place_video(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def write_split(name, eps, sources, out, videos_mode, provenance_extra):
    root = out / name
    if root.exists():
        raise FileExistsError(f"{root} exists; remove it or pass a different --out")
    (root / "meta").mkdir(parents=True)
    ref = sources[0]
    tasks = sorted(eps["task"].unique())
    task_index = {t: i for i, t in enumerate(tasks)}
    chunks_size = ref.info["chunks_size"]
    episodes_rows, stats_rows, prov_rows = [], [], []
    global_index = 0
    eps = eps.sort_values(["source", "src_episode"])
    for new_ep, (_, e) in enumerate(eps.iterrows()):
        s = sources[e["source"]]
        df = pd.read_parquet(s.data_path(e["src_episode"]))
        n = len(df)
        df["episode_index"] = new_ep
        df["index"] = np.arange(global_index, global_index + n, dtype=np.int64)
        df["task_index"] = task_index[e["task"]]
        dst = root / ref.info["data_path"].format(episode_chunk=new_ep // chunks_size, episode_index=new_ep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        for key in ref.video_keys:
            _place_video(s.video_path(e["src_episode"], key),
                         root / ref.info["video_path"].format(episode_chunk=new_ep // chunks_size, video_key=key,
                                                              episode_index=new_ep), videos_mode)
        episodes_rows.append({"episode_index": new_ep, "tasks": [e["task"]], "length": n})
        st = json.loads(json.dumps(s.stats.get(e["src_episode"], {})))
        if st:
            idx = np.arange(global_index, global_index + n, dtype=np.float64)
            if "episode_index" in st:
                st["episode_index"].update(min=[new_ep], max=[new_ep], mean=[float(new_ep)], std=[0.0])
            if "index" in st:
                st["index"].update(min=[int(idx[0])], max=[int(idx[-1])], mean=[float(idx.mean())], std=[float(idx.std())])
            if "task_index" in st:
                ti = task_index[e["task"]]
                st["task_index"].update(min=[ti], max=[ti], mean=[float(ti)], std=[0.0])
            stats_rows.append({"episode_index": new_ep, "stats": st})
        prov_rows.append({"episode_index": new_ep, "source": str(s.root.resolve()), "source_episode_index": int(e["src_episode"])})
        global_index += n

    info = {k: v for k, v in ref.info.items() if k not in INFO_PER_DATASET}
    info.update(
        total_episodes=len(eps), total_frames=int(global_index), total_tasks=len(tasks),
        total_videos=len(eps) * len(ref.video_keys), total_chunks=(len(eps) - 1) // chunks_size + 1,
        splits={name: f"0:{len(eps)}"}, merged_from=[str(s.root.resolve()) for s in sources], **provenance_extra,
    )
    json.dump(info, open(root / "meta" / "info.json", "w"), indent=4)
    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": i, "task": t} for t, i in task_index.items()])
    write_jsonl(root / "meta" / "episodes.jsonl", episodes_rows)
    if stats_rows:
        write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)
    write_jsonl(root / "meta" / "provenance.jsonl", prov_rows)
    if (ref.root / "meta" / "modality.json").exists():
        shutil.copy2(ref.root / "meta" / "modality.json", root / "meta" / "modality.json")
    print(f"[write] {root}: {len(eps)} episodes, {global_index} frames, {len(tasks)} tasks")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sources", nargs="+", help="LeRobot v2.1 dataset roots (same schema as openpi train/ val/)")
    p.add_argument("--out", required=True, help="output dir; <out>/train, <out>/val, <out>/test are created")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search", type=int, default=300,
                   help="candidate splits (seeds) to try; the best-matched one is kept. 1 = use --seed as is")
    p.add_argument("--idle_range", type=float, default=0.05, help="right-hand action range below which an episode is idle")
    p.add_argument("--keep_idle", action="store_true", help="do not drop idle episodes")
    p.add_argument("--videos", choices=["copy", "symlink", "hardlink"], default="copy",
                   help="copy is portable (e.g. rsync to a training box); symlink saves space on the same machine")
    p.add_argument("--dry_run", action="store_true", help="print the split plan and report, write nothing")
    args = p.parse_args()

    sources = [Source(s) for s in args.sources]
    if len({s.name for s in sources}) != len(sources):
        raise ValueError("Source folders must have distinct names (they label the source in the report).")
    check_schema(sources)
    eps = scan_episodes(sources, args.idle_range)

    idle = {}
    for s in sources:
        d = eps[eps["source_name"] == s.name]
        idle[s.name] = {"n": len(d), "n_idle": int(d["idle"].sum()),
                        "frame_frac": float(d.loc[d["idle"], "length"].sum() / max(d["length"].sum(), 1)),
                        "episodes": d.loc[d["idle"], "src_episode"].tolist()}
    if not args.keep_idle:
        eps = eps[~eps["idle"]].reset_index(drop=True)

    # Try --search seeds and keep the split whose held-out sets best match the whole dataset's action/state means.
    scores = []
    for seed in range(args.seed, args.seed + max(args.search, 1)):
        assign, small = stratified_split(eps, args.ratios, seed)
        scores.append((split_score(eps, assign), seed))
    best, best_seed = min(scores)
    eps["split"], small = stratified_split(eps, args.ratios, best_seed)
    search_info = {"seed": best_seed, "n": len(scores), "best": best, "median": float(np.median([s for s, _ in scores]))}
    args.seed = best_seed
    rep = distribution_report(eps, args.ratios)
    out = Path(args.out)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    rep["search"] = search_info
    write_report(rep, idle, small, out, args.dry_run)
    if args.dry_run:
        return

    extra = {"discarded_idle_episodes": {k: v["episodes"] for k, v in idle.items()} if not args.keep_idle else {},
             "split_seed": args.seed, "split_ratios": list(args.ratios)}
    for sp in SPLITS:
        d = eps[eps["split"] == sp]
        if len(d):
            write_split(sp, d, sources, out, args.videos, extra)
    json.dump({sp: [{"source": str(sources[r.source].root.resolve()), "episode": int(r.src_episode)}
                    for r in eps[eps["split"] == sp].itertuples()] for sp in SPLITS},
              open(out / "split_assignment.json", "w"), indent=1)
    print(f"\nDone. Use local_root={out}/train (and {out}/val, {out}/test) in the openpi data configs.")


if __name__ == "__main__":
    main()
