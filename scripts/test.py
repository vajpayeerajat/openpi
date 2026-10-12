"""Evaluate a checkpoint on a held-out LeRobot test split (e.g. <split>/test from scripts/split_lerobot_datasets.py).

Metrics are the ones train_pytorch.py logs for val (same code, same groups and tolerances), so they line up with the
TensorBoard val curves:
  loss            flow-matching loss (random noise / time, as in val)
  all/l1, all/mse, all/first_step_l1, all/acc@0.05, all/acc@0.1   on fully denoised action chunks, normalised space
plus a breakdown the val curves do not have:
  g1/<left_arm|left_hand|right_arm|right_hand|all28>/l1   per joint group, padding dims excluded
  by task         per task string with >= --min_task_samples samples, the rest pooled as "(rare tasks)"

Works for PyTorch checkpoints (model.safetensors: Cosmos, pi0/pi05 PyTorch) and JAX checkpoints (params/: pi05).
Norm stats come from the checkpoint's assets/, so the test data is normalised exactly as in training.

Usage:
  # Cosmos (torch venv)
  JAX_PLATFORMS=cpu ~/venvs/openpi_cosmos/bin/python scripts/test.py \
      --config_name cosmos2_8b_lora_g1_pickplace \
      --checkpoint /mnt/nas/rajat_ws/checkpoints/cosmos_as_vlm/cosmos2_8b_lora_g1_pickplace/stage2_lora_bs16/12000 \
      --test_dir /mnt/nas/rajat_ws/splitted_lerobot_and_school/test/
  # pi05 (JAX venv)
  .venv/bin/python scripts/test.py --config_name <pi05 config> --checkpoint <dir with params/> --test_dir <test>
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

G1_GROUPS = {"left_arm": (0, 7), "left_hand": (7, 14), "right_arm": (14, 21), "right_hand": (21, 28), "all28": (0, 28)}
RARE = "(rare tasks)"


# ============================================================================
# Data
# ============================================================================


def frame_table(test_dir, drop_idle):
    """(episode, task) per frame, in the order LeRobotDataset indexes frames (episodes in order, frames in order)."""
    from openpi.training import data_loader as _data

    root = Path(test_dir)
    info = json.load(open(root / "meta" / "info.json"))
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(root / "meta" / "tasks.jsonl") if l.strip()}
    episodes = sorted(json.loads(l)["episode_index"] for l in open(root / "meta" / "episodes.jsonl") if l.strip())
    rows = []
    for ep in episodes:
        path = root / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        df = pd.read_parquet(path, columns=["episode_index", "task_index", "frame_index"])
        rows.append(df)
    table = pd.concat(rows, ignore_index=True)
    table["task"] = table["task_index"].map(tasks)
    if drop_idle:
        from lerobot.common.datasets import lerobot_dataset

        idle = set(_data.find_idle_episodes(lerobot_dataset.LeRobotDatasetMetadata("test", root=root)))
        table = table[~table["episode_index"].isin(idle)].reset_index(drop=True)
    return table


def build_dataset(config, checkpoint, test_dir, drop_idle):
    from openpi.shared import normalize as _normalize
    from openpi.training import data_loader as _data

    assets = [p.parent for p in Path(checkpoint, "assets").glob("*/norm_stats.json")]
    assets = ['/mnt/nas/rajat_ws/checkpoints/95000/95000/norm_stats.json']
    temp = Path(checkpoint, "assets").glob("*/norm_stats.json")
    print(f"Path : {temp}")
    if len(assets) != 1:
        raise FileNotFoundError(f"expected one assets/<id>/norm_stats.json in {checkpoint}, found {assets}")
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(data_config, repo_id="test", local_root=str(test_dir),
                                      norm_stats=_normalize.load(assets[0]), drop_idle_episodes=drop_idle)
    raw = _data.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    return _data.transform_dataset(raw, data_config)


def batches(dataset, indices, batch_size, num_workers):
    import torch

    from openpi.training import data_loader as _data

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers, collate_fn=_data._collate_fn)  # noqa: SLF001
    yield from loader


# ============================================================================
# Backends: per-sample loss [B] and sampled normalised actions [B, H, D]
# ============================================================================


class TorchBackend:
    def __init__(self, config, checkpoint):
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = config.model.load_pytorch(config, os.path.join(checkpoint, "model.safetensors"))
        if hasattr(model, "paligemma_with_expert"):
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model = model.to(self.device).eval()

    def __call__(self, batch, num_steps):
        from openpi.models import model as _model

        t = self.torch
        b = {k: v for k, v in batch.items() if k != "actions"}
        obs = _model.Observation.from_dict(
            __import__("jax").tree.map(lambda x: t.from_numpy(np.asarray(x)).to(self.device), b))
        actions = t.from_numpy(np.asarray(batch["actions"], np.float32)).to(self.device)
        with t.no_grad():
            loss = self.model(obs, actions).float().mean(dim=(1, 2))
            pred = self.model.sample_actions(self.device, obs, num_steps=num_steps).float()
        return loss.cpu().numpy(), pred.cpu().numpy()


class JaxBackend:
    def __init__(self, config, checkpoint):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        self.jax, self.jnp, self._model = jax, jnp, _model
        model = config.model.load(_model.restore_params(os.path.join(checkpoint, "params"), dtype=jnp.bfloat16))
        graphdef, self.state = nnx.split(model)
        self._loss = jax.jit(lambda st, rng, obs, a: nnx.merge(graphdef, st).compute_loss(rng, obs, a, train=False))
        self._sample = jax.jit(lambda st, rng, obs, n: nnx.merge(graphdef, st).sample_actions(rng, obs, num_steps=n),
                               static_argnums=3)
        self.rng = jax.random.key(0)

    def __call__(self, batch, num_steps):
        jax, jnp = self.jax, self.jnp
        b = {k: v for k, v in batch.items() if k != "actions"}
        obs = self._model.Observation.from_dict(jax.tree.map(jnp.asarray, b))
        actions = jnp.asarray(batch["actions"], jnp.float32)
        self.rng, r1, r2 = jax.random.split(self.rng, 3)
        loss = np.asarray(self._loss(self.state, r1, obs, actions)).mean(1)
        pred = np.asarray(self._sample(self.state, r2, obs, num_steps), np.float32)
        return loss, pred


# ============================================================================
# Metrics (per sample, so they can be grouped; overall values use train_pytorch's definitions)
# ============================================================================


def per_sample_metrics(pred, target, groups, tolerances):
    out = {}
    for name, (lo, hi) in groups.items():
        err = np.abs(pred[..., lo:hi] - target[..., lo:hi])
        out[f"{name}/l1"] = err.mean((1, 2))
        out[f"{name}/mse"] = (err ** 2).mean((1, 2))
        if name == "all":
            out["all/first_step_l1"] = err[:, 0].mean(1)
            for tol in tolerances:
                out[f"all/acc@{tol}"] = (err < tol).mean((1, 2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config_name", required=True)
    p.add_argument("--checkpoint", required=True, help="step dir: model.safetensors (PyTorch) or params/ (JAX)")
    p.add_argument("--test_dir", required=True, help="LeRobot v2.1 folder with the same schema as training")
    p.add_argument("--frame_stride", type=int, default=10, help="evaluate every n-th frame (1 = every frame)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=10, help="denoising steps, as in training val")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--drop_idle", action="store_true", help="skip idle episodes (not needed for an idle-free split)")
    p.add_argument("--min_task_samples", type=int, default=30, help="tasks with fewer samples are pooled")
    p.add_argument("--out", default=None, help="output dir (default plots/test/<config>__<run>__<step>)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import train_pytorch  # metric groups / tolerances exactly as logged during training

    from openpi.training import config as _config

    config = _config.get_config(args.config_name)
    ckpt = Path(args.checkpoint.rstrip("/"))
    out = Path(args.out or f"plots/test/{args.config_name}__{ckpt.parent.name}__{ckpt.name}")
    out.mkdir(parents=True, exist_ok=True)

    if (ckpt / "model.safetensors").exists():
        backend = TorchBackend(config, str(ckpt))
    elif (ckpt / "params").is_dir():
        backend = JaxBackend(config, str(ckpt))
    else:
        raise FileNotFoundError(f"{ckpt}: neither model.safetensors nor params/")

    dataset = build_dataset(config, ckpt, args.test_dir, args.drop_idle)
    table = frame_table(args.test_dir, args.drop_idle)
    if len(table) != len(dataset):
        raise RuntimeError(f"frame table ({len(table)}) and dataset ({len(dataset)}) disagree")
    indices = list(range(0, len(dataset), args.frame_stride))
    groups = {**train_pytorch.get_action_metric_groups(config), **{f"g1/{k}": v for k, v in G1_GROUPS.items()}}
    print(f"[test] {len(indices)} samples (every {args.frame_stride}th of {len(dataset)} frames, "
          f"{table['episode_index'].nunique()} episodes)")

    rows, t0, done = [], time.time(), 0
    for batch in batches(dataset, indices, args.batch_size, args.num_workers):
        loss, pred = backend(batch, args.num_steps)
        m = per_sample_metrics(pred, np.asarray(batch["actions"], np.float32), groups, train_pytorch.ACCURACY_TOLERANCES)
        for i in range(len(loss)):
            rows.append({"loss": float(loss[i]), **{k: float(v[i]) for k, v in m.items()}})
        done += len(loss)
        print(f"  {done}/{len(indices)}  ({(time.time() - t0) / done * (len(indices) - done) / 60:.1f} min left)", end="\r")
    print()

    df = pd.DataFrame(rows)
    df = pd.concat([table.iloc[indices].reset_index(drop=True)[["episode_index", "frame_index", "task"]], df], axis=1)
    counts = df["task"].value_counts()
    df["task_group"] = df["task"].where(df["task"].map(counts) >= args.min_task_samples, RARE)
    df.to_csv(out / "per_sample.csv", index=False)

    metric_cols = [c for c in df.columns if "/" in c or c == "loss"]
    overall = df[metric_cols].mean().to_dict()
    by_task = df.groupby("task_group")[["loss", "all/l1", "all/first_step_l1", "all/acc@0.05", "g1/right_arm/l1",
                                        "g1/right_hand/l1"]].mean()
    by_task.insert(0, "samples", df["task_group"].value_counts())
    result = {"checkpoint": str(ckpt), "config": args.config_name, "test_dir": str(args.test_dir),
              "samples": len(df), "frame_stride": args.frame_stride, "num_steps": args.num_steps,
              "overall": overall, "by_task": by_task.reset_index().to_dict(orient="records")}
    json.dump(result, open(out / "metrics.json", "w"), indent=2)

    print(f"\n== {args.config_name} @ {ckpt.name} on {args.test_dir} ({len(df)} samples)")
    print("   same definitions as TensorBoard val/*:")
    for k in ("loss", "all/l1", "all/mse", "all/first_step_l1", "all/acc@0.05", "all/acc@0.1"):
        print(f"   {k:20s} {overall[k]:.4f}")
    print("   per joint group (l1, padding excluded):")
    for k in G1_GROUPS:
        print(f"   {k:20s} {overall[f'g1/{k}/l1']:.4f}")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n   by task:\n" + by_task.to_string())
    print(f"\nSaved {out / 'metrics.json'} and {out / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
