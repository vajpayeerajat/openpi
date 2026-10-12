#cd /mnt/drive2/rajat_ws/tasks/openpi; /home/rajatvajpayee/venvs/openpi_cosmos/bin/python - <<'EOF' 2>&1 | grep -v Warn
import glob, json, numpy as np, pandas as pd
import argparse 


parser = argparse.ArgumentParser(description="Evaluate normalized statistics from a json file.")
parser.add_argument(
    "--json",
    type=str,
    default='assets/cosmos2_8b_g1_pickplace/train/norm_stats.json',
    help="Path to the norm_stats.json file."
)
args = parser.parse_args()


# st = json.load(open('assets/cosmos2_8b_g1_pickplace/train/norm_stats.json'))['norm_stats']['actions']
st = json.load(open(args.json))['norm_stats']['actions']
q01, q99 = np.array(st['q01']), np.array(st['q99'])
files = sorted(glob.glob('train/data/**/*.parquet', recursive=True))
H = 50
mask = np.array([1]*7 + [0]*7 + [1]*7 + [0]*7, bool)
chunks = []
rng = np.random.default_rng(0)
for f in rng.choice(files, 60, replace=False):
    df = pd.read_parquet(f, columns=['observation.state', 'action'])
    s = np.stack(df['observation.state'].to_numpy()); a = np.stack(df['action'].to_numpy())
    for t in rng.choice(len(df), 20):
        idx = np.minimum(np.arange(t, t + H), len(df) - 1)
        c = a[idx].copy(); c[:, mask] -= s[t, mask]   # DeltaActions on arms
        chunks.append(c)
c = np.stack(chunks)                                   # (N, 50, 28)
norm = (c - q01) / (q99 - q01 + 1e-6) * 2 - 1
print(f"{len(c)} sampled training chunks (batch-like), 50 steps x 28 dims\n")
print("dim | q99-q01 | max |normalized| | mean normalized^2")
for d in range(28):
    print(f"{d:3d} | {q99[d]-q01[d]:7.4f} | {np.abs(norm[...,d]).max():12.1f} | {np.mean(norm[...,d]**2):12.1f}")
print("\nnonzero frac in left-hand dims 7-13:", [round(float(np.mean(c[...,d] != 0)), 4) for d in range(7, 14)])
sq = norm**2
print("\nmean normalized^2 all dims: %.1f   excluding 7-13: %.3f" % (sq.mean(), np.delete(sq, range(7,14), -1).mean()))
per_chunk = sq.mean(axis=(1,2))
print("per-chunk mean^2: median %.2f, 90th pct %.1f, max %.1f" % (np.median(per_chunk), np.percentile(per_chunk,90), per_chunk.max()))



for k in ['mean','std','q01','q99']: print(k, [f"{v:.3e}" for v in st[k][7:14]])
a = np.concatenate([np.stack(pd.read_parquet(f, columns=['action'])['action'].to_numpy())[:, 7:14]
                    for f in glob.glob('train/data/**/*.parquet', recursive=True)])
print("\nfull train set, frames:", len(a))
print("nonzero frac:", np.round((a != 0).mean(0), 4))
print("min:", a.min(0)); print("max:", a.max(0)); print("std:", a.std(0))