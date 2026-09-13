"""
Find the speed threshold that separates walking from stopped, using the
method built into MARGO itself.

From margo/utilities/blockActivity.m and kthresh_distribution.m:

    1. take every step speed
    2. take the log
    3. k-means the log speeds into two clusters (stopped, walking)
    4. the threshold is the midpoint of the gap between the two clusters
    5. moving = speed > threshold

Log first because the two modes are roughly log-normal: tracking jitter
piles up near zero and real walking sits well above, but on a linear axis
the jitter peak is so compressed that clustering separates the fast tail
from everything else instead of separating jitter from walking.

Edit SETTINGS, then run.
"""

from pathlib import Path

import numpy as np

from analyze_ymaze import PIXELS_PER_MM, get_xy, read_file

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")

# Speeds are sampled rather than using every frame — MARGO subsamples to
# 1000 points before clustering. More points give a steadier answer.
N_SAMPLE = 200_000

# Report the threshold as mm per second. MARGO's own "speed" is mm per
# FRAME (it never divides by elapsed time), so set this False to match
# MARGO's convention exactly.
PER_SECOND = True


# ============================================================
# MARGO's threshold
# ============================================================

def kmeans_1d(values, n_iter=100, seed=0):
    """Two-cluster k-means on a 1-D array. Returns the two centres."""
    rng = np.random.default_rng(seed)
    lo, hi = np.percentile(values, [10, 90])
    centres = np.array([lo, hi], dtype=float)

    for _ in range(n_iter):
        assign = np.abs(values[:, None] - centres[None, :]).argmin(axis=1)
        new = np.array([values[assign == k].mean() if (assign == k).any()
                        else centres[k] for k in range(2)])
        if np.allclose(new, centres):
            break
        centres = new
    return np.sort(centres), assign


def margo_threshold(speeds):
    """Speed threshold separating stopped from walking.

    Mirrors kthresh_distribution.m: cluster log speeds into two groups,
    then take the midpoint between the top of the lower cluster and the
    bottom of the upper one — the middle of the gap, not the midpoint
    between the cluster means.
    """
    s = np.asarray(speeds, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) < 100:
        return np.nan

    logs = np.log(s)
    centres, assign = kmeans_1d(logs)

    # re-derive assignment against the sorted centres
    assign = np.abs(logs[:, None] - centres[None, :]).argmin(axis=1)
    low, high = logs[assign == 0], logs[assign == 1]
    if not len(low) or not len(high):
        return np.nan

    bounds = np.sort([low.min(), low.max(), high.min(), high.max()])
    return float(np.exp(bounds[1:3].mean()))


# ============================================================
# Collect speeds
# ============================================================

def file_speeds(file_path, per_second=PER_SECOND):
    centroids, _, time = read_file(file_path)
    out = []
    for fly in range(centroids.shape[0]):
        x, y = get_xy(centroids[fly])
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 1000:
            continue

        step = np.hypot(np.diff(x), np.diff(y)) / PIXELS_PER_MM   # mm
        seconds = time[1:]
        usable = valid[:-1] & valid[1:] & np.isfinite(seconds) & (seconds > 0)

        s = step[usable] / seconds[usable] if per_second else step[usable]
        out.append(s[np.isfinite(s)])
    return np.concatenate(out) if out else np.array([])


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    files = sorted(DATA_DIR.rglob("*.hdf5"))
    if not files:
        raise SystemExit(f"No HDF5 files found under {DATA_DIR}")

    unit = "mm/s" if PER_SECOND else "mm/frame"
    rng = np.random.default_rng(0)

    print(f"Speed threshold per recording ({unit}):")
    per_file, pooled = [], []
    for file_path in files:
        s = file_speeds(file_path)
        if not len(s):
            continue
        if len(s) > N_SAMPLE:
            s = rng.choice(s, N_SAMPLE, replace=False)
        t = margo_threshold(s)
        per_file.append(t)
        pooled.append(s)
        print(f"  {file_path.parent.name}/{file_path.name}: {t:.3f}")

    allspeed = np.concatenate(pooled)
    if len(allspeed) > N_SAMPLE:
        allspeed = rng.choice(allspeed, N_SAMPLE, replace=False)
    overall = margo_threshold(allspeed)

    print("\n" + "=" * 60)
    print(f"POOLED THRESHOLD: {overall:.4f} {unit}")
    print("=" * 60)
    print(f"across files: median {np.median(per_file):.3f}, "
          f"range {np.min(per_file):.3f} to {np.max(per_file):.3f}")

    below = (allspeed < overall).mean() * 100
    print(f"\n{below:.1f}% of steps fall below the threshold (counted as stopped)")

    print("\nUse in analyze_ymaze.py:")
    print(f"    MIN_WALKING_SPEED = {overall:.3f}")

    # print("\nSanity check — plot this and confirm two humps with the")
    # print("threshold in the valley between them:")
    # print("    import matplotlib.pyplot as plt")
    # print("    plt.hist(np.log10(allspeed[allspeed>0]), bins=200)")
    # print(f"    plt.axvline(np.log10({overall:.4f}), color='r')")