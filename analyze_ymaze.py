"""
Per-fly Y-maze metrics.

Walks every HDF5 under DATA_DIR, computes one row per fly, saves a CSV.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")
OUTPUT_CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")

# Below this speed the fly counts as stopped, and those frames are left out
# of the walking-speed average. Pixels/second — pick from your own data.
MIN_WALKING_SPEED = 3.0

# Faster than this is a tracking glitch, not a fly.
MAX_PLAUSIBLE_SPEED = 150.0


# ============================================================
# Reading
# ============================================================

def read_file(file_path):
    """centroid (flies, 2, frames) px | Turns (flies, frames) arm ID or 0
    | time (frames,) seconds since previous frame"""
    with h5py.File(file_path, "r") as handle:
        centroids = np.asarray(handle["centroid"])
        turns = np.asarray(handle["Turns"])
        time = np.asarray(handle["time"])[0]

    if centroids.ndim != 3 or centroids.shape[1] != 2:
        raise ValueError(f"Unexpected centroid shape in {file_path.name}: {centroids.shape}")
    return centroids, turns, time


def get_xy(centroid):
    """y is negated because image coordinates count downward from the top."""
    return centroid[0], -centroid[1]


# ============================================================
# Maze orientation
# ============================================================

def has_horizontal_spread(points, y_bin_size=5, x_threshold=25):
    """True if points at similar heights are spread out horizontally.
    Two arms side by side produce that spread; a single arm does not."""
    if len(points) < 10:
        return False

    bands = np.round(points[:, 1] / y_bin_size) * y_bin_size
    x_by_band = {}
    for x_value, band in zip(points[:, 0], bands):
        x_by_band.setdefault(band, []).append(x_value)

    return any(len(v) >= 2 and np.ptp(v) > x_threshold for v in x_by_band.values())


def detect_orientation(centroid):
    """A Y-maze has two arms at one end, one at the other. Whichever end
    shows horizontal spread is the two-armed end.

    Returns "upright", "inverse", "ambiguous", or "unknown". This decides
    which rotation counts as a right turn, so an error flips left/right.
    """
    x, y = get_xy(centroid)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 50 or np.ptp(y) == 0:
        return "unknown"

    y_scaled = (y - y.min()) / np.ptp(y)     # 0-1, so "top" means the same everywhere
    top = np.column_stack((x[y_scaled > 0.8], y[y_scaled > 0.8]))
    bottom = np.column_stack((x[y_scaled < 0.2], y[y_scaled < 0.2]))

    top_arms = has_horizontal_spread(top)
    bottom_arms = has_horizontal_spread(bottom)

    if top_arms and not bottom_arms:
        return "upright"
    if bottom_arms and not top_arms:
        return "inverse"
    return "ambiguous" if top_arms and bottom_arms else "unknown"


# ============================================================
# Turn direction
# ============================================================

def turn_directions(arm_sequence, orientation):
    """determining left or right turns

    Arms are numbered 1,2,3 around the maze. Going from arm a to arm b,
    (b - a) mod 3 is 1 or 2 — the two rotation directions. Which one is
    "right" from the fly's view depends on maze orientation.

    """
    arm_sequence = np.asarray(arm_sequence, dtype=float)
    out = np.full(len(arm_sequence), np.nan)

    if orientation not in ("upright", "inverse") or len(arm_sequence) < 2:
        return out

    steps = np.mod(np.diff(arm_sequence), 3)
    right = 1 if orientation == "upright" else 2
    out[1:] = np.where(steps == right, 1.0, np.where(steps == 3 - right, 0.0, np.nan))
    return out


# ============================================================
# Turn metrics
# ============================================================

def turn_metrics(turns, time, orientation):
    turn_frames = np.flatnonzero(turns > 0)
    directions = turn_directions(turns[turn_frames], orientation)
    scored = np.isfinite(directions)

    # n_turns counts arm entries. turn_bias and switchiness also need to know
    # which way the maze points, so they are nan when orientation failed even
    # though n_turns may be large.
    n_turns = len(turn_frames)

    # Bias: right turns / scored turns.
    bias = float(directions[scored].mean()) if scored.any() else np.nan

    # Switchiness: total number of alternating transitions normalized by the
    # fly's overall turning bias.
    #   =1 memoryless | >1 alternates | <1 repeats itself
    # Only pairs where BOTH turns scored, so gaps never fake adjacency.
    adjacent = scored[:-1] & scored[1:]
    n_pairs = int(adjacent.sum())
    if n_pairs and 0 < bias < 1:
        switches = int(np.sum(directions[:-1][adjacent] != directions[1:][adjacent]))
        switchiness = switches / (2 * bias * (1 - bias) * n_pairs)
    else:
        switchiness = np.nan

    # Clumpiness: are fly turns clustered or are they spread out?
    # Take the gaps between turns, then divide their spread by their average.
    #   =1 random | >1 bursty | <1 evenly spaced
    gaps = np.diff(np.cumsum(time)[turn_frames])
    clumpiness = float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 and np.mean(gaps) > 0 else np.nan

    return {
        "n_turns": n_turns,
        "turn_bias": bias,
        "switchiness": switchiness,
        "clumpiness": clumpiness,
    }


# ============================================================
# Movement metrics
# ============================================================

def path_tortuosity(x, y, turn_frames):
    """Path walked between consecutive turns / straight line between them.
    1.0 = perfectly straight. Median across traversals."""
    ratios = []
    for start, end in zip(turn_frames[:-1], turn_frames[1:]):
        sx, sy = x[start:end + 1], y[start:end + 1]

        # skip traversals with tracking gaps rather than cutting across them
        if len(sx) < 2 or not np.all(np.isfinite(sx) & np.isfinite(sy)):
            continue

        path = np.sum(np.hypot(np.diff(sx), np.diff(sy)))
        straight = np.hypot(sx[-1] - sx[0], sy[-1] - sy[0])
        if straight > 0:
            ratios.append(path / straight)

    return float(np.median(ratios)) if ratios else np.nan


def movement_metrics(centroid, turns, time):
    blank = {"distance": np.nan, "walking_speed": np.nan, "tortuosity": np.nan}

    x, y = get_xy(centroid)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return blank

    # Step between frame i and i+1 by Pythagoras: sqrt(dx^2 + dy^2)
    step = np.hypot(np.diff(x), np.diff(y))

    # time[i] is the gap BEFORE frame i, so the step i -> i+1 took time[i+1]
    seconds = time[1:]

    # keep only steps where both frames were tracked and the clock moved
    usable = valid[:-1] & valid[1:] & np.isfinite(seconds) & (seconds > 0)
    step, seconds = step[usable], seconds[usable]
    if not len(step):
        return blank

    speed = step / seconds
    real = speed < MAX_PLAUSIBLE_SPEED
    step, speed = step[real], speed[real]
    if not len(step):
        return blank

    moving = speed > MIN_WALKING_SPEED

    return {
        "distance": float(step.sum()),
        "walking_speed": float(speed[moving].mean()) if moving.any() else np.nan,
        "tortuosity": path_tortuosity(x, y, np.flatnonzero(turns > 0)),
    }


# ============================================================
# Run
# ============================================================

def analyze_all(data_dir=DATA_DIR):
    files = sorted(data_dir.rglob("*.hdf5"))
    if not files:
        raise SystemExit(f"No HDF5 files found under {data_dir}")

    rows = []
    for file_path in files:
        centroids, turns, time = read_file(file_path)
        print(f"{file_path.parent.name}/{file_path.name}: {centroids.shape[0]} flies")

        for fly in range(centroids.shape[0]):
            orientation = detect_orientation(centroids[fly])
            rows.append({
                "file": file_path.name,
                "treatment": file_path.parent.name,
                "fly": fly,
                "orientation": orientation,
                **turn_metrics(turns[fly], time, orientation),
                **movement_metrics(centroids[fly], turns[fly], time),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = analyze_all()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(results)} flies to {OUTPUT_CSV}\n")
    print(results.groupby(["treatment", "orientation"]).size())