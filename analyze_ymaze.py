"""
Per-fly Y-maze metrics.

Walks every HDF5 under DATA_DIR, computes one row per fly, saves a CSV.
See METRICS.md for what each number means.
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

# Camera calibration, from calibrate_pixels.py. Run that first and paste
# the number it prints here.
PIXELS_PER_MM = 3.2742

# Walking is classified with a Schmitt trigger, following Corfas, Sharma &
# Dickinson 2019 (Curr Biol 29:1660): a fly counts as stopped once its speed
# falls below STOP_BELOW, and as walking once it rises above WALK_ABOVE.
# Between the two it keeps whatever state it was already in. mm/second.
STOP_BELOW = 1.0
WALK_ABOVE = 3.0

# Faster than this is a tracking glitch, not a fly. mm/second. The paper
# rejects jumps over 1.5 mm between consecutive frames, which at their 30 Hz
# works out near 45 mm/s.
MAX_PLAUSIBLE_SPEED = 50.0


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
    """Arm IDs -> turns. 1 = right, 0 = left, nan = couldn't be scored.

    A turn is worked out from which arm the fly came from and which it went
    to. In a Y-maze every move is a left or a right — there's no straight.

    Subtracting the arm numbers tells you which way, except they're numbered
    1, 2, 3 in a circle, so going past 3 wraps back to 1 and the subtraction
    breaks. The remainder after dividing by 3 fixes that: one way around
    always gives 1, the other always gives 2. Which of those is the fly's
    right depends on how the maze points, hence the orientation argument.

    Same length as the input, unscored turns left as nan, so the caller can
    tell turns 5 and 7 weren't adjacent just because 6 went missing.
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

    # Bias: right turns / scored turns. 0.5 = no preference.
    bias = float(directions[scored].mean()) if scored.any() else np.nan

    # Switchiness: does the fly alternate more or less than you'd expect from
    # its bias alone? A fly that turns right 90% of the time can't switch
    # often, so we compare against a coin with the same bias, which switches
    # 2*r*(1-r) of the time.
    #   =1 memoryless | >1 alternates | <1 repeats itself
    # Only pairs where BOTH turns scored: if turn 6 is missing, turns 5 and 7
    # weren't really adjacent and pairing them invents a switch.
    adjacent = scored[:-1] & scored[1:]
    n_pairs = int(adjacent.sum())
    if n_pairs and 0 < bias < 1:
        switches = int(np.sum(directions[:-1][adjacent] != directions[1:][adjacent]))
        switchiness = switches / (2 * bias * (1 - bias) * n_pairs)
    else:
        switchiness = np.nan

    # Clumpiness: are the turns clustered in time or spread out? Take the gaps
    # between turns, divide their spread by their average. Dividing by the
    # average is what lets you compare a slow fly to a fast one.
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

def walking_mask(speed, stop_below=STOP_BELOW, walk_above=WALK_ABOVE):
    """Which frames count as walking, by Schmitt trigger.

    A single cutoff makes a fly hovering near it flicker between walking and
    stopped many times a second, which inflates the count of transitions and
    makes the walking-speed average depend on exactly where the cutoff sits.
    Two thresholds with hysteresis fix that: the fly must exceed walk_above
    to be called walking, and fall under stop_below to be called stopped.
    In between it keeps its current state.
    """
    state = np.full(len(speed), np.nan)
    state[speed > walk_above] = 1.0
    state[speed < stop_below] = 0.0

    # carry the last decided state forward across the ambiguous band
    idx = np.where(~np.isnan(state), np.arange(len(state)), 0)
    np.maximum.accumulate(idx, out=idx)
    state = state[idx]
    state[np.isnan(state)] = 0.0        # before any crossing, count as stopped
    return state.astype(bool)


def path_tortuosity(x, y, turn_frames):
    """Path walked between consecutive turns / straight line between them.
    1.0 = perfectly straight. Median across traversals.

    Measured between turns rather than across the whole recording: after
    hours in a closed maze the fly ends up wherever it happens to be, so a
    whole-recording straight line is an arbitrary number.
    """
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
    blank = {"distance_mm": np.nan, "walking_speed_mm_s": np.nan,
             "fraction_walking": np.nan, "tortuosity": np.nan}

    x, y = get_xy(centroid)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return blank

    # Step between frame i and i+1 by Pythagoras: sqrt(dx^2 + dy^2)
    step = np.hypot(np.diff(x), np.diff(y))

    # time[i] is the gap BEFORE frame i, so the step i -> i+1 took time[i+1]
    seconds = time[1:]

    # Keep only steps where both frames were tracked and the clock moved.
    # Stripping NaNs first would leave frame 500 next to frame 900 and turn
    # a tracking gap into one huge step the fly never walked.
    usable = valid[:-1] & valid[1:] & np.isfinite(seconds) & (seconds > 0)
    step, seconds = step[usable], seconds[usable]
    if not len(step):
        return blank

    # convert to mm before thresholding, so the cutoffs mean what they say
    step = step / PIXELS_PER_MM
    speed = step / seconds
    real = speed < MAX_PLAUSIBLE_SPEED          # drop tracking glitches
    step, speed = step[real], speed[real]
    if not len(step):
        return blank

    moving = walking_mask(speed)

    return {
        "distance_mm": float(step.sum()),
        "walking_speed_mm_s": float(speed[moving].mean()) if moving.any() else np.nan,
        "fraction_walking": float(moving.mean()),
        # tortuosity is a ratio, so it is unitless and needs no conversion
        "tortuosity": path_tortuosity(x, y, np.flatnonzero(turns > 0)),
    }

# ============================================================
# Cohort and session
# ============================================================

def label_sessions(df):
    """Tag each fly in fly_metrics.csv with its cohort and light condition.

    session is one of July, Oct-on, Oct-off. July flies were recorded once,
    with lights off; October/November flies were recorded twice and appear
    in both Oct-on and Oct-off, which is why those two are never combined:
    a pooled column would count each animal twice.

    The light suffix has to be matched carefully: every filename contains
    the word "control", which contains the letters "on", so a plain search
    for "on" would label every control fly as lights-on.
    """
    df = df.copy()
    df["month"] = df["file"].str.extract(r"^NNYM_(\d{2})-")[0]
    df["cohort"] = np.where(df["month"] == "07", "July", "Oct-Nov")

    low = df["file"].str.lower()
    off = low.str.contains(r"lights?_?off|_off\.", regex=True)
    on = low.str.contains(r"lights?_?on|_on\.", regex=True)
    df["lights"] = np.where(off, "off", np.where(on, "on", "not recorded"))

    df["session"] = np.where(df["cohort"] == "July", "July",
                             np.where(df["lights"] == "on", "Oct-on",
                                      np.where(df["lights"] == "off",
                                               "Oct-off", "unassigned")))
    return df


# ============================================================
# Fly-to-fly spread
# ============================================================

def corrected_variance(bias, n_turns):
    """Among-fly variance in turn bias with binomial sampling noise removed.

    Observed spread = real spread + noise from estimating each fly's bias
    from a finite number of turns. Without subtracting the second term, a
    group whose flies turn less often looks more variable for that reason.
    """
    bias = np.asarray(bias, dtype=float)
    n_turns = np.asarray(n_turns, dtype=float)
    keep = np.isfinite(bias) & np.isfinite(n_turns) & (n_turns > 0)
    bias, n_turns = bias[keep], n_turns[keep]
    if len(bias) < 2:
        return np.nan
    return np.var(bias, ddof=1) - np.mean(bias * (1 - bias) / n_turns)


def spread_of(sub, column, corrected):
    """Fly-to-fly spread: corrected SD for turn bias, plain SD otherwise.

    The corrected variance can come out slightly negative when the true
    among-fly spread is near zero, so it is clipped before the square root.
    """
    v = sub[column].to_numpy(float)
    if corrected:
        return np.sqrt(max(corrected_variance(
            v, sub["n_turns"].to_numpy(float)), 0.0))
    v = v[np.isfinite(v)]
    return v.std(ddof=1) if len(v) > 1 else np.nan

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