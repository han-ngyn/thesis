"""
Figures for the Y-maze analysis.

Reads fly_metrics.csv (from analyze_ymaze.py) and writes every figure to
FIGURE_DIR. Edit SETTINGS, then run.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# ============================================================
# SETTINGS
# ============================================================

CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")
FIGURE_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/figures")
DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")

# ---- Figure sizes, in mm ----
MAX_WIDTH_MM = 180        # absolute ceiling
MAX_HEIGHT_MM = 210       # absolute ceiling

PANEL_MM = (85, 70)       # single metric panel — fits two side by side
WIDE_MM = (170, 60)       # raster

# Flies with fewer turns than this are dropped from turn-based figures.
# Report results at a few values to show this isn't driving the result.
MIN_TURNS = 20

# Order and labels for the x axis
TREATMENTS = ["control", "0.5uM", "1uM"]
LABELS = ["Control", "0.5 µM", "1 µM"]
COLORS = ["#35978f", "#dfc27d", "#bf812d"]

# Flies to show in the raster: (filename substring, fly index). One row each.
RASTER_FLIES = [
    ("control", 4),
    ("0.5uM", 12),
    ("1uM", 35),
]

N_BOOT = 2000

# How to arrange the dots in the per-fly plots.
#   "swarm"  - nudged sideways by density; shows shape, but with thousands of
#              flies packed into a narrow range it reads as horizontal bands
#   "jitter" - random horizontal offset, like the notebook version; looser
#              and easier to read when points are heavily concentrated
POINT_STYLE = "jitter"

# Type sizes tuned for small panels. At 85 mm wide, matplotlib's defaults
# are far too large and labels collide.
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


# ============================================================
# Helpers
# ============================================================

MM = 1 / 25.4     # millimetres to inches, which is what matplotlib wants


def point_colors(color, face_alpha=0.4, darken=0.62):
    """Translucent fill with a darker, more solid outline.

    Overlapping points stay readable: the fills blend so density shows as
    deeper colour, while the outlines keep individual flies distinguishable
    instead of merging into one blob.
    """
    r, g, b = mcolors.to_rgb(color)
    face = (r, g, b, face_alpha)
    edge = (r * darken, g * darken, b * darken, 0.9)
    return face, edge


def new_figure(size_mm):
    """Create a figure sized in mm, refusing anything over the limits."""
    width, height = size_mm
    if width > MAX_WIDTH_MM or height > MAX_HEIGHT_MM:
        raise ValueError(
            f"Figure {width}x{height} mm exceeds the "
            f"{MAX_WIDTH_MM}x{MAX_HEIGHT_MM} mm limit"
        )
    return plt.subplots(figsize=(width * MM, height * MM))


def bootstrap_ci(values, statistic=np.mean, n_boot=N_BOOT, seed=0):
    """95% confidence interval by resampling flies with replacement.

    Draw a new set of flies the same size as the real one, allowing repeats,
    recompute the statistic, and repeat N_BOOT times. The middle 95% of those
    results is the interval. Works for any statistic, including ones with no
    tidy formula.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    stats = statistic(values[draws], axis=1)
    return tuple(np.percentile(stats, [2.5, 97.5]))


def swarm_positions(values, center, width=0.34, n_rows=70,
                    min_spread=0.35, seed=0):
    """Beeswarm x positions: points nudged sideways so none overlap.

    Slices the y range into thin rows and spreads the points in each row
    around the centre. Rows holding more flies spread wider, so the width of
    the cloud at any height shows how many flies sit at that value.

    min_spread keeps sparse rows from collapsing onto the centre line, which
    is what creates a hard vertical stripe down the middle. A little random
    noise is added on top so the rows don't read as rigid columns.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([])
    if len(values) == 1:
        return np.array([float(center)])

    rng = np.random.default_rng(seed)

    lo, hi = values.min(), values.max()
    if hi == lo:
        row = np.zeros(len(values), dtype=int)
    else:
        edges = np.linspace(lo, hi, n_rows + 1)
        row = np.clip(np.digitize(values, edges) - 1, 0, n_rows - 1)

    fullest = np.bincount(row).max()

    x = np.full(len(values), float(center))
    for r in np.unique(row):
        members = np.flatnonzero(row == r)
        k = len(members)

        # how wide this row is allowed to spread, with a floor so thin rows
        # still occupy some width instead of stacking on the centre
        span = width * max(k / fullest, min_spread)

        if k == 1:
            offsets = rng.uniform(-span, span, 1)
        else:
            members = members[np.argsort(values[members])]   # tidy ordering
            offsets = np.linspace(-1, 1, k) * span

        x[members] = center + offsets + rng.normal(0, width * 0.035, k)
    return x


def tidy_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def groups(df, column, require_turns=True):
    """Split a column into one array per treatment, in TREATMENTS order."""
    out = []
    for treatment in TREATMENTS:
        sub = df[df["treatment"] == treatment]
        if require_turns:
            sub = sub[sub["n_turns"] >= MIN_TURNS]
        values = sub[column].to_numpy(dtype=float)
        out.append(values[np.isfinite(values)])
    return out


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    w, h = fig.get_size_inches() / MM
    plt.close(fig)
    print(f"  {path.name}  ({w:.0f}x{h:.0f} mm)")


# ============================================================
# Figure 1: one dot per fly
# ============================================================

def scatter_figure(df, column, ylabel, name, reference=None,
                   require_turns=True, log=False, style=None):
    """One dot per fly, with two kinds of error bar.

    The black bar is the bootstrap 95% CI of the mean: how sure we are about
    where the average sits. The spread of the flies themselves is shown by
    the point cloud, and quantified in the variability figures.
    """
    data = groups(df, column, require_turns)
    style = style or POINT_STYLE
    rng = np.random.default_rng(0)

    fig, ax = new_figure(PANEL_MM)
    if log:
        ax.set_yscale("log")

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        # 10^0, 10^1 rather than 1, 10 — signals the axis is log at a glance
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    for i, (values, color) in enumerate(zip(data, COLORS)):
        if not len(values):
            continue

        if style == "swarm":
            x = swarm_positions(values, i)
        else:
            x = i + rng.uniform(-0.3, 0.3, len(values))

        face, edge = point_colors(color)
        ax.scatter(x, values, facecolors=face, edgecolors=edge,
                   linewidths=0.35, s=11, zorder=2)

        # a log axis compresses the top, so summarize with the median and
        # geometric-style interval rather than a mean dragged up by the tail
        center = np.median(values) if log else values.mean()

        stat = np.median if log else np.mean
        lo, hi = bootstrap_ci(values, statistic=lambda a, axis: stat(a, axis=axis))
        ax.errorbar(i, center, yerr=[[center - lo], [hi - center]], fmt="none",
                    ecolor="black", capsize=2.5, lw=1.3, zorder=5)
        ax.hlines(center, i - 0.22, i + 0.22, color="black", lw=1.4, zorder=6)

    if reference is not None:
        ax.axhline(reference, color="grey", ls=":", lw=0.7, zorder=1)

    if log:
        # Round the bottom down to a whole power of ten so at least two
        # decade labels appear; leave the top tight, since rounding it up
        # too would waste most of the panel on empty space.
        allv = np.concatenate([v for v in data if len(v)])
        ax.set_ylim(10 ** np.floor(np.log10(allv.min())), allv.max() * 1.15)

    ax.set_xticks(range(len(TREATMENTS)))
    ax.set_xticklabels([f"{l}\nn={len(v)}" for l, v in zip(LABELS, data)])
    ax.set_xlim(-0.5, len(TREATMENTS) - 0.5)
    ax.set_ylabel(f"{ylabel} (log scale)" if log else ylabel)
    tidy_axes(ax)
    save(fig, name)


# ============================================================
# Figure 2: distributions as violins
# ============================================================

def distribution_figure(df, column="turn_bias",
                        xlabel="Turn bias (right turns / total)",
                        name="turn_bias_distribution", reference=0.5,
                        span=None):
    """Overlaid smooth density curves, one per treatment.

    A kernel density estimate replaces each fly with a small bump and adds
    them up, giving a smooth curve instead of blocky histogram bars. Curves
    are normalised to unit area, so groups of different sizes are directly
    comparable and there is no need to subsample the larger one.

    Overlaying rather than separating them is the point: differences in
    width between the curves are the visible signature of a change in
    behavioural variability.
    """
    data = groups(df, column)
    keep = [(v, c, l) for v, c, l in zip(data, COLORS, LABELS) if len(v) > 1]
    if not keep:
        print(f"  {name} skipped: not enough data")
        return

    if span is None:
        allv = np.concatenate([v for v, _, _ in keep])
        pad = 0.05 * (allv.max() - allv.min() or 1)
        span = (allv.min() - pad, allv.max() + pad)
    grid = np.linspace(span[0], span[1], 400)

    fig, ax = new_figure(PANEL_MM)

    for values, color, label in keep:
        density = gaussian_kde(values)(grid)
        ax.fill_between(grid, density, color=color, alpha=0.35, lw=0)
        ax.plot(grid, density, color=color, lw=1.6,
                label=f"{label} (n={len(values)})")

    if reference is not None:
        ax.axvline(reference, color="grey", ls="--", lw=0.7, zorder=1)

    ax.set_xlim(span)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised density")
    ax.legend(frameon=False, loc="upper right")
    tidy_axes(ax)
    save(fig, name)


# ============================================================
# Figure 3: variability
# ============================================================

def corrected_variance(bias, n_turns):
    """Among-fly variance in turn bias, with sampling noise removed.

    Each fly's bias is estimated from a finite number of turns, so what you
    observe is:

        Var(observed) = Var(true) + average binomial sampling variance

    Subtracting the second term matters. Without it, any group whose flies
    turn less often looks more variable, so a treatment that reduces activity
    would appear to increase behavioural variability on its own.
    """
    bias = np.asarray(bias, dtype=float)
    n_turns = np.asarray(n_turns, dtype=float)
    keep = np.isfinite(bias) & np.isfinite(n_turns) & (n_turns > 0)
    bias, n_turns = bias[keep], n_turns[keep]

    if len(bias) < 2:
        return np.nan
    return np.var(bias, ddof=1) - np.mean(bias * (1 - bias) / n_turns)


def variability_figure(df, column="turn_bias", ylabel=None, name=None,
                       corrected=False, require_turns=True):
    """Violin of the bootstrap distribution of among-fly spread.

    Resample the flies many times, recompute the spread each time, and draw
    the whole distribution of answers. The violin's width IS the uncertainty
    in the estimate, not the spread of the flies. Groups with more flies get
    narrower violins because their estimate is more precise.

    corrected=True is only valid for turn bias. Bias is a proportion, so its
    sampling variance has an exact formula that can be subtracted. Without
    that subtraction a group whose flies turn less often looks more variable
    for that reason alone. The other metrics have no equivalent formula, so
    they are reported as plain SD and read as descriptive.
    """
    ylabel = ylabel or f"Fly-to-fly variability in {column}"
    name = name or f"variability_{column}"

    fig, ax = new_figure(PANEL_MM)
    rng = np.random.default_rng(0)

    distributions, positions, colors, labels = [], [], [], []

    for i, (treatment, color) in enumerate(zip(TREATMENTS, COLORS)):
        sub = df[df["treatment"] == treatment]
        if require_turns:
            sub = sub[sub["n_turns"] >= MIN_TURNS]
        sub = sub.dropna(subset=[column] + (["n_turns"] if corrected else []))
        if len(sub) < 2:
            continue

        values = sub[column].to_numpy(float)
        values = values[np.isfinite(values)]
        draws = rng.integers(0, len(values), (N_BOOT, len(values)))

        if corrected:
            n = sub["n_turns"].to_numpy(float)
            boot = np.array([corrected_variance(values[d], n[d]) for d in draws])
            boot = np.sqrt(np.clip(boot, 0, None))
        else:
            boot = values[draws].std(axis=1, ddof=1)

        boot = boot[np.isfinite(boot)]
        if len(boot) < 2:
            continue

        distributions.append(boot)
        positions.append(i)
        colors.append(color)
        labels.append(f"{LABELS[i]}\nn={len(values)}")

    if not distributions:
        print(f"  {name} skipped: not enough data")
        return

    parts = ax.violinplot(distributions, positions=positions, widths=0.7,
                          showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

    for pos, boot in zip(positions, distributions):
        mean = boot.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.errorbar(pos, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="black", markersize=3, capsize=2.5, lw=1.1, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(TREATMENTS) - 0.5)
    ax.set_ylabel(ylabel)
    tidy_axes(ax)
    save(fig, name)


# ============================================================
# Figure 4: raster of individual flies' turns
# ============================================================

def list_files(data_dir=DATA_DIR):
    """Print every recording, so you can copy a name into RASTER_FLIES."""
    for f in sorted(data_dir.rglob("*.hdf5")):
        print(f"  {f.parent.name:10} {f.name}")


def raster_figure(chosen=RASTER_FLIES, data_dir=DATA_DIR):
    """Every turn a fly made, as a tick on a timeline, coloured by direction.

    One row per fly. Shows the raw sequence behind bias and switchiness — a
    strongly biased fly looks mostly one colour, an alternating fly striped.
    """
    from analyze_ymaze import detect_orientation, read_file, turn_directions

    files = sorted(data_dir.rglob("*.hdf5"))
    loaded = []

    for match, fly in chosen:
        hits = [f for f in files if match in str(f)]
        if not hits:
            print(f"  raster: nothing matches {match!r}. "
                  f"Run list_files() to see what's available.")
            continue
        if len(hits) > 1:
            print(f"  raster: {match!r} matches {len(hits)} files, "
                  f"using {hits[0].name}")

        file_path = hits[0]
        centroids, turns, time = read_file(file_path)
        if fly >= centroids.shape[0]:
            print(f"  raster: fly {fly} not in {file_path.name}")
            continue

        orientation = detect_orientation(centroids[fly])
        turn_frames = np.flatnonzero(turns[fly] > 0)

        # index the clock by the FRAMES where turns happened, not by turn
        # number, or every tick collapses into the start of the recording
        minutes = np.cumsum(time)[turn_frames] / 60
        directions = turn_directions(turns[fly][turn_frames], orientation)

        scored = np.isfinite(directions)
        if not scored.any():
            print(f"  raster: {file_path.name} fly {fly} orientation "
                  f"{orientation}, no scored turns")
            continue

        loaded.append({
            "label": f"{file_path.parent.name}\nfly {fly}",
            "minutes": minutes[scored],
            "directions": directions[scored],
        })

    if not loaded:
        print("  raster skipped: no usable flies")
        return

    height = min(14 * len(loaded) + 22, MAX_HEIGHT_MM)
    fig, ax = new_figure((WIDE_MM[0], height))

    for row, fly in enumerate(loaded):
        for value, color in [(1, "#bf812d"), (0, "#35978f")]:
            mask = fly["directions"] == value
            ax.vlines(fly["minutes"][mask], row - 0.35, row + 0.35,
                      color=color, lw=0.5)

    ax.set_yticks(range(len(loaded)))
    ax.set_yticklabels([f["label"] for f in loaded])
    ax.set_ylim(-0.7, len(loaded) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Time (minutes)")
    ax.set_title("Turns over time   (right = brown, left = teal)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, "raster_turns")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv(CSV)

    print(f"{len(df)} flies in {CSV.name}")
    kept = df[df["n_turns"] >= MIN_TURNS]
    print(f"{len(kept)} pass n_turns >= {MIN_TURNS}\n")
    print("Retention by treatment:")
    print(pd.DataFrame({
        "total": df.groupby("treatment").size(),
        "kept": kept.groupby("treatment").size(),
    }).assign(fraction=lambda d: (d["kept"] / d["total"]).round(3)))
    print("\nWriting figures:")

    # One dot per fly
    scatter_figure(df, "turn_bias", "Turn bias", "turn_bias_all_flies",
                   reference=0.5)
    scatter_figure(df, "n_turns", "Number of turns", "total_turns",
                   require_turns=False)
    scatter_figure(df, "distance", "Distance travelled (px)", "distance",
                   require_turns=False)
    scatter_figure(df, "walking_speed", "Walking speed (px/s)",
                   "walking_speed", require_turns=False)
    scatter_figure(df, "switchiness", "Switchiness", "switchiness", reference=1.0)
    scatter_figure(df, "clumpiness", "Clumpiness", "clumpiness", reference=1.0)
    scatter_figure(df, "tortuosity", "Tortuosity", "tortuosity")

    # Distributions
    distribution_figure(df, "turn_bias", "Turn bias (right turns / total)",
                        "turn_bias_distribution", reference=0.5, span=(0, 1))

    # Variability — bootstrap distribution of among-fly spread
    variability_figure(df, "turn_bias", "Fly-to-fly variability in turn bias",
                       "variability_turn_bias", corrected=True)
    variability_figure(df, "switchiness", "Fly-to-fly variability in switchiness",
                       "variability_switchiness")
    variability_figure(df, "clumpiness", "Fly-to-fly variability in clumpiness",
                       "variability_clumpiness")
    variability_figure(df, "tortuosity", "Fly-to-fly variability in tortuosity",
                       "variability_tortuosity")
    variability_figure(df, "walking_speed", "Fly-to-fly variability in walking speed (px/s)",
                       "variability_walking_speed", require_turns=False)
    variability_figure(df, "distance", "Fly-to-fly variability in distance (px)",
                       "variability_distance", require_turns=False)
    variability_figure(df, "n_turns", "Fly-to-fly variability in turn count",
                       "variability_n_turns", require_turns=False)

    raster_figure()

    print("\nDone.")