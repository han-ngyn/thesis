"""
Figures for the Y-maze analysis.

Reads fly_metrics.csv (from analyze_ymaze.py) and writes every figure to
FIGURE_DIR:

    fig1b_raster                    example turn sequences
    fig2a_turns, fig2b_distance     locomotor activity
    fig3a_turn_bias,
    fig3b_turn_bias_distribution,
    fig3c_variability_turn_bias     locomotor handedness
    fig4a_switchiness,
    fig4b_clumpiness                turn sequence and timing
    supp1_control_cohorts           control flies across cohorts/sessions

Every panel is split by cohort and session rather than pooling:

    July            control vs 0.5 uM   (one 2 h session, lights off)
    Oct/Nov  on     control vs 1 uM     (first 2 h session, lights on)
    Oct/Nov  off    control vs 1 uM     (second 2 h session, lights off)
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from analyze_ymaze import (
    corrected_variance,
    detect_orientation,
    label_sessions,
    read_file,
    turn_directions,
)

# ============================================================
# SETTINGS
# ============================================================

CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")
FIGURE_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/figures")
DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")

# ---- Figure sizes, in mm ----
MAX_WIDTH_MM = 180        # absolute ceiling
MAX_HEIGHT_MM = 210       # absolute ceiling

PANEL_MM = (115, 72)      # six-column panel — one per page-width row
DIST_MM = (170, 58)       # three side-by-side density panels

MIN_TURNS = 20
N_BOOT = 2000

COLOR = {
    "control": "#35978f",
    "0.5uM": "#dfc27d",
    "1uM": "#bf812d",
}

# The six columns: (treatment, session, tick label, x position).
# The 0.8 gap inside a pair and 1.2 between pairs make each dose read
# against its own control rather than across the panel.
GROUPS = [
    ("control", "July",    "Ctl",    0.0),
    ("0.5uM",   "July",    "0.5 µM", 0.8),
    ("control", "Oct-on",  "Ctl",    2.0),
    ("1uM",     "Oct-on",  "1 µM",   2.8),
    ("control", "Oct-off", "Ctl",    4.0),
    ("1uM",     "Oct-off", "1 µM",   4.8),
]

SESSION_LABELS = [
    (0.4, "July\nlights off"),
    (2.4, "Oct/Nov\nlights on"),
    (4.4, "Oct/Nov\nlights off"),
]

X_LIMITS = (-0.6, 5.4)

# Panels for the turn bias distribution figure: one comparison each.
DIST_PANELS = [
    ("July", "0.5uM", "July (lights off)"),
    ("Oct-on", "1uM", "Oct/Nov (lights on)"),
    ("Oct-off", "1uM", "Oct/Nov (lights off)"),
]

# Supplementary figure: control flies only, one column per session.
SUPP_GROUPS = [
    ("July", "July\nlights off", "#35978f"),
    ("Oct-on", "Oct/Nov\nlights on", "#dfae3a"),
    ("Oct-off", "Oct/Nov\nlights off", "#4a3f8f"),
]
SUPP_METRICS = [
    ("turn_bias", "Turn bias", True),
    ("switchiness", "Switchiness", True),
    ("clumpiness", "Clumpiness", True),
    ("n_turns", "Number of turns", False),
    ("distance_mm", "Distance (mm)", False),
]

# Flies to show in the raster: (filename substring, fly index). One row each.
RASTER_FLIES = [
    ("control", 4),
    ("0.5uM", 12),
    ("1uM", 35),
]
# A 10-minute window makes individual turns distinguishable; across two
# hours the ticks merge into a solid block.
RASTER_WINDOW_MIN = (30, 40)

# Left and right turn colours, matching the schematic in Figure 1A. These
# are deliberately NOT the treatment colours: Figure 1B is about one fly's
# left/right choices, not about which group it came from.
RIGHT_COLOR = "#128b22"
LEFT_COLOR = "#681A98"

# "swarm" shows distribution shape but reads as bands when hundreds of flies
# sit in a narrow range; "jitter" is looser and easier to read.
POINT_STYLE = "jitter"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
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


def select(df, treatment, session, column, require_turns=True):
    """Finite values of one column for one treatment in one session."""
    sub = df[(df["treatment"] == treatment) & (df["session"] == session)]
    if require_turns:
        sub = sub[sub["n_turns"] >= MIN_TURNS]
    sub = sub.dropna(subset=[column])
    values = sub[column].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def point_colors(color, face_alpha=0.4, darken=0.62):
    """Translucent fill with a darker, more solid outline.

    Overlapping points stay readable: the fills blend so density shows as
    deeper colour, while the outlines keep individual flies distinguishable
    instead of merging into one blob.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r, g, b, face_alpha), (r * darken, g * darken, b * darken, 0.9)


def new_figure(size_mm, ncols=1):
    """Create a figure sized in mm, refusing anything over the limits."""
    width, height = size_mm
    if width > MAX_WIDTH_MM or height > MAX_HEIGHT_MM:
        raise ValueError(
            f"Figure {width}x{height} mm exceeds the "
            f"{MAX_WIDTH_MM}x{MAX_HEIGHT_MM} mm limit"
        )
    return plt.subplots(1, ncols, figsize=(width * MM, height * MM))


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """95% CI of the mean by resampling flies with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    return tuple(np.percentile(values[draws].mean(axis=1), [2.5, 97.5]))


def swarm_positions(values, center, width=0.30, n_rows=70,
                    min_spread=0.35, seed=0):
    """Beeswarm x positions: points nudged sideways so none overlap.

    Rows holding more flies spread wider, so the width of the cloud at any
    height shows how many flies sit at that value. min_spread keeps sparse
    rows from collapsing onto the centre line, which is what creates a hard
    vertical stripe down the middle.
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


def draw_points(ax, values, pos, color, rng, style, size=9):
    """One dot per fly, plus the mean and its bootstrap CI."""
    if style == "swarm":
        x = swarm_positions(values, pos)
    else:
        x = pos + rng.uniform(-0.28, 0.28, len(values))

    face, edge = point_colors(color)
    ax.scatter(x, values, facecolors=face, edgecolors=edge,
               linewidths=0.35, s=size, zorder=2)

    lo, hi = bootstrap_ci(values)
    ax.errorbar(pos, values.mean(),
                yerr=[[values.mean() - lo], [hi - values.mean()]],
                fmt="none", ecolor="black", capsize=2.5, lw=1.3, zorder=5)
    ax.hlines(values.mean(), pos - 0.22, pos + 0.22, color="black",
              lw=1.4, zorder=6)


def label_group_axis(ax, positions, labels):
    """Treatment ticks, with the session named once under each pair."""
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(*X_LIMITS)
    for pos, text in SESSION_LABELS:
        ax.text(pos, -0.16, text, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5)


def save(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight")
    w, h = fig.get_size_inches() / MM
    plt.close(fig)
    print(f"  {path.name}  ({w:.0f}x{h:.0f} mm)")


# ============================================================
# One dot per fly
# ============================================================

def scatter_figure(df, column, ylabel, name, reference=None,
                   require_turns=True, style=None):
    """One dot per fly, grouped by session.

    The black bar is the bootstrap 95% CI of the mean: how sure we are about
    where the average sits. The spread of the flies themselves is shown by
    the point cloud, and quantified in the variability figure.
    """
    style = style or POINT_STYLE
    rng = np.random.default_rng(0)

    fig, ax = new_figure(PANEL_MM)
    positions, ticks = [], []

    for treatment, session, tick, pos in GROUPS:
        values = select(df, treatment, session, column, require_turns)
        positions.append(pos)
        ticks.append(f"{tick}\nn={len(values)}")
        if len(values):
            draw_points(ax, values, pos, COLOR[treatment], rng, style)

    if reference is not None:
        ax.axhline(reference, color="grey", ls=":", lw=0.7, zorder=1)

    label_group_axis(ax, positions, ticks)
    ax.set_ylabel(ylabel)
    tidy_axes(ax)
    save(fig, name)


# ============================================================
# Turn bias distribution
# ============================================================

def distribution_figure(df, column="turn_bias",
                        xlabel="Turn bias (right turns / total)",
                        name="fig3b_turn_bias_distribution", reference=0.5,
                        span=(0, 1)):
    """One panel per comparison, each overlaying dose on its own control.

    A kernel density estimate replaces each fly with a small bump and adds
    them up, giving a smooth curve instead of blocky histogram bars. Curves
    are normalised to unit area, so groups of different sizes are directly
    comparable. Sessions get separate panels rather than more curves in one,
    because six overlapping curves cannot be read.
    """
    fig, axes = new_figure(DIST_MM, ncols=len(DIST_PANELS))
    axes = np.atleast_1d(axes)
    grid = np.linspace(span[0], span[1], 400)

    for ax, (session, dose, title) in zip(axes, DIST_PANELS):
        for treatment in ("control", dose):
            values = select(df, treatment, session, column)
            if len(values) < 2:
                continue
            density = gaussian_kde(values)(grid)
            label = ("Control" if treatment == "control"
                     else dose.replace("uM", " µM"))
            ax.fill_between(grid, density, color=COLOR[treatment],
                            alpha=0.35, lw=0)
            ax.plot(grid, density, color=COLOR[treatment], lw=1.6,
                    label=f"{label} (n={len(values)})")

        if reference is not None:
            ax.axvline(reference, color="grey", ls="--", lw=0.7, zorder=1)

        ax.set_xlim(span)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right")
        tidy_axes(ax)

    axes[0].set_ylabel("Normalised density")
    fig.tight_layout()
    save(fig, name)


# ============================================================
# Variability
# ============================================================

def variability_figure(df, column="turn_bias", ylabel=None, name=None,
                       corrected=False, require_turns=True):
    """Violin of the bootstrap distribution of among-fly spread.

    Resample the flies many times, recompute the spread each time, and draw
    the whole distribution of answers. The violin's width IS the uncertainty
    in the estimate, not the spread of the flies. Groups with more flies get
    narrower violins because their estimate is more precise.

    corrected=True is only valid for turn bias: bias is a proportion, so its
    sampling variance has an exact formula that can be subtracted.
    """
    ylabel = ylabel or f"Fly-to-fly variability in {column}"
    name = name or f"variability_{column}"

    fig, ax = new_figure(PANEL_MM)
    rng = np.random.default_rng(0)

    distributions, violin_at, colors = [], [], []
    positions, ticks = [], []

    for treatment, session, tick, pos in GROUPS:
        sub = df[(df["treatment"] == treatment) & (df["session"] == session)]
        if require_turns:
            sub = sub[sub["n_turns"] >= MIN_TURNS]
        sub = sub.dropna(subset=[column] + (["n_turns"] if corrected else []))

        positions.append(pos)
        ticks.append(f"{tick}\nn={len(sub)}")
        if len(sub) < 2:
            continue

        values = sub[column].to_numpy(float)
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
        violin_at.append(pos)
        colors.append(COLOR[treatment])

    if not distributions:
        print(f"  {name} skipped: not enough data")
        plt.close(fig)
        return

    parts = ax.violinplot(distributions, positions=violin_at, widths=0.62,
                          showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)

    for pos, boot in zip(violin_at, distributions):
        mean = boot.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.errorbar(pos, mean, yerr=[[mean - lo], [hi - mean]], fmt="o",
                    color="black", markersize=3, capsize=2.5, lw=1.1, zorder=4)

    label_group_axis(ax, positions, ticks)
    ax.set_ylabel(ylabel)
    tidy_axes(ax)
    save(fig, name)


# ============================================================
# Supplementary figure: control flies only
# ============================================================

def control_cohorts_figure(df, name="supp1_control_cohorts"):
    """Control flies across both cohorts and both sessions.

    This is the benchmark the dose effects are judged against: how much
    behaviour moves with no drug involved. Turn bias barely shifts while
    activity changes a great deal, which is why each dose is compared only
    to controls from its own cohort and session.
    """
    ncol = 3
    nrow = int(np.ceil(len(SUPP_METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 56 * MM, nrow * 46 * MM))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, (column, label, require_turns) in zip(axes, SUPP_METRICS):
        if column not in df:
            continue
        ticks = []
        for i, (session, tick, color) in enumerate(SUPP_GROUPS):
            values = select(df, "control", session, column, require_turns)
            ticks.append(f"{tick}\nn={len(values)}")
            if len(values):
                draw_points(ax, values, i, color, rng, POINT_STYLE, size=5)

        ax.set_xticks(range(len(SUPP_GROUPS)))
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.6, len(SUPP_GROUPS) - 0.4)
        ax.set_ylabel(label)
        tidy_axes(ax)

    for ax in axes[len(SUPP_METRICS):]:
        ax.set_visible(False)

    fig.tight_layout()
    save(fig, name)


# ============================================================
# Raster of individual flies' turns
# ============================================================

def list_files(data_dir=DATA_DIR):
    """Print every recording, so you can copy a name into RASTER_FLIES."""
    for f in sorted(data_dir.rglob("*.hdf5")):
        print(f"  {f.parent.name:10} {f.name}")


def pick_raster_flies(df, min_turns=100):
    """Three flies spanning the range of turn bias: left-biased, unbiased,
    right-biased.

    A contrast makes the individuality point far better than three average
    flies would — the reference figures in the literature do the same.
    """
    kept = df[df["n_turns"] >= min_turns].dropna(subset=["turn_bias"])
    if len(kept) < 3:
        print("  raster: not enough flies to choose from, using RASTER_FLIES")
        return RASTER_FLIES

    low = kept.loc[kept["turn_bias"].idxmin()]
    mid = kept.loc[(kept["turn_bias"] - 0.5).abs().idxmin()]
    high = kept.loc[kept["turn_bias"].idxmax()]

    print("  raster flies chosen by turn bias:")
    for tag, r in [("left-biased", low), ("unbiased", mid),
                   ("right-biased", high)]:
        print(f"    {tag:13} {r['file']} fly {int(r['fly'])}  "
              f"bias {r['turn_bias']:.2f}  n={int(r['n_turns'])}")
    return [(r["file"], int(r["fly"])) for r in (low, mid, high)]


def raster_figure(chosen=RASTER_FLIES, data_dir=DATA_DIR,
                  window=RASTER_WINDOW_MIN):
    """Each fly's turns as ticks on its own timeline.

    Right turns sit on the upper sub-row and left turns on the lower, so the
    balance between them reads directly instead of having to be judged from
    colour. Counts and bias are computed over the plotted window, so they
    describe what is actually drawn rather than the whole recording.
    """
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

        minutes, directions = minutes[scored], directions[scored]

        if window is not None:
            keep = (minutes >= window[0]) & (minutes <= window[1])
            if keep.sum() < 2:
                print(f"  raster: {file_path.name} fly {fly} has only "
                      f"{keep.sum()} turns in {window[0]}-{window[1]} min")
                continue
            minutes, directions = minutes[keep], directions[keep]

        loaded.append({"minutes": minutes, "directions": directions})

    if not loaded:
        print("  raster skipped: no usable flies")
        return

    # One panel per fly, each with its own time axis. Built with add_axes
    # rather than subplots so the vertical spacing is set in mm.
    plot_w = 120.0                 # mm, the tick area
    left, right = 22.0, 18.0       # mm, space for counts and bias text
    panel_h, gap_h = 14.0, 12.0    # mm, plot height then room for the axis
    bottom, top = 4.0, 3.0

    W = plot_w + left + right
    H = len(loaded) * (panel_h + gap_h) + bottom + top
    if W > MAX_WIDTH_MM or H > MAX_HEIGHT_MM:
        raise ValueError(f"Raster {W:.0f}x{H:.0f} mm exceeds the limit")

    fig = plt.figure(figsize=(W * MM, H * MM))

    # Sub-rows sit well apart and the tick marks are short, so the two row
    # labels cannot collide however long the counts get.
    OFFSET, HALF = 0.30, 0.17

    for i, fly in enumerate(loaded):
        # stack downward, first fly at the top
        y0 = bottom + (len(loaded) - 1 - i) * (panel_h + gap_h)
        ax = fig.add_axes([left / W, y0 / H, plot_w / W, panel_h / H])

        n_right = int((fly["directions"] == 1).sum())
        n_left = int((fly["directions"] == 0).sum())
        total = n_right + n_left
        bias = n_right / total if total else np.nan

        # counts on a single line each — two-line labels are what collide
        yticks, ylabels = [], []
        for value, color, off, label in [
            (1, RIGHT_COLOR, +OFFSET, f"Right: {n_right}"),
            (0, LEFT_COLOR, -OFFSET, f"Left: {n_left}"),
        ]:
            t = fly["minutes"][fly["directions"] == value]
            ax.vlines(t, off - HALF, off + HALF, color=color, lw=1.2)
            yticks.append(off)
            ylabels.append(label)

        # No treatment label: the figure is about the spread of turn bias
        # between individuals, not about which group they came from.
        ax.text(1.02, 0.5, f"Turn bias\n{bias:.2f}", transform=ax.transAxes,
                va="center", ha="left", fontsize=7)

        if window is not None:
            ax.set_xlim(*window)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Time (minutes)")

        # bottom axis only — a full box would crowd the labels on both sides
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    save(fig, "fig1b_raster")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = label_sessions(pd.read_csv(CSV))

    print(f"{len(df)} rows in {CSV.name}")
    print("\nFlies per treatment and session (Oct/Nov animals appear in "
          "both Oct sessions):")
    print(pd.crosstab(df["treatment"], df["session"]).to_string())

    kept = df[df["n_turns"] >= MIN_TURNS]
    print(f"\n{len(kept)} rows pass n_turns >= {MIN_TURNS}:")
    print(pd.crosstab(kept["treatment"], kept["session"]).to_string())

    if (df["session"] == "unassigned").any():
        print("\nWARNING: some files had no light suffix and are excluded "
              "from every panel:")
        print(df[df["session"] == "unassigned"]["file"].unique())

    print("\nWriting figures:")

    # Figure 2 — locomotor activity
    scatter_figure(df, "n_turns", "Number of turns", "fig2a_turns",
                   require_turns=False)
    scatter_figure(df, "distance_mm", "Distance travelled (mm)",
                   "fig2b_distance", require_turns=False)

    # Figure 3 — locomotor handedness
    scatter_figure(df, "turn_bias", "Turn bias", "fig3a_turn_bias",
                   reference=0.5)
    distribution_figure(df)
    variability_figure(df, "turn_bias", "Fly-to-fly variability in turn bias",
                       "fig3c_variability_turn_bias", corrected=True)

    # Figure 4 — turn sequence and timing
    scatter_figure(df, "switchiness", "Switchiness", "fig4a_switchiness",
                   reference=1.0)
    scatter_figure(df, "clumpiness", "Clumpiness", "fig4b_clumpiness",
                   reference=1.0)

    # Figure 1B — example turn sequences
    raster_figure(chosen=pick_raster_flies(df))

    # Supplementary Figure 1 — control flies across cohorts and sessions
    control_cohorts_figure(df)

    print("\nDone.")