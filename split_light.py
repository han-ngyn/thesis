"""
Does the 1 uM result survive being split by light condition?

The October/November experiment recorded flies with lights on and lights off.
Both the 1 uM flies and their controls are currently pooled across the two,
and light matters a lot: control flies made 767 turns on average with lights
off versus 448 with lights on. A difference between pooled 1 uM and pooled
control could therefore reflect the drug, or an imbalance in how the two
groups split across light conditions.

This compares 1 uM against control within each light condition separately,
alongside the pooled comparison for reference, and plots the same thing.

Edit SETTINGS, then run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")
FIGURE_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/figures")

MIN_TURNS = 20
N_BOOT = 10_000
N_BOOT_SPREAD = 2000

METRICS = [
    ("turn_bias", "Turn bias", True),
    ("switchiness", "Switchiness", True),
    ("clumpiness", "Clumpiness", True),
    ("tortuosity", "Tortuosity", True),
    ("n_turns", "Number of turns", False),
    ("distance_mm", "Distance (mm)", False),
    ("walking_speed_mm_s", "Walking speed (mm/s)", False),
]

# Control and 1 uM within each light condition. Positions leave a gap between
# the two light conditions so the dose comparison reads within each pair.
PANEL_GROUPS = [
    ("control", "on", "Ctl", 0.0, "#35978f"),
    ("1uM", "on", "1 \u00b5M", 0.8, "#bf812d"),
    ("control", "off", "Ctl", 2.0, "#35978f"),
    ("1uM", "off", "1 \u00b5M", 2.8, "#bf812d"),
]

# Light condition is annotated once under each pair rather than repeated on
# every tick, which is what makes the labels collide at this panel size.
LIGHT_LABELS = [(0.4, "lights on"), (2.4, "lights off")]

MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


# ============================================================
# Cohort and light condition
# ============================================================

def label_lights(df):
    """Tag each fly with its recording period and light condition.

    The light suffix has to be matched carefully: every filename contains the
    word "control", which contains the letters "on", so a plain search for
    "on" would label every control fly as lights-on.
    """
    df = df.copy()
    df["month"] = df["file"].str.extract(r"^NNYM_(\d{2})-")[0]
    df["cohort"] = np.where(df["month"] == "07", "July", "Oct-Nov")

    low = df["file"].str.lower()
    off = low.str.contains(r"lights?_?off|_off\.", regex=True)
    on = low.str.contains(r"lights?_?on|_on\.", regex=True)
    df["lights"] = np.where(off, "off", np.where(on, "on", "not recorded"))
    return df


def pick(df, treatment, lights, column, require_turns):
    sub = df[(df["treatment"] == treatment) & (df["cohort"] == "Oct-Nov")]
    if lights != "pooled":
        sub = sub[sub["lights"] == lights]
    if require_turns:
        sub = sub[sub["n_turns"] >= MIN_TURNS]
    return sub.dropna(subset=[column])


# ============================================================
# Statistics
# ============================================================

def corrected_variance(bias, n_turns):
    """Among-fly variance in turn bias with binomial sampling noise removed."""
    keep = np.isfinite(bias) & np.isfinite(n_turns) & (n_turns > 0)
    bias, n_turns = bias[keep], n_turns[keep]
    if len(bias) < 2:
        return np.nan
    return np.var(bias, ddof=1) - np.mean(bias * (1 - bias) / n_turns)


def spread_of(sub, column, corrected):
    v = sub[column].to_numpy(float)
    if corrected:
        return np.sqrt(max(corrected_variance(
            v, sub["n_turns"].to_numpy(float)), 0.0))
    v = v[np.isfinite(v)]
    return v.std(ddof=1) if len(v) > 1 else np.nan


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    return tuple(np.percentile(values[draws].mean(axis=1), [2.5, 97.5]))


def diff_means(c, d, n_boot=N_BOOT, seed=0):
    """Bootstrap CI on the difference in means.

    Resampling each group and checking whether their intervals overlap is not
    a test; resampling the difference itself is the right comparison.
    """
    rng = np.random.default_rng(seed)
    c, d = c[np.isfinite(c)], d[np.isfinite(d)]
    if len(c) < 2 or len(d) < 2:
        return (np.nan,) * 3
    ci = rng.integers(0, len(c), (n_boot, len(c)))
    di = rng.integers(0, len(d), (n_boot, len(d)))
    diffs = d[di].mean(axis=1) - c[ci].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def diff_spreads(c_df, d_df, column, corrected,
                 n_boot=N_BOOT_SPREAD, seed=0):
    """Same, on fly-to-fly spread instead of the mean."""
    rng = np.random.default_rng(seed)
    if len(c_df) < 3 or len(d_df) < 3:
        return (np.nan,) * 3
    diffs = []
    for _ in range(n_boot):
        c = c_df.iloc[rng.integers(0, len(c_df), len(c_df))]
        d = d_df.iloc[rng.integers(0, len(d_df), len(d_df))]
        diffs.append(spread_of(d, column, corrected)
                     - spread_of(c, column, corrected))
    diffs = np.array(diffs)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 100:
        return (np.nan,) * 3
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def clears_zero(lo, hi):
    return bool(np.isfinite(lo) and (lo > 0 or hi < 0))


# ============================================================
# Table
# ============================================================

def run_table(df, kind):
    """kind is 'mean' or 'variability'."""
    print(f"\n{'=' * 96}")
    print(f"1 uM vs Oct/Nov control - {kind.upper()}")
    print("=" * 96)
    print(f"{'metric':<21}{'lights':<9}{'n ctl':>7}{'n 1uM':>7}"
          f"{'diff':>12}{'95% CI':>26}   verdict")
    print("-" * 96)

    for column, label, require_turns in METRICS:
        if column not in df:
            continue
        corrected = (column == "turn_bias")

        for lights in ("pooled", "on", "off"):
            c_df = pick(df, "control", lights, column, require_turns)
            d_df = pick(df, "1uM", lights, column, require_turns)

            if kind == "mean":
                diff, lo, hi = diff_means(c_df[column].to_numpy(float),
                                          d_df[column].to_numpy(float))
            else:
                diff, lo, hi = diff_spreads(c_df, d_df, column, corrected)

            mark = "*" if clears_zero(lo, hi) else " "
            verdict = ("DIFFERS" if mark == "*"
                       else "no difference detected" if np.isfinite(lo)
                       else "too few flies")
            print(f"{label if lights == 'pooled' else '':<21}{lights:<9}"
                  f"{len(c_df):>7}{len(d_df):>7}{diff:>12.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>26} {mark} {verdict}")
        print()


# ============================================================
# Figure
# ============================================================

def make_figure(df):
    n = len(METRICS)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 44 * MM, nrow * 42 * MM))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, (column, label, require_turns) in zip(axes, METRICS):
        if column not in df:
            continue
        for treatment, lights, tick, pos, color in PANEL_GROUPS:
            sub = pick(df, treatment, lights, column, require_turns)
            v = sub[column].to_numpy(float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue

            x = pos + rng.uniform(-0.26, 0.26, len(v))
            ax.scatter(x, v, color=color, alpha=0.35, s=3,
                       edgecolors="none", zorder=2)
            lo, hi = bootstrap_ci(v)
            ax.errorbar(pos, v.mean(),
                        yerr=[[v.mean() - lo], [hi - v.mean()]],
                        fmt="none", ecolor="black", capsize=2, lw=1.2, zorder=4)
            ax.hlines(v.mean(), pos - 0.24, pos + 0.24, color="black",
                      lw=1.4, zorder=5)

        ax.set_xticks([g[3] for g in PANEL_GROUPS])
        ax.set_xticklabels([g[2] for g in PANEL_GROUPS])
        ax.set_xlim(-0.5, 3.3)
        for pos, text in LIGHT_LABELS:
            ax.text(pos, -0.19, text, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("October/November experiment, split by light condition",
                 fontsize=8)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "lights_split.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = label_lights(pd.read_csv(CSV))

    oct_nov = df[df["cohort"] == "Oct-Nov"]
    print("October/November flies by treatment and light condition:")
    print(pd.crosstab(oct_nov["treatment"], oct_nov["lights"]).to_string())

    run_table(df, "mean")
    run_table(df, "variability")

    print()
    make_figure(df)