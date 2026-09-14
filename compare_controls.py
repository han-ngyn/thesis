"""
Compare the two control cohorts.

The 0.5 uM experiment ran in July 2025 and the 1 uM experiment in
October/November 2025, each with its own controls. This plots control flies
only, so any difference is between cohorts rather than between treatments.

October/November recordings were split into lights-on and lights-off
sessions; July recordings were not. Those two conditions are plotted
separately, because if light condition affects locomotion then part of any
July vs October difference is a lights effect rather than a time effect.

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
N_BOOT = 5000

METRICS = [
    ("turn_bias", "Turn bias", True),
    ("switchiness", "Switchiness", True),
    ("clumpiness", "Clumpiness", True),
    ("tortuosity", "Tortuosity", True),
    ("n_turns", "Number of turns", False),
    ("distance_mm", "Distance (mm)", False),
    ("walking_speed_mm_s", "Walking speed (mm/s)", False),
]

# July | Oct pooled | Oct off | Oct on. The pooled column is included so the
# cost of pooling is visible: if the on and off groups differ, the pooled
# value sits between them and represents neither.
GROUP_ORDER = ["July", "Oct-Nov pooled", "Oct-Nov off", "Oct-Nov on"]
COLORS = ["#35978f", "#9e9e9e", "#4a3f8f", "#dfae3a"]

MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
})


# ============================================================
# Cohort and light condition
# ============================================================

def label_cohorts(df):
    """Tag each fly with its recording period and light condition.

    The light suffix has to be matched carefully: the word "control" appears
    in every filename and contains the letters "on", so a naive search for
    "on" labels every control fly as lights-on.
    """
    df = df.copy()
    df["month"] = df["file"].str.extract(r"^NNYM_(\d{2})-")[0]
    df["cohort"] = np.where(df["month"] == "07", "July", "Oct-Nov")

    low = df["file"].str.lower()
    off = low.str.contains(r"lights?_?off|_off\.", regex=True)
    on = low.str.contains(r"lights?_?on|_on\.", regex=True)
    df["lights"] = np.where(off, "off", np.where(on, "on", "not recorded"))

    df["group"] = np.where(df["cohort"] == "July", "July",
                           "Oct-Nov " + df["lights"])
    return df


def in_group(df, group):
    """Rows belonging to a plotting group.

    "Oct-Nov pooled" is every October/November fly regardless of light
    condition, so it overlaps the on and off groups by design — it is what
    the analysis currently uses as the control baseline.
    """
    if group == "Oct-Nov pooled":
        return df["cohort"] == "Oct-Nov"
    return df["group"] == group


def bootstrap_ci(values, statistic=np.mean, n_boot=N_BOOT, seed=0):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n_boot, len(values)))
    return tuple(np.percentile(statistic(values[draws], axis=1), [2.5, 97.5]))


def get(df, group, column, require_turns, treatment="control"):
    sub = df[(df["treatment"] == treatment) & in_group(df, group)]
    if require_turns:
        sub = sub[sub["n_turns"] >= MIN_TURNS]
    v = sub[column].to_numpy(float)
    return v[np.isfinite(v)]


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = label_cohorts(pd.read_csv(CSV))
    controls = df[df["treatment"] == "control"]

    print("Control flies by cohort and light condition:")
    print(pd.crosstab(controls["cohort"], controls["lights"]).to_string())

    groups = [g for g in GROUP_ORDER
              if len(controls[in_group(controls, g)])]

    # ---- numbers ----
    short = [g.replace("Oct-Nov ", "Oct ") for g in groups]
    print(f"\n{'metric':<21}" + "".join(f"{g:>14}" for g in short)
          + f"{'July v Oct':>12}{'off v on':>11}")
    print("-" * (21 + 14 * len(groups) + 23))

    for column, label, require_turns in METRICS:
        if column not in df:
            continue
        vals = {g: get(df, g, column, require_turns) for g in groups}
        means = {g: (v.mean() if len(v) else np.nan) for g, v in vals.items()}

        pooled = means.get("Oct-Nov pooled", np.nan)
        july = means.get("July", np.nan)
        on, off = means.get("Oct-Nov on", np.nan), means.get("Oct-Nov off", np.nan)

        cohort_gap = (pooled - july) / july * 100 if np.isfinite(july) else np.nan
        light_gap = (off - on) / on * 100 if np.isfinite(on) else np.nan

        print(f"{label:<21}" + "".join(f"{means[g]:>14.3f}" for g in groups)
              + f"{cohort_gap:>11.1f}%{light_gap:>10.1f}%")

    # ---- figure ----
    n = len(METRICS)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 42 * MM, nrow * 40 * MM))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, (column, label, require_turns) in zip(axes, METRICS):
        if column not in df:
            continue
        for i, (g, color) in enumerate(zip(groups, COLORS)):
            v = get(df, g, column, require_turns)
            if not len(v):
                continue
            x = i + rng.uniform(-0.28, 0.28, len(v))
            ax.scatter(x, v, color=color, alpha=0.35, s=3,
                       edgecolors="none", zorder=2)
            lo, hi = bootstrap_ci(v)
            ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]],
                        fmt="none", ecolor="black", capsize=2, lw=1.2, zorder=4)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color="black",
                      lw=1.4, zorder=5)

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("Oct-Nov ", "Oct\n") for g in groups])
        ax.set_xlim(-0.55, len(groups) - 0.45)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Control flies only  (Oct/Nov pooled column overlaps off and on)",
                 fontsize=8)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "control_cohorts.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"\nwrote {out}")