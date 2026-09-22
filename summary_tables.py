"""
Summary tables for the Y-maze analysis.

Reads fly_metrics.csv (from analyze_ymaze.py), runs a few quality checks,
and writes four CSVs:

    group_summary.csv           n, mean and fly-to-fly spread for every
                                treatment in every session

    dose_comparisons.csv        each dose against controls from its own
                                cohort and session: means, change, CI

    variability_comparisons.csv the same comparisons, on fly-to-fly spread

    control_benchmark.csv       control flies only: how much behaviour
                                moves between sessions and cohorts with no
                                drug involved
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ymaze import label_sessions, spread_of

# ============================================================
# SETTINGS
# ============================================================

CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")
OUT_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs")

MIN_TURNS = 20
N_BOOT = 10_000          # for differences in means
N_BOOT_SPREAD = 5000     # for differences in spread; each draw is costlier

# A fly that covers less than this in two hours did not explore the maze.
# Reported per recording, because a whole file of them is a tracking
# failure rather than a biological result.
STILL_MM = 2000

# (column, label, apply the MIN_TURNS filter, decimals to round to)
METRICS = [
    ("n_turns", "Number of turns", False, 1),
    ("distance_mm", "Distance (mm)", False, 0),
    ("turn_bias", "Turn bias", True, 4),
    ("switchiness", "Switchiness", True, 4),
    ("clumpiness", "Clumpiness", True, 3),
    # ("tortuosity", "Tortuosity", True, 3),
    # ("walking_speed_mm_s", "Walking speed (mm/s)", False, 3),
]

# How treatments and sessions are written in the output files.
TREATMENT_NAME = {"control": "Control", "0.5uM": "0.5 uM", "1uM": "1 uM"}
SESSION_NAME = {
    "July": "July (lights off)",
    "Oct-on": "Oct/Nov (lights on)",
    "Oct-off": "Oct/Nov (lights off)",
}

# Each dose against controls recorded in the same cohort and session, so the
# two groups are always different flies.
# (label, session of both groups, dosed treatment)
DOSE_COMPARISONS = [
    ("0.5 uM vs control - July", "July", "0.5uM"),
    ("1 uM vs control - Oct/Nov lights on", "Oct-on", "1uM"),
    ("1 uM vs control - Oct/Nov lights off", "Oct-off", "1uM"),
]

# Control flies only. The two Oct sessions are the SAME flies recorded
# twice, which is why no confidence intervals are reported here.
# (label, first session, second session)
BENCHMARKS = [
    ("Oct/Nov control: lights on to lights off", "Oct-on", "Oct-off"),
    ("Control: July to Oct/Nov lights off", "July", "Oct-off"),
]


def subset(df, treatment, session, column, require_turns):
    sub = df[(df["treatment"] == treatment) & (df["session"] == session)]
    if require_turns:
        sub = sub[sub["n_turns"] >= MIN_TURNS]
    return sub.dropna(subset=[column])


# ============================================================
# Bootstrap
# ============================================================

def diff_means(a, b, n_boot=N_BOOT, seed=0):
    """Observed difference b - a, with a percentile bootstrap 95% CI.

    Resampling each group separately and checking whether their intervals
    overlap is not a test: two overlapping intervals can still hide a clear
    difference. Resampling the difference itself answers the question
    directly.
    """
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, len(a), (n_boot, len(a)))
    bi = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = b[bi].mean(axis=1) - a[ai].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def diff_spreads(a_df, b_df, column, corrected,
                 n_boot=N_BOOT_SPREAD, seed=0):
    """Same, on fly-to-fly spread instead of the mean.

    Whole flies are resampled, not values, so that the turn counts used by
    the binomial correction travel with the biases they belong to.
    """
    if len(a_df) < 3 or len(b_df) < 3:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        a = a_df.iloc[rng.integers(0, len(a_df), len(a_df))]
        b = b_df.iloc[rng.integers(0, len(b_df), len(b_df))]
        diffs.append(spread_of(b, column, corrected)
                     - spread_of(a, column, corrected))
    diffs = np.array(diffs)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 100:
        return (np.nan,) * 3
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    observed = (spread_of(b_df, column, corrected)
                - spread_of(a_df, column, corrected))
    return float(observed), float(lo), float(hi)


def ci_text(lo, hi, decimals):
    """A CI as one readable cell, e.g. [100.4 to 164.5]."""
    if not np.isfinite(lo):
        return ""
    return f"[{lo:.{decimals}f} to {hi:.{decimals}f}]"


def verdict(lo, hi):
    if not np.isfinite(lo):
        return "too few flies"
    return "differs" if (lo > 0 or hi < 0) else "no difference detected"


# ============================================================
# Quality checks
# ============================================================

def quality_checks(df):
    """Print anything that would make a group's numbers untrustworthy.

    Flies that never moved are reported per recording rather than per
    treatment: a few in every file is normal, but a whole file of them is a
    tracking failure and that recording should be excluded before the
    tables below mean anything.
    """
    print("\n" + "=" * 100)
    print("QUALITY CHECKS")
    print("=" * 100)

    print("\nFlies per treatment and session (Oct/Nov animals appear in "
          "both Oct sessions):")
    print(pd.crosstab(df["treatment"], df["session"]).to_string())

    if (df["session"] == "unassigned").any():
        print("\nWARNING: files with no light suffix, excluded from every "
              "table:")
        for name in df[df["session"] == "unassigned"]["file"].unique():
            print(f"    {name}")

    if "distance_mm" in df:
        still = (df.assign(still=df["distance_mm"] < STILL_MM)
                   .groupby(["treatment", "file"])["still"]
                   .agg(["sum", "count"]))
        still["fraction"] = (still["sum"] / still["count"]).round(3)
        bad = still[still["fraction"] > 0.25]
        print(f"\nFlies covering less than {STILL_MM} mm, by recording:")
        print(still.to_string())
        if len(bad):
            print("\nWARNING: these recordings are mostly motionless flies "
                  "and look like tracking failures:")
            for (treatment, name), row in bad.iterrows():
                print(f"    {treatment}/{name}: "
                      f"{int(row['sum'])} of {int(row['count'])}")

    orientation_failed = df["orientation"].isin(["ambiguous", "unknown"])
    if orientation_failed.any():
        print("\nFlies whose maze orientation could not be determined "
              "(no turn bias for these):")
        print(df[orientation_failed].groupby("treatment").size().to_string())


# ============================================================
# Tables
# ============================================================

def group_table(df):
    """n, mean and spread for every treatment in every session."""
    rows = []
    for column, label, require_turns, dec in METRICS:
        if column not in df:
            continue
        corrected = (column == "turn_bias")
        for session in ("July", "Oct-on", "Oct-off"):
            for treatment in ("control", "0.5uM", "1uM"):
                sub = subset(df, treatment, session, column, require_turns)
                if not len(sub):
                    continue
                v = sub[column].to_numpy(float)
                rows.append({
                    "Metric": label,
                    "Session": SESSION_NAME[session],
                    "Treatment": TREATMENT_NAME[treatment],
                    "Flies (n)": len(sub),
                    "Mean": round(v.mean(), dec),
                    "Fly-to-fly spread": round(spread_of(sub, column, corrected), dec + 1),
                    "Spread type": "corrected SD" if corrected else "SD",
                })
    return pd.DataFrame(rows)


def dose_tables(df):
    """Means and spreads for each dose against its own control."""
    mean_rows, spread_rows = [], []

    for column, label, require_turns, dec in METRICS:
        if column not in df:
            continue
        corrected = (column == "turn_bias")

        for name, session, dose in DOSE_COMPARISONS:
            ctl_df = subset(df, "control", session, column, require_turns)
            dose_df = subset(df, dose, session, column, require_turns)
            if len(ctl_df) < 3 or len(dose_df) < 3:
                continue

            ctl = ctl_df[column].to_numpy(float)
            dosed = dose_df[column].to_numpy(float)

            diff, lo, hi = diff_means(ctl, dosed)
            mean_rows.append({
                "Metric": label,
                "Comparison": name,
                "Control flies (n)": len(ctl_df),
                "Dosed flies (n)": len(dose_df),
                "Control mean": round(ctl.mean(), dec),
                "Dosed mean": round(dosed.mean(), dec),
                "Difference": round(diff, dec),
                "Change (%)": round(diff / ctl.mean() * 100, 1) if ctl.mean() else np.nan,
                "95% CI of difference": ci_text(lo, hi, dec),
                "Result": verdict(lo, hi),
            })

            s_ctl = spread_of(ctl_df, column, corrected)
            s_dose = spread_of(dose_df, column, corrected)
            s_diff, s_lo, s_hi = diff_spreads(ctl_df, dose_df, column, corrected)
            spread_rows.append({
                "Metric": label,
                "Comparison": name,
                "Spread type": "corrected SD" if corrected else "SD",
                "Control spread": round(s_ctl, dec + 1),
                "Dosed spread": round(s_dose, dec + 1),
                "Difference": round(s_diff, dec + 1),
                "Change (%)": round(s_diff / s_ctl * 100, 1) if s_ctl else np.nan,
                "95% CI of difference": ci_text(s_lo, s_hi, dec + 1),
                "Result": verdict(s_lo, s_hi),
            })

    return pd.DataFrame(mean_rows), pd.DataFrame(spread_rows)


def benchmark_table(df):
    """Control flies only: how much behaviour moves with no drug involved.

    No confidence intervals: the two October/November sessions are the same
    flies recorded twice, and a CI that assumes independent groups would be
    wrong. Means and percentage change are unaffected.
    """
    rows = []
    for column, label, require_turns, dec in METRICS:
        if column not in df:
            continue
        for name, first, second in BENCHMARKS:
            a = subset(df, "control", first, column, require_turns)[column].to_numpy(float)
            b = subset(df, "control", second, column, require_turns)[column].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 3 or len(b) < 3:
                continue
            rows.append({
                "Metric": label,
                "Comparison": name,
                "Flies (n), from": len(a),
                "Flies (n), to": len(b),
                "Mean, from": round(a.mean(), dec),
                "Mean, to": round(b.mean(), dec),
                "Change (%)": round((b.mean() - a.mean()) / a.mean() * 100, 1),
            })
    return pd.DataFrame(rows)


# ============================================================
# Run
# ============================================================

def show(title, table):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(table.to_string(index=False))


if __name__ == "__main__":
    df = label_sessions(pd.read_csv(CSV))
    print(f"{len(df)} rows in {CSV.name}")

    quality_checks(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = group_table(df)
    means, spreads = dose_tables(df)
    benchmark = benchmark_table(df)

    groups.to_csv(OUT_DIR / "group_summary.csv", index=False)
    means.to_csv(OUT_DIR / "dose_comparisons.csv", index=False)
    spreads.to_csv(OUT_DIR / "variability_comparisons.csv", index=False)
    benchmark.to_csv(OUT_DIR / "control_benchmark.csv", index=False)

    show("GROUPS", groups)
    show("DOSE COMPARISONS - GROUP MEANS", means)
    show("DOSE COMPARISONS - FLY-TO-FLY SPREAD", spreads)
    show("BENCHMARK - control flies only, no drug involved", benchmark)

    for name in ("group_summary.csv", "dose_comparisons.csv",
                 "variability_comparisons.csv", "control_benchmark.csv"):
        print(f"Wrote {OUT_DIR / name}")

    print("""
  "no difference detected" is not the same as no difference: a small effect
  could still be missed at these sample sizes.""")