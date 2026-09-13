"""
Is a result real, or not?

Each concentration is compared against controls recorded in the same period.
The two experiments ran three months apart with separate control cohorts, and
those cohorts differ (July controls walked ~46% less than October controls),
so pooling them would compare each dose to a baseline drawn half from the
other experiment.

Prints a table of differences with bootstrap 95% confidence intervals, then a
summary of which results clear zero.

Edit SETTINGS, then run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

CSV = Path("/Users/hannguyen/debivort/thesis/ymaze_analysis/outputs/fly_metrics.csv")

MIN_TURNS = 20
N_BOOT = 10_000

# Which controls each dose is compared against, by recording month.
COHORTS = {
    "0.5uM": ("0.5 uM", "July", ("07",)),
    "1uM": ("1 uM", "Oct-Nov", ("10", "11")),
}

METRICS = [
    ("turn_bias", "Turn bias", True),      # True = apply the MIN_TURNS filter
    ("switchiness", "Switchiness", True),
    ("clumpiness", "Clumpiness", True),
    ("tortuosity", "Tortuosity", True),
    ("n_turns", "Turn count", False),
    ("distance_mm", "Distance (mm)", False),
    ("walking_speed_mm_s", "Walking speed (mm/s)", False),
]


# ============================================================
# Helpers
# ============================================================

def subset(df, treatment, column, require_turns, months):
    sub = df[(df["treatment"] == treatment) & (df["month"].isin(months))]
    if require_turns:
        sub = sub[sub["n_turns"] >= MIN_TURNS]
    return sub.dropna(subset=[column])


def corrected_variance(bias, n_turns):
    """Among-fly variance in turn bias with binomial sampling noise removed.

    Observed spread = real spread + noise from estimating each fly's bias
    from a finite number of turns. Without subtracting the second term, a
    group whose flies turn less often looks more variable for that reason.
    """
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


def diff_means(c, d, n_boot=N_BOOT, seed=0):
    """Bootstrap CI on the difference in means.

    Resampling each group and checking whether their intervals overlap is
    not a test — overlapping intervals can still hide a clear difference.
    Resampling the difference itself is the right comparison.
    """
    rng = np.random.default_rng(seed)
    c, d = c[np.isfinite(c)], d[np.isfinite(d)]
    if len(c) < 2 or len(d) < 2:
        return (np.nan,) * 4
    ci = rng.integers(0, len(c), (n_boot, len(c)))
    di = rng.integers(0, len(d), (n_boot, len(d)))
    diffs = d[di].mean(axis=1) - c[ci].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1))


def diff_spreads(c_df, d_df, column, corrected, n_boot=2000, seed=0):
    """Same, on fly-to-fly spread instead of the mean."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        c = c_df.iloc[rng.integers(0, len(c_df), len(c_df))]
        d = d_df.iloc[rng.integers(0, len(d_df), len(d_df))]
        diffs.append(spread_of(d, column, corrected)
                     - spread_of(c, column, corrected))
    diffs = np.array(diffs)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 100:
        return (np.nan,) * 4
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1))


def report(title, rows):
    print(f"\n{title}")
    print(f"{'metric':<21}{'dose':<18}{'control':>10}{'dosed':>10}"
          f"{'diff':>11}{'95% CI':>24}{'p':>8}")
    print("-" * 102)
    hits = []
    for label, dose, era, c_val, d_val, diff, lo, hi, p in rows:
        real = np.isfinite(lo) and (lo > 0 or hi < 0)
        mark = "  *" if real else ""
        print(f"{label:<21}{dose+' ('+era+')':<18}{c_val:>10.3f}{d_val:>10.3f}"
              f"{diff:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>24}{p:>8.3f}{mark}")
        if real:
            hits.append((label, dose, era, diff, lo, hi))
        # a CI that only just includes zero is not evidence of no effect
        elif np.isfinite(lo) and min(abs(lo), abs(hi)) < 0.15 * abs(diff):
            print(f"{'':<21}{'':<18}{'(borderline: interval barely includes zero)':>63}")
    return hits


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    df["month"] = df["file"].str.extract(r"^NNYM_(\d{2})-")[0]

    print(f"{len(df)} flies")
    if "distance_mm" in df:
        low = (df["distance_mm"] < 2000).groupby(df["treatment"]).mean()
        print("Barely moved (<2000 mm): " +
              ", ".join(f"{k} {v*100:.1f}%" for k, v in low.items()))

    mean_rows, var_rows = [], []
    for column, label, require_turns in METRICS:
        if column not in df:
            continue
        corrected = (column == "turn_bias")
        for treatment, (dose, era, months) in COHORTS.items():
            c_df = subset(df, "control", column, require_turns, months)
            d_df = subset(df, treatment, column, require_turns, months)
            c = c_df[column].to_numpy(float)
            d = d_df[column].to_numpy(float)

            mean_rows.append((label, dose, era, np.mean(c), np.mean(d),
                              *diff_means(c, d)))
            var_rows.append((label, dose, era,
                             spread_of(c_df, column, corrected),
                             spread_of(d_df, column, corrected),
                             *diff_spreads(c_df, d_df, column, corrected)))

    hits_mean = report("GROUP MEANS", mean_rows)
    hits_var = report("FLY-TO-FLY VARIABILITY  "
                      "(turn bias corrected for sampling noise; others are plain SD)",
                      var_rows)

    print("\n" + "=" * 102)
    print("REAL RESULTS  (* above: bootstrap 95% CI on the difference excludes zero)\n")
    if not hits_mean and not hits_var:
        print("  none")
    for tag, hits in (("mean", hits_mean), ("variability", hits_var)):
        for label, dose, era, diff, lo, hi in hits:
            direction = "higher" if diff > 0 else "lower"
            print(f"  {label} {tag}: {direction} at {dose} than {era} controls "
                  f"({diff:+.3f} [{lo:.3f}, {hi:.3f}])")
    print("""
  Anything without a * is "no difference detected", which is not the same as
  no difference — a small effect could still be missed at these sample sizes.
  Watch for borderline intervals flagged above; a CI that only just includes
  zero is weak evidence of absence, not strong evidence of nothing.""")