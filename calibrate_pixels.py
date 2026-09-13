"""
Work out pixels-per-mm from the fly trajectories.

Geometry comes from the lab's own laser-cutting schematic, which is drawn
1:1 (verified: every wall segment measures 7.983 mm, sd 0.009):

    dblab-schematics/Ymaze/Laser Cutting Schematics/
    Ymaze_96_MiddleLayer_noMazeNumbers_(sixteenth_in_black).pdf

    chamber centre -> chamber centre   19.13 mm
    maze centre -> chamber centre      11.04 mm
    chamber diameter                    5.30 mm
    corridor width                      3.43 mm
    wall segment                        7.98 mm

Over a two-hour recording a fly reaches the outer wall of all three end
chambers, so the outer extent of its trajectory marks a known distance from
the maze centre: 11.04 + 2.65 = 13.69 mm, less however close the fly's
CENTROID gets to the wall (roughly half a body width).


Edit SETTINGS, then run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ymaze import get_xy, read_file

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")

# From the schematic (mm)
CENTRE_TO_CHAMBER_MM = 11.04
CHAMBER_RADIUS_MM = 2.65
OUTER_MM = CENTRE_TO_CHAMBER_MM + CHAMBER_RADIUS_MM      # 13.69

# How close the fly's CENTROID gets to a wall, in mm. A fly is about 1 mm
# across, so its centroid stops roughly half a body width out. This is the
# one assumed quantity: a 0.15 mm error here shifts the calibration by
# about 1%.
BODY_RADIUS_MM = 0.50

MIN_FRAMES = 5000


# ============================================================
# Geometry from one fly's trajectory
# ============================================================

def maze_centre(x, y, n_iter=8):
    """Centre of the maze, as the centroid of the three chamber centres.

    A plain mean or median is biased when a fly favours one arm, and a
    median is biased anyway because the shape has three-fold rather than
    mirror symmetry. Locating the three chambers first and averaging them
    removes both problems.
    """
    cx, cy = np.mean(x), np.mean(y)

    for _ in range(n_iter):
        dx, dy = x - cx, y - cy
        radius = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)

        outer = radius > 0.6 * np.percentile(radius, 99.7)
        if outer.sum() < 100:
            break

        # The three arms sit 120 deg apart, so tripling the angle collapses
        # them onto one direction; the circular mean recovers the rotation.
        phase = np.angle(np.mean(np.exp(3j * ang[outer]))) / 3

        centres = []
        for a in phase + np.arange(3) * 2 * np.pi / 3:
            member = outer & (np.abs(np.angle(np.exp(1j * (ang - a)))) < np.pi / 3)
            if member.sum() < 30:
                return cx, cy
            centres.append([x[member].mean(), y[member].mean()])

        centres = np.array(centres)
        new_cx, new_cy = centres[:, 0].mean(), centres[:, 1].mean()
        if abs(new_cx - cx) < 0.01 and abs(new_cy - cy) < 0.01:
            return new_cx, new_cy
        cx, cy = new_cx, new_cy

    return cx, cy


def px_per_mm_for_fly(x, y, body_radius=BODY_RADIUS_MM):
    """Pixels per mm from one fly's trajectory, or nan if unusable."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < MIN_FRAMES:
        return np.nan
    x, y = x[valid], y[valid]

    cx, cy = maze_centre(x, y)
    radius = np.hypot(x - cx, y - cy)

    # 99.7th percentile rather than the maximum: one tracking glitch would
    # otherwise set the scale for the whole fly
    outer_px = np.percentile(radius, 99.7)

    scale = outer_px / (OUTER_MM - body_radius)
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def calibrate_file(file_path):
    centroids, _, _ = read_file(file_path)
    out = []
    for fly in range(centroids.shape[0]):
        x, y = get_xy(centroids[fly])
        s = px_per_mm_for_fly(x, y)
        if np.isfinite(s):
            out.append(s)
    return np.array(out)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    files = sorted(DATA_DIR.rglob("*.hdf5"))
    if not files:
        raise SystemExit(f"No HDF5 files found under {DATA_DIR}")

    rows = []
    for file_path in files:
        est = calibrate_file(file_path)
        if not len(est):
            print(f"  {file_path.name}: no usable flies")
            continue
        rows.append({"file": file_path.name, "treatment": file_path.parent.name,
                     "n_flies": len(est), "px_per_mm": np.median(est),
                     "sd": est.std()})
        print(f"  {file_path.parent.name}/{file_path.name}: "
              f"{np.median(est):.3f} px/mm  (n={len(est)}, sd={est.std():.3f})")

    df = pd.DataFrame(rows)
    overall = float(np.median(df["px_per_mm"]))

    print("\n" + "=" * 62)
    print(f"OVERALL:  {overall:.4f} px/mm   ({1/overall:.4f} mm per pixel)")
    print("=" * 62)

    print("\nPer treatment:")
    print(df.groupby("treatment")["px_per_mm"].agg(["median", "std", "count"]))

    spread = (df["px_per_mm"].max() - df["px_per_mm"].min()) / overall * 100
    print(f"\nSpread across recordings: {spread:.1f}% of the median")
    if spread > 3:
        print("  WARNING: calibration differs between recordings by more than")
        print("  3%, so the camera or its height changed between sessions. A")
        print("  single constant is not valid, and a group difference in")
        print("  distance or speed could be an optics artefact. Apply the")
        print("  per-file value instead.")
    else:
        print("  Consistent; a single constant is fine.")

    print(f"\nAssumed centroid-to-wall clearance: {BODY_RADIUS_MM} mm.")
    print("  Shifting that by 0.15 mm moves the calibration about 1%:")
    for b in (0.35, 0.50, 0.65):
        print(f"    {b:.2f} mm -> {overall*(OUTER_MM-BODY_RADIUS_MM)/(OUTER_MM-b):.4f} px/mm")

    print("\nUse in analyze_ymaze.py:")
    print(f"    PIXELS_PER_MM = {overall:.4f}")