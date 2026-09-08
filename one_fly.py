"""Show the metrics for one fly. Edit SETTINGS, then run."""

from pathlib import Path

from analyze_ymaze import (
    detect_orientation,
    movement_metrics,
    read_file,
    turn_metrics,
)

# ---- SETTINGS ----
DATA_DIR = Path("/Users/hannguyen/debivort/thesis/ymaze_rawdata")

FILE = None      # None = first file found, or e.g. "1uM/NNYM_....hdf5"
FLY = 0
# ------------------


if FILE is None:
    matches = sorted(DATA_DIR.rglob("*.hdf5"))
    if not matches:
        raise SystemExit(f"No .hdf5 files found under {DATA_DIR}")
    file_path = matches[0]
else:
    file_path = DATA_DIR / FILE
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

centroids, turns, time = read_file(file_path)

if not 0 <= FLY < centroids.shape[0]:
    raise SystemExit(f"FLY must be 0 to {centroids.shape[0] - 1}")

orientation = detect_orientation(centroids[FLY])
results = {
    **turn_metrics(turns[FLY], time, orientation),
    **movement_metrics(centroids[FLY], turns[FLY], time),
}

print(f"File:        {file_path.name}")
print(f"Treatment:   {file_path.parent.name}")
print(f"Fly:         {FLY} of {centroids.shape[0]}")
print(f"Orientation: {orientation}")
print()
for name, value in results.items():
    print(f"  {name}: {value}")