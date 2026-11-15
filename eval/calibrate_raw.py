"""A camera's noise levels per ISO, from lens-cap and flat-white raws: step 1 of the faithful model's plan.

    uv run calibrate_raw.py DIR data/noise/gr3x.json

DIR holds dark-<ISO>-<n>.DNG (lens cap on, the shutter speed you'd use at night) and flat-<ISO>-<n>.DNG
(a plain white wall or sky, out of focus, any exposure that keeps it below clipping). For each ISO:
read noise and row banding from the dark frames, then shot noise from how the flat frames' variance
grows with brightness. Levels are in the units train_raw.py works in: black-subtracted, over the white
level. train_raw.py --profile draws its made-up noise from them.
"""

import collections
import json
import pathlib
import re
import sys

import numpy as np

import raw_pairs

BLOCK = 64


def planes(path):
    cfa = raw_pairs.mosaic(path)
    return [cfa[i::2, j::2] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]


def pixel_variance(values):
    """Variance from neighbours in a row: smooth vignetting and a row's banding cancel out."""
    return np.diff(values, axis=-1).var(axis=-1) / 2


def dark_levels(paths):
    """Read noise (variance) and row banding (standard deviation of a row's offset), over every plane."""
    reads, rows = [], []
    for path in paths:
        for plane in planes(path):
            reads.append(float(pixel_variance(plane.reshape(1, -1))[0]))
            # A row mean carries some pixel noise of its own; the rest is banding.
            rows.append(max(float(plane.mean(axis=1).var()) - reads[-1] / plane.shape[1], 0.0))
    return float(np.mean(reads)), float(np.sqrt(np.mean(rows)))


def shot_level(paths, read):
    """Variance minus read noise, over brightness, in small blocks: the slope is the shot noise."""
    slopes = []
    for path in paths:
        for plane in planes(path):
            h, w = plane.shape[0] // BLOCK * BLOCK, plane.shape[1] // BLOCK * BLOCK
            blocks = (
                plane[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).swapaxes(1, 2).reshape(-1, BLOCK * BLOCK)
            )
            mean, var = blocks.mean(axis=1), pixel_variance(blocks)
            usable = (mean > 0.02) & (mean < 0.8)
            slopes.extend(((var - read) / mean)[usable].tolist())
    return float(np.median(slopes))


def main(folder, out):
    frames = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(pathlib.Path(folder).iterdir()):
        m = re.match(r"(dark|flat)-(\d+)-", path.name.lower())
        if m:
            frames[int(m.group(2))][m.group(1)].append(path)
    profile = {"camera": pathlib.Path(out).stem, "iso": {}}
    for iso, kinds in sorted(frames.items()):
        if not kinds["dark"] or not kinds["flat"]:
            print(f"ISO {iso}: needs both dark and flat frames, skipped")
            continue
        read, row = dark_levels(kinds["dark"])
        shot = shot_level(kinds["flat"], read)
        profile["iso"][str(iso)] = {"shot": shot, "read": float(np.sqrt(read)), "row": row}
        print(f"ISO {iso}: shot {shot:.3g}, read {np.sqrt(read):.3g}, row {row:.3g}")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(profile, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
