"""Real dark frames as fields of unit strength, for train_raw.py --darks to use as read noise: uv run dark_bank.py

data/darks/<camera>/dark-<ISO>-<n>.<raw> are lens-cap frames, named as calibrate_raw.py wants them, so a
camera's calibration shots serve both. A dark frame is read noise as the camera makes it: banding,
pixels that sit high, long tails and the black offset the file's black level misses. Each colour
plane is centred and divided by its spread, measured without the few pixels that sit far out, and the
frame's middle goes into data/noise/darks.npy, float16 (frames, 2048, 2048). darks.json keeps each
frame's camera and ISO and, per colour plane, the offset in units of that spread and the spread itself
in normalised raw units.
"""

import json
import re

import numpy as np

import raw_pairs

FRAMES = raw_pairs.DATA / "darks"
OUT = raw_pairs.DATA / "noise"
SIDE = 2048
SITES = ((0, 0), (0, 1), (1, 0), (1, 1))


def unit(cfa):
    """A dark mosaic with every colour plane centred and at unit strength, and each plane's offset and spread."""
    field, lifts, spreads = np.empty_like(cfa), [], []
    for i, j in SITES:
        plane = cfa[i::2, j::2]
        middle = np.median(plane)
        rough = 1.4826 * np.median(np.abs(plane - middle))
        near = plane[np.abs(plane - middle) < 6 * rough]
        field[i::2, j::2] = (plane - near.mean()) / near.std()
        lifts.append(float(near.mean() / near.std()))
        spreads.append(float(near.std()))
    return field, lifts, spreads


def main():
    fields, frames = [], []
    for path in sorted(FRAMES.glob("*/dark-*")):
        cfa = raw_pairs.mosaic(path)
        iso = re.match(r"dark-(\d+)-", path.name)
        if cfa is None or not iso or min(cfa.shape) < SIDE:
            continue
        y, x = ((side - SIDE) // 4 * 2 for side in cfa.shape)
        field, lifts, spreads = unit(cfa[y : y + SIDE, x : x + SIDE])
        fields.append(field.clip(-60, 60).astype(np.float16))
        frames.append({"camera": path.parent.name, "iso": int(iso.group(1)), "offset": lifts, "read": spreads})
        print(path.parent.name, path.name, "offset", np.round(lifts, 3), "read", np.round(spreads, 5), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "darks.npy", np.stack(fields))
    (OUT / "darks.json").write_text(json.dumps(frames, indent=1))


if __name__ == "__main__":
    main()
