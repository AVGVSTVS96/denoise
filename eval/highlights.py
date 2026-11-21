"""The bright parts of dark photos, where a model is shown light far past what training showed it.

    uv run highlights.py build                                       cut the tiles once, into data/bench/bright.npz
    uv run highlights.py v4=data/models/raw/faithful-v4/faithful.pth ...    each model's numbers, both ways

denoise() scales a photo by one gain, from its mean, so a lamp in a night photo reaches the network at
tens of times the brightness of any crop in training. train_raw.bright() measures what a model does
there: on flat tiles from 1 to 640 times the usual level, and on real ones, the brightest tile of
1,024 px from each of the night photos in data/night-raw whose brightest tile the photo's gain lifts
furthest, two photos a camera at most. None of them trains.

Every model is scored twice: at the whole photo's gain, which is where the garbage is, and at the
gain denoise() gives a tile, held down for a bright one (train_raw.tile_gain).
"""

import json
import sys

import numpy as np

import bench_raw
import raw_pairs

TILE, PHOTOS, PER_CAMERA = 1024, 8, 2


def build():
    import rawpy

    import train_raw

    found = []
    for row in map(json.loads, (raw_pairs.NIGHT / "index.jsonl").read_text().splitlines()):
        if row["file"].startswith("pairs/"):
            continue
        try:
            mosaic = raw_pairs.mosaic(raw_pairs.NIGHT / row["file"])
        except (OSError, rawpy.LibRawError):
            continue
        if mosaic is None:
            continue
        mean = float(mosaic.mean())
        gain = min(max(train_raw.LEVEL / mean, 1), train_raw.MAX_GAIN)
        h, w = (side // TILE for side in mosaic.shape)
        means = mosaic[: h * TILE, : w * TILE].reshape(h, TILE, w, TILE).mean((1, 3))
        y, x = np.unravel_index(means.argmax(), means.shape)
        tile = mosaic[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE].astype(np.float16)
        found.append((float(means.max()) * gain, row["camera"], row["file"], mean, tile))
        print(row["file"], round(mean, 4), round(found[-1][0], 2), flush=True)
    kept = []
    for lift, camera, file, mean, tile in sorted(found, key=lambda photo: -photo[0]):
        if sum(camera == other for _, other, *_ in kept) < PER_CAMERA and len(kept) < PHOTOS:
            kept.append((lift, camera, file, mean, tile))
    _, _, files, means, tiles = zip(*kept)
    bench_raw.save("bright", tiles=np.stack(tiles), means=means, files=files)
    print([(file, round(lift, 2)) for lift, _, file, _, _ in kept])


def main(specs):
    import train_raw

    for name, weights in (spec.split("=") for spec in specs):
        model = train_raw.trained(weights)
        for label, tiled in (("the photo's gain", False), ("denoise()'s tile gain", True)):
            found = train_raw.bright(train_raw.cleaner(model, tiled))
            flats = "  ".join(f"{level}x {error:.3f}" for level, error in zip(train_raw.FLATS, found["flats"]))
            print(f"{name:10} {label:22} tiles {found.get('tiles', float('nan')):.4f}  flats {flats}", flush=True)


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else main(sys.argv[1:])
