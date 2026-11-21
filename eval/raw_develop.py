"""A raw photo through the joint model, developed the way the app develops it.

    uv run raw_develop.py MODEL.pth PHOTO.dng OUT.png             the picture
    uv run raw_develop.py MODEL.pth PHOTO.dng OUT.png APP.ppm     and how far the app's picture is from it

The reference for the app's raw path (src/dng.ts, src/raw.ts), built from other parts: rawpy reads the
mosaic and the camera's colour, exiv2 the DNG's crop and exposure, blacks.py's offsets come off as in
raw_denoise.py, and the model runs in full precision. A photo none of whose tiles ship_raw.BRIGHTEST
touches goes through train_raw.denoise() itself. One with lights in it goes tile by tile as the app
does: the same tiles, each bright one at its own gain, a tile that differs from a neighbour faded
into it over the middle of their overlap.

What is shared with the app is the recipe after the model: white balance with highlights clipped
where the sensor clipped, the camera's matrix to sRGB, the DNG's baseline exposure, a filmic tone
curve (Narkowicz's ACES fit, twice the exposure, scaled so the sensor's clip lands on white) and the
sRGB curve. raw_check.ts makes APP.ppm, in the sensor's orientation, so the comparison is for photos
held level.
"""

import pathlib
import subprocess
import sys

import cv2
import numpy as np
import rawpy
import torch

import bench_raw
import blacks
import raw_pairs
import ship_raw
import train_raw

EXPOSURE = 2
STEPS = 65535
TILE, OVERLAP = 1024, 64
BLEND = OVERLAP // 2
TURNS = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}


def tag(path, name, missing):
    """A DNG tag's numbers, from whichever of its images carries it."""
    listed = subprocess.run(["exiv2", "-Pv", "-g", name, str(path)], check=False, capture_output=True, text=True)
    values = listed.stdout.splitlines()[0].split() if listed.stdout.strip() else []
    ratio = lambda value: float(value.split("/")[0]) / float(value.split("/")[1]) if "/" in value else float(value)
    return [ratio(value) for value in values] or missing


def tone_curve(exposure):
    filmic = lambda x: x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
    lift = 2.0**exposure
    toned = np.minimum(1, filmic(EXPOSURE * lift * np.arange(STEPS + 1) / STEPS) / filmic(EXPOSURE * min(1, lift)))
    encoded = np.where(toned <= 0.0031308, 12.92 * toned, 1.055 * toned ** (1 / 2.4) - 0.055)
    return np.floor(255 * encoded + 0.5).astype(np.uint8)


def look(path):
    """A photo's white balance, lowest at 1, its matrix from camera colour to sRGB, and its tone curve."""
    balance, matrix = bench_raw.colour(path)
    return balance / balance.min(), matrix.astype(np.float64), tone_curve(tag(path, "BaselineExposure", [0.0])[0])


def developed(picture, balance, matrix, curve):
    """Camera RGB (3, H, W) as the sRGB picture (H, W, 3, 8 bits) the app shows, before it is turned upright."""
    balanced = np.minimum(picture * balance[:, None, None], 1)
    linear = np.einsum("ij,jhw->hwi", matrix, balanced.astype(np.float64))
    return curve[np.floor(np.clip(linear, 0, 1) * STEPS + 0.5).astype(np.int32)]


def tile_by_tile(model, mosaic, colour):
    """The app's way through a photo: the developed picture, or None when every tile runs as denoise() runs it."""
    gain = float(np.clip(train_raw.LEVEL / max(float(mosaic.mean()), 1e-6), 1, train_raw.MAX_GAIN))
    block = ship_raw.BRIGHTEST["block"] if ship_raw.BRIGHTEST else TILE
    blocks = ship_raw.levels(mosaic * gain, block)
    padded = np.pad(mosaic * gain, OVERLAP, "reflect")
    starts = [sorted({*range(0, side - TILE, TILE - 2 * OVERLAP), side - TILE}) for side in padded.shape]
    inside = lambda start, side: slice(max(0, start - OVERLAP) // block, (min(side, start - OVERLAP + TILE) - 1) // block + 1)
    own = [
        [ship_raw.own_gain(blocks[inside(y, mosaic.shape[0]), inside(x, mosaic.shape[1])]) for x in starts[1]]
        for y in starts[0]
    ]
    if all(scale == 1 for row in own for scale in row):
        return None
    canvas = np.zeros((*mosaic.shape, 3), np.float64)
    ends = [[*along[1:], side] for along, side in zip(starts, mosaic.shape)]

    def faded(first, start, low, high, blended):
        at = np.arange(low, high)
        ramp = np.clip((at + 0.5 - start + BLEND) / (2 * BLEND), 0, 1) if blended else (at >= start).astype(np.float64)
        return np.ones(high - low) if first else ramp

    for r, y in enumerate(starts[0]):
        y0, y1 = (y - BLEND if r else 0), min(ends[0][r] + BLEND, mosaic.shape[0])
        for c, x in enumerate(starts[1]):
            x0, x1 = (x - BLEND if c else 0), min(ends[1][c] + BLEND, mosaic.shape[1])
            piece = torch.from_numpy(np.ascontiguousarray(padded[y : y + TILE, x : x + TILE] * own[r][c])).cuda()[None]
            result = train_raw.made(model, model(train_raw.fed(model, piece)).float())[0].cpu().numpy()
            kept = result[:, y0 - y + OVERLAP : y1 - y + OVERLAP, x0 - x + OVERLAP : x1 - x + OVERLAP]
            shown = np.outer(
                faded(r == 0, y, y0, y1, r > 0 and own[r - 1][c] != own[r][c]),
                faded(c == 0, x, x0, x1, c > 0 and own[r][c - 1] != own[r][c]),
            )
            shown = np.floor(255 * shown + 0.5)[..., None] / 255
            new = developed(kept / (gain * own[r][c]), *colour)
            canvas[y0:y1, x0:x1] = np.floor(new * shown + canvas[y0:y1, x0:x1] * (1 - shown) + 0.5)
    return canvas.astype(np.uint8)


def main(weights, photo, out, app=None):
    photo = pathlib.Path(photo)
    model = ship_raw.loaded(pathlib.Path(weights))[0].model.cuda()
    with rawpy.imread(str(photo)) as raw:
        colours, pattern, flip = raw.color_desc.decode(), raw.raw_pattern, raw.sizes.flip
    red = next((i, j) for i in range(2) for j in range(2) if colours[pattern[i, j]] == "R")
    mosaic = raw_pairs.mosaic(photo)
    mosaic = mosaic - np.tile(blacks.of(photo).reshape(2, 2), (mosaic.shape[0] // 2, mosaic.shape[1] // 2))
    colour = look(photo)
    whole = tile_by_tile(model, mosaic, colour)
    if whole is None:
        whole = developed(train_raw.denoise(model, mosaic), *colour)
    x, y = (max(0, round(at) - shift) for at, shift in zip(tag(photo, "DefaultCropOrigin", [0, 0]), red[::-1]))
    width, height = (round(side) for side in tag(photo, "DefaultCropSize", mosaic.shape[::-1]))
    level = whole[y : y + height, x : x + width]
    upright = cv2.rotate(level, TURNS[flip]) if flip in TURNS else level
    cv2.imwrite(out, cv2.cvtColor(upright, cv2.COLOR_RGB2BGR))
    if app:
        theirs = cv2.cvtColor(cv2.imread(app), cv2.COLOR_BGR2RGB).astype(np.int32)
        apart = np.abs(theirs - level)
        psnr = 10 * np.log10(255**2 / (apart.astype(np.float64) ** 2).mean())
        print(
            f"{photo.name}: {level.shape[1]}x{level.shape[0]}, the app against this: PSNR {psnr:.2f} dB, "
            f"mean {apart.mean():.4f} and worst {apart.max()} of 255, "
            f"{(apart == 0).mean():.2%} of values the same, {(apart <= 1).mean():.3%} within 1, "
            f"{(apart > 4).mean():.4%} more than 4 apart"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main(*sys.argv[1:])
