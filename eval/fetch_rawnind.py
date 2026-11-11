"""Training crops from RawNIND (Brummer & De Vleeschouwer, CC BY-SA 4.0, UCLouvain Dataverse).

    uv run fetch_rawnind.py 60          every scene up to the 60th
    uv run fetch_rawnind.py 60 3 0      the first of three workers sharing them

Each scene is a tripod series of camera raws, base ISO to the camera's highest. For every scene
that reaches ISO 51,200 and isn't one of the test scenes, this develops the base-ISO frame and one
frame from each high-ISO band the same way (the base frame's white balance, one shared exposure),
lines them up, and saves matching 512 px crops under data/train/ as <scene>_<n>_{clean,iso<N>}.png.
"""

import json
import os
import pathlib
import re
import sys
import urllib.request

import cv2
import numpy as np
import rawpy

DATA = pathlib.Path(__file__).parent / "data"
OUT = DATA / "train"
HOST = "https://dataverse.uclouvain.be/api"
DOI = "doi:10.14428/DVN/DEQCIM"
CROP = 512
CROPS = 12
#: One noisy frame is taken from each band.
BANDS = ((6400, 10000), (12800, 20000), (25600, 40000), (51200, 10**9))
#: Scenes held out: NIND shots of them are the test set, and the authors' own reserve.
HELD_OUT = ("bark", "blombukett", "bruegel", "museel-bobo", "ursulines", "stefantiek", "courtine", "pen-pile")
MAX_SHIFT = 16


def scenes():
    listing = json.load(urllib.request.urlopen(f"{HOST}/datasets/:persistentId/?persistentId={DOI}"))
    found = {}
    for entry in listing["data"]["latestVersion"]["files"]:
        file = entry["dataFile"]
        m = re.match(r"Bayer_(.+?)_(GT_)?ISO(\d+)_sha1=\w+\.arw$", file["filename"])
        if not m:
            continue
        scene = found.setdefault(m.group(1), {"clean": None, "noisy": {}})
        if m.group(2):
            scene["clean"] = scene["clean"] or file["id"]
        else:
            scene["noisy"][int(m.group(3))] = file["id"]
    return {
        name: s
        for name, s in sorted(found.items())
        if s["clean"] and s["noisy"] and max(s["noisy"]) >= 51200 and not name.lower().startswith(HELD_OUT)
    }


def develop(file_id, white_balance=None):
    """Linear sRGB as float in [0, 1], and the white balance used."""
    data = urllib.request.urlopen(f"{HOST}/access/datafile/{file_id}").read()
    path = DATA / f"raw-{os.getpid()}.arw"
    path.write_bytes(data)
    with rawpy.imread(str(path)) as raw:
        white_balance = white_balance or list(raw.camera_whitebalance)
        image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, user_wb=white_balance)
    path.unlink()
    return image.astype(np.float32) / 65535, white_balance


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return cv2.cvtColor((encoded * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def shift_between(clean, noisy):
    """Whole-pixel offset of the noisy frame from the clean one, from a blurred quarter-size copy."""
    small = [cv2.resize(cv2.GaussianBlur(i.mean(axis=2), (0, 0), 6), None, fx=0.25, fy=0.25) for i in (clean, noisy)]
    (dx, dy), _ = cv2.phaseCorrelate(small[0], small[1])
    return round(dx * 4), round(dy * 4)


def corners(clean, rng):
    """Top-left corners on a 512 px grid: the most detailed crops, plus a few picked at random."""
    h, w = clean.shape[:2]
    grid = [
        (y, x)
        for y in range(MAX_SHIFT, h - CROP - MAX_SHIFT, CROP)
        for x in range(MAX_SHIFT, w - CROP - MAX_SHIFT, CROP)
    ]
    gray = clean.mean(axis=2)
    detail = [float(np.abs(cv2.Laplacian(gray[y : y + CROP, x : x + CROP], cv2.CV_32F)).mean()) for y, x in grid]
    ranked = [grid[i] for i in np.argsort(detail)[::-1]]
    rest = ranked[CROPS * 2 // 3 :]
    return ranked[: CROPS * 2 // 3] + [rest[i] for i in rng.choice(len(rest), CROPS // 3, replace=False)]


def main(limit, workers=1, worker=0):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(worker)
    for name, scene in list(scenes().items())[:limit][worker::workers]:
        if (OUT / f"{name}.done").exists():
            continue
        clean, white_balance = develop(scene["clean"])
        # One exposure for the whole scene, so the base frame's highlights sit just under white.
        exposure = 0.9 / max(float(np.percentile(clean, 99.5)), 1e-3)
        luminance = clean.mean(axis=2)
        bright = (luminance > np.percentile(luminance, 60)) & (luminance < np.percentile(luminance, 99))
        picks = corners(clean, rng)
        for n, (y, x) in enumerate(picks):
            cv2.imwrite(str(OUT / f"{name}_{n}_clean.png"), srgb(clean[y : y + CROP, x : x + CROP] * exposure))
        kept = []
        for low, high in BANDS:
            isos = [iso for iso in scene["noisy"] if low <= iso <= high]
            if not isos:
                continue
            iso = int(rng.choice(isos))
            noisy, _ = develop(scene["noisy"][iso], white_balance)
            if noisy.shape != clean.shape:
                continue
            dx, dy = shift_between(clean, noisy)
            if max(abs(dx), abs(dy)) > MAX_SHIFT:
                continue
            # The camera's shutter and ISO steps don't cancel exactly, so the frame's brightness is matched
            # to the base frame's: over its brighter parts only, because in shadows noise clipped at black
            # lifts the average.
            gain = float(clean[bright].mean() / noisy[bright].mean())
            for n, (y, x) in enumerate(picks):
                cv2.imwrite(
                    str(OUT / f"{name}_{n}_iso{iso}.png"),
                    srgb(noisy[y + dy : y + dy + CROP, x + dx : x + dx + CROP] * gain * exposure),
                )
            kept.append((iso, dx, dy, round(gain, 2)))
        (OUT / f"{name}.done").touch()
        print(name, kept, flush=True)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
