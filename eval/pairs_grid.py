"""Phone-readable grid of models on the night-pair crops, under the real clean frame, same 1:1 windows.

uv run pairs_grid.py out.jpg method[=label] ...    (method "input" is the noisy crop, "clean" the tripod frame)
"""

import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import evaluate

WINDOWS = [
    ("gr3x-025600-labels", 600, 560, 288),
    ("gr3x-102400-labels", 600, 560, 288),
    ("gr3x-102400-labels", 140, 680, 288),
]
LABEL_H = 34


def font():
    """The system's bold sans, so labels stay readable once a phone shrinks the sheet."""
    if shutil.which("fc-match"):
        path = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans:bold"], check=True, capture_output=True, text=True
        ).stdout
        return ImageFont.truetype(path, 22)
    return ImageFont.load_default(22)


def tile(method, crop, x, y, size):
    pairs = evaluate.DATA / "his" / "pairs"
    if method == "input":
        path = pairs / f"{crop}.png"
    elif method == "clean":
        path = pairs / f"{crop}-clean.png"
    else:
        path = evaluate.DATA / "out" / "pairs" / f"{crop}_{method}.png"
    image = cv2.imread(str(path))
    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
    return image[y : y + size, x : x + size]


def row(method, label):
    tiles = np.hstack([tile(method, *w) for w in WINDOWS])
    bar = Image.new("RGB", (tiles.shape[1], LABEL_H), "white")
    ImageDraw.Draw(bar).text((8, 5), label, fill="black", font=FONT)
    bar = np.asarray(bar)
    return np.vstack([bar, tiles, np.full((4, tiles.shape[1], 3), 255, np.uint8)])


FONT = font()

if __name__ == "__main__":
    rows = [row(*(s.split("=", 1) if "=" in s else (s, s))) for s in sys.argv[2:]]
    cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
