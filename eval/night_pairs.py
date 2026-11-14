"""1024 crops of a noisy raw and its clean twin, same window and brightness, as model inputs and ground truth.

usage: python -I night_pairs.py OUT_DIR NAME NOISY CLEAN x y
Writes OUT_DIR/NAME.png (noisy) and OUT_DIR/NAME-clean.png; x and y are the window's centre as fractions of the frame.
"""

import sys
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

S = 1024


def linear(path):
    with rawpy.imread(path) as raw:
        return raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16).astype(np.float32) / 65535


def window(img, x, y):
    h, w = img.shape[:2]
    left = min(max(int(x * w) - S // 2, 0), w - S)
    top = min(max(int(y * h) - S // 2, 0), h - S)
    return img[top : top + S, left : left + S]


def srgb(lin, gain):
    return Image.fromarray((np.clip(lin * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))


out, name, noisy_path, clean_path, x, y = sys.argv[1:]
noisy, clean = linear(noisy_path), linear(clean_path)
gain = 0.18 / max(np.median(clean), 1e-4) * 0.6
match = np.median(clean) / max(np.median(noisy), 1e-6)
Path(out).mkdir(parents=True, exist_ok=True)
srgb(window(noisy, float(x), float(y)), gain * match).save(Path(out) / f"{name}.png")
srgb(window(clean, float(x), float(y)), gain).save(Path(out) / f"{name}-clean.png")
