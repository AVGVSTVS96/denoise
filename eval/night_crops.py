"""Phone-sized sample crops for the night raw collection.

usage: python -I night_crops.py OUT.jpg single RAW [x y]
       python -I night_crops.py OUT.jpg pair NOISY CLEAN [x y]
A single shot gives one 100% crop; a pair gives noisy (brightened to match) beside clean, same crop.
"""

import sys

import numpy as np
import rawpy
from PIL import Image

W, H = 640, 640


def linear(path):
    with rawpy.imread(path) as raw:
        return raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16).astype(np.float32) / 65535


def show(lin, gain):
    return (np.clip(lin * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def crop(img, x, y):
    h, w = img.shape[:2]
    x = int(x * w) - W // 2 if x is not None else w // 2 - W // 2
    y = int(y * h) - H // 2 if y is not None else h // 2 - H // 2
    x, y = max(0, min(x, w - W)), max(0, min(y, h - H))
    return img[y : y + H, x : x + W]


out, kind, *rest = sys.argv[1:]
if kind == "single":
    path, *xy = rest
    lin = linear(path)
    x, y = (float(v) for v in xy) if xy else (None, None)
    c = crop(lin, x, y)
    Image.fromarray(show(c, 0.9 / max(np.percentile(lin, 99), 1e-4))).save(out, quality=88)
else:
    noisy_path, clean_path, *xy = rest
    noisy, clean = linear(noisy_path), linear(clean_path)
    x, y = (float(v) for v in xy) if xy else (None, None)
    gain = 0.18 / max(np.median(clean), 1e-4) * 0.6
    match = np.median(clean) / max(np.median(noisy), 1e-6)
    a, b = crop(noisy, x, y), crop(clean, x, y)
    Image.fromarray(np.concatenate([show(a, gain * match), np.full((H, 8, 3), 255, np.uint8), show(b, gain)], 1)).save(out, quality=88)
