"""One phone-readable grid of every model on his photo: a row per model, the same 1:1 windows of his crops.

    uv run his_grid.py out.jpg method[=label] ...

Each output is warped onto the noisy crop first (frontier.align), so a model that shifts or reframes
shows the same pixels as the others.
"""

import sys

import cv2
import numpy as np

import evaluate
import frontier

WINDOWS = [("signs", 335, 330, 288), ("truck", 460, 300, 288), ("facade", 560, 520, 288)]
LABEL_H = 28


def tile(method, crop, x, y, size):
    noisy = cv2.cvtColor(cv2.imread(str(evaluate.DATA / "his" / f"R0006362-{crop}.png")), cv2.COLOR_BGR2RGB)
    if method != "input":
        path = evaluate.DATA / "out" / "frontier" / f"R0006362-{crop}_{method}.png"
        out = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        noisy = frontier.align(out, noisy)[0]
    return noisy[y : y + size, x : x + size]


def row(method, label):
    tiles = np.hstack([tile(method, *w) for w in WINDOWS])
    bar = np.full((LABEL_H, tiles.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, tiles, np.full((4, tiles.shape[1], 3), 255, np.uint8)])


def main(out, specs):
    rows = [row(*(s.split("=", 1) if "=" in s else (s, s))) for s in specs]
    cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
