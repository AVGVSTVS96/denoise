"""Before/after at 100% on a real photo's crops: uv run photo_sheet.py out.jpg <crop stem> x y size -- method ..."""

import sys

import cv2
import numpy as np

import evaluate

LABELS = {"input": "his shot, ISO 10,000", "tuned": "ours (app)"}


def main(out, stem, x, y, size, methods):
    panels = []
    for m in methods:
        path = (
            evaluate.DATA
            / ("his" if m == "input" else "out/frontier")
            / (f"{stem}.png" if m == "input" else f"{stem}_{m}.png")
        )
        image = cv2.imread(str(path))
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)[y : y + size, x : x + size].copy()
        label = LABELS.get(m, m)
        cv2.rectangle(image, (0, 0), (12 + 10 * len(label), 26), (0, 0, 0), -1)
        cv2.putText(image, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    cv2.imwrite(out, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 93])


if __name__ == "__main__":
    split = sys.argv.index("--")
    out, stem, x, y, size = sys.argv[1:split]
    main(out, stem, int(x), int(y), int(size), sys.argv[split + 1 :])
