"""Before/after sheet, one row per test image: uv run sheet.py out.jpg nind <key> <key> ... -- noisy app clean"""

import sys

import cv2
import numpy as np

import evaluate

SIZE = 320
LABELS = {"noisy": "shot", "app": "denoised", "clean": "base ISO"}


def panel(image, label):
    y, x = (image.shape[0] - SIZE) // 2, (image.shape[1] - SIZE) // 2
    crop = np.ascontiguousarray(image[y : y + SIZE, x : x + SIZE])
    cv2.rectangle(crop, (0, 0), (12 + 11 * len(label), 26), (0, 0, 0), -1)
    cv2.putText(crop, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def main(out, set_name, keys, methods):
    images = {k: (iso, n, c) for k, iso, n, c in evaluate.SETS[set_name]() if k in keys}
    rows = []
    for key in keys:
        iso, noisy, clean = images[key]
        panels = []
        for m in methods:
            image = {"noisy": noisy, "clean": clean}.get(m)
            if image is None:
                image = cv2.cvtColor(
                    cv2.imread(str(evaluate.DATA / "out" / set_name / m / f"{key}.png")), cv2.COLOR_BGR2RGB
                )
            panels.append(panel(image, f"ISO {iso:,}" if m == "noisy" else LABELS.get(m, m)))
        rows.append(np.hstack(panels))
    cv2.imwrite(out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])


if __name__ == "__main__":
    split = sys.argv.index("--")
    main(sys.argv[1], sys.argv[2], sys.argv[3:split], sys.argv[split + 1 :])
