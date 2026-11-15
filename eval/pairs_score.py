"""Score every model's output on the GR IIIx night pairs against the ISO 100 frame: uv run pairs_score.py

Reads data/out/pairs/<crop>_<method>.png next to data/his/pairs/<crop>-clean.png, prints PSNR, SSIM
and LPIPS per ISO (LPIPS is the perceptual distance, lower is better) and writes data/out/pairs.json.
A model is scored on a crop only where it has an output, so n says how many crops each mean covers.
"""

import collections
import json

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import evaluate

PAIRS = evaluate.DATA / "his" / "pairs"
OUT = evaluate.DATA / "out" / "pairs"


def read(path):
    image = cv2.imread(str(path))
    if image.shape[:2] != (1024, 1024):
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    perceptual = lpips.LPIPS(net="alex", verbose=False)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for noisy in sorted(p for p in PAIRS.glob("*.png") if not p.stem.endswith("-clean")):
        clean = read(PAIRS / f"{noisy.stem}-clean.png")
        iso = noisy.stem.split("-")[1].lstrip("0")
        outputs = {"noisy": noisy} | {p.stem.split("_", 1)[1]: p for p in OUT.glob(f"{noisy.stem}_*.png")}
        for method, path in outputs.items():
            image = read(path)
            with torch.no_grad():
                distance = perceptual(evaluate.to_tensor(image) * 2 - 1, evaluate.to_tensor(clean) * 2 - 1).item()
            scores = (
                peak_signal_noise_ratio(clean, image),
                structural_similarity(clean, image, channel_axis=-1),
                distance,
            )
            for group in (iso, "all"):
                rows[group][method].append(scores)
    table = {
        group: {m: [len(s), *np.mean(s, axis=0).round(4).tolist()] for m, s in by_method.items()}
        for group, by_method in rows.items()
    }
    (evaluate.DATA / "out" / "pairs.json").write_text(json.dumps(table, indent=1))
    for group in sorted(table):
        print(f"\nGR IIIx ISO {group}\n{'method':24}{'n':>3}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
        for m, (n, psnr, ssim, dist) in sorted(table[group].items(), key=lambda kv: -kv[1][1]):
            print(f"{m:24}{n:3}{psnr:8.2f}{ssim:8.3f}{dist:8.3f}")


if __name__ == "__main__":
    main()
