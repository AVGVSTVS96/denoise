"""Tile a 1024 crop into 512 tiles at stride 256 for PixRestore (trained at 512), then feather its outputs back together.

    python pixrestore_tiles.py tile <crop.png> <tiles-dir>
    python pixrestore_tiles.py blend <crop.png> <pixrestore-out-dir> <out.png>
"""
import sys, pathlib
import numpy as np, cv2
T, S = 512, 256
def starts(n): return list(range(0, n - T + 1, S))
mode, src, tiles = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
img = cv2.imread(str(src)); h, w = img.shape[:2]
if mode == "tile":
    tiles.mkdir(parents=True, exist_ok=True)
    for y in starts(h):
        for x in starts(w):
            cv2.imwrite(str(tiles / f"{src.stem}__{y}_{x}.png"), img[y:y+T, x:x+T])
else:
    out = pathlib.Path(sys.argv[4])
    ramp = np.minimum(np.arange(T) + 1, T - np.arange(T)).astype(np.float32)
    wgt = np.outer(ramp, ramp)[..., None]
    acc = np.zeros((h, w, 3), np.float32); norm = np.zeros((h, w, 1), np.float32)
    for y in starts(h):
        for x in starts(w):
            f = next(tiles.rglob(f"{src.stem}__{y}_{x}.*"))
            t = cv2.imread(str(f)).astype(np.float32)
            t = cv2.resize(t, (T, T)) if t.shape[:2] != (T, T) else t
            acc[y:y+T, x:x+T] += t * wgt; norm[y:y+T, x:x+T] += wgt
    cv2.imwrite(str(out), np.clip(acc / norm, 0, 255).astype(np.uint8))
