"""Run local denoisers (evaluate.METHODS) on the night-pair crops: uv run pairs_run.py method ...

Reads data/his/pairs/<name>.png, writes data/out/pairs/<name>_<method>.png.
"""

import sys

import cv2

import evaluate

PAIRS = evaluate.DATA / "his" / "pairs"
OUT = evaluate.DATA / "out" / "pairs"

for name in sys.argv[1:]:
    method = evaluate.METHODS[name]()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(p for p in PAIRS.glob("*.png") if not p.stem.endswith("-clean")):
        image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(OUT / f"{f.stem}_{name}.png"), cv2.cvtColor(method(image), cv2.COLOR_RGB2BGR))
        print(name, f.stem, flush=True)
