import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "repo"), str(ROOT / "repo/diffusers/src")]
W = ROOT / "w"
NAMES = sorted(p.stem for p in (ROOT / "in").glob("*.png"))
PROMPT = "Please remove noise from the image."
SEED = 42
