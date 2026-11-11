"""Real high-ISO test crops from NIND (Brummer & De Vleeschouwer, CC-BY/CC0 on Wikimedia Commons).

Each scene is a tripod series of one subject from base ISO to the camera's highest. For every
scene with a top ISO of 12,800 or more this saves the two most detailed 768 px crops of the
base-ISO frame and the same crops of each frame at ISO 12,800 or more, as 8-bit PNG under data/nind/.
"""

import collections
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

DATA = pathlib.Path(__file__).parent / "data"
OUT = DATA / "nind"
AGENT = {"User-Agent": "denoise-eval/0.1 (research script)"}
CROP = 768
#: Fujifilm X-T1's extended settings, as ISO numbers.
EXTENDED = {"H1": 12800, "H2": 25600, "H3": 51200, "H4": 102400}
HIGH = 12800
#: Seconds between full-size downloads.
PAUSE = 90


def api(**params):
    query = urllib.parse.urlencode({"format": "json", "action": "query", **params})
    request = urllib.request.Request(f"https://commons.wikimedia.org/w/api.php?{query}", headers=AGENT)
    return json.load(urllib.request.urlopen(request))


def index():
    files, cont = [], {}
    while True:
        page = api(list="allimages", aiprefix="NIND_", ailimit=500, aiprop="url", **cont)
        files += page["query"]["allimages"]
        if "continue" not in page:
            break
        cont = page["continue"]
    scenes = collections.defaultdict(dict)
    for f in files:
        m = re.match(r"NIND_(.+)_ISO(\w+)\.png", f["name"])
        if m:
            iso = EXTENDED.get(m.group(2)) or int(m.group(2))
            scenes[m.group(1)][iso] = f["url"]
    return scenes


def load(url):
    """One full-size file, at the pace Commons allows: a pause after each, and its Retry-After on a 429."""
    request = urllib.request.Request(url, headers=AGENT)
    for _ in range(4):
        try:
            data = np.frombuffer(urllib.request.urlopen(request).read(), np.uint8)
            break
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 300))
    else:
        raise SystemExit("Commons is still refusing full-size files; try again later")
    time.sleep(PAUSE)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)[:, :, :3]
    return (image / 257).round().astype(np.uint8) if image.dtype == np.uint16 else image


def detailed_crops(base):
    """Top-left corners of the two non-overlapping crops with the most edges."""
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = cv2.boxFilter(
        np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 2), cv2.CV_32F)), -1, (CROP, CROP), anchor=(0, 0)
    )
    energy = energy[: gray.shape[0] - CROP, : gray.shape[1] - CROP].copy()
    corners = []
    for _ in range(2):
        y, x = np.unravel_index(np.argmax(energy), energy.shape)
        corners.append((int(y), int(x)))
        energy[max(0, y - CROP) : y + CROP, max(0, x - CROP) : x + CROP] = -1
    return corners


def main(limit):
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = {name: isos for name, isos in index().items() if max(isos) >= 12800}
    order = sorted(scenes, key=lambda s: (-min(max(scenes[s]), 51200), s))[:limit]
    for name in order:
        isos = scenes[name]
        if (OUT / f"{name}.done").exists():
            continue
        base = load(isos[min(isos)])
        corners = detailed_crops(base)
        frames = {min(isos): base} | {iso: load(isos[iso]) for iso in sorted(isos) if iso >= HIGH}
        for iso, frame in frames.items():
            if frame.shape != base.shape:
                continue
            for n, (y, x) in enumerate(corners):
                tag = "clean" if iso == min(isos) else f"iso{iso}"
                cv2.imwrite(str(OUT / f"{name}-{n}_{tag}.png"), frame[y : y + CROP, x : x + CROP])
        (OUT / f"{name}.done").touch()
        print(name, sorted(frames), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
