"""Aligned raw crops of real noisy/clean pairs, the faithful model's training data (train_raw.py).

    uv run raw_pairs.py 4 0          the first of four workers; every worker skips scenes already done
    uv run raw_pairs.py 6 0 raise    the same for RAISE's clean frames

Sources: every RawNIND Bayer scene with a frame at ISO 3,200 or more (fetched from the UCLouvain
Dataverse, CC BY-SA 4.0, and deleted once cropped), and the pairs already in data/night-raw: ELD,
SID Sony, RawNIND, and Imaging Resource's Ricoh GR II and GR III lab scenes. Held out: every Ricoh
GR IIIx frame, every Imaging Resource still life (the GR IIIx test scene, shot the same way on each
camera), X-Trans sensors, and the RawNIND scenes evaluate.py's NIND set comes from.

RAISE's clean Nikon raws (research use only) add scenes for made-up noise.

Each scene becomes data/rawpairs/<scene>.npy, float16 (frames, crops, 512, 512): frame 0 is the
clean mosaic, the rest are noisy frames brought to its brightness. Mosaics are black-subtracted,
divided by the white level and cut to start on red, so every camera reads as RGGB.
"""

import hashlib
import json
import pathlib
import re
import sys
import urllib.request

import cv2
import numpy as np
import rawpy

DATA = pathlib.Path(__file__).parent / "data"
OUT = DATA / "rawpairs"
NIGHT = DATA / "night-raw"
HOST = "https://dataverse.uclouvain.be/api"
DOI = "doi:10.14428/DVN/DEQCIM"
CROP = 512
CROPS = 24
MAX_SHIFT = 16
BANDS = ((3200, 6400), (12800, 20000), (25600, 40000), (51200, 10**9))
NIND_TEST = ("bark", "blombukett", "bruegel", "museel-bobo", "ursulines", "stefantiek", "courtine", "pen-pile")
IR_SCENES = {
    "ir-gr3-inbi": (
        "ricoh-gr-iii--GR3INBI000100.DNG",
        "ricoh-gr-iii--GR3INBI{:06d}.DNG",
        (3200, 6400, 12800, 25600, 51200, 102400),
    ),
    "ir-gr2-inbi": ("ricoh-gr-ii--GR2INBI00100.DNG", "ricoh-gr-ii--GR2INBI{:05d}.DNG", (3200, 6400, 12800, 25600)),
    "ir-gr2-ll3": ("ricoh-gr-ii--GR2LL001003.DNG", "ricoh-gr-ii--GR2LL{:03d}003.DNG", (3200, 25600)),
    "ir-gr2-ll7": ("ricoh-gr-ii--GR2LL001007.DNG", "ricoh-gr-ii--GR2LL{:03d}007.DNG", (3200, 25600)),
}


def mosaic(path):
    """Normalised RGGB mosaic, or None for a sensor that isn't a 2x2 Bayer."""
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern
        if pattern is None or pattern.shape != (2, 2):
            return None
        colours = raw.color_desc.decode()
        names = [[colours[pattern[i, j]] for j in range(2)] for i in range(2)]
        cfa = raw.raw_image_visible.astype(np.float32)
        black = np.asarray(raw.black_level_per_channel, np.float32)[pattern]
        white = float(raw.white_level)
    level = np.tile(black, (cfa.shape[0] // 2 + 1, cfa.shape[1] // 2 + 1))[: cfa.shape[0], : cfa.shape[1]]
    cfa = (cfa - level) / (white - level)
    y, x = next((i, j) for i in range(2) for j in range(2) if names[i][j] == "R")
    if names[1 - y][1 - x] != "B":
        return None
    cfa = cfa[y:, x:]
    return cfa[: cfa.shape[0] // 2 * 2, : cfa.shape[1] // 2 * 2]


def green(cfa):
    return (cfa[0::2, 1::2] + cfa[1::2, 0::2]) / 2


def shift_between(clean, noisy):
    """Whole-pixel offset of the noisy frame, in sensor pixels and kept even so the colours line up."""
    small = [cv2.resize(cv2.GaussianBlur(green(i), (0, 0), 3), None, fx=0.5, fy=0.5) for i in (clean, noisy)]
    (dx, dy), _ = cv2.phaseCorrelate(small[0], small[1])
    return round(dx * 2) * 2, round(dy * 2) * 2


def corners(clean, rng, count):
    """Even top-left corners on a 512 px grid: the most detailed crops, and a third at random."""
    h, w = clean.shape
    grid = [
        (y, x)
        for y in range(MAX_SHIFT, h - CROP - MAX_SHIFT, CROP)
        for x in range(MAX_SHIFT, w - CROP - MAX_SHIFT, CROP)
    ]
    g = green(clean)
    detail = [
        float(np.abs(cv2.Laplacian(g[y // 2 : (y + CROP) // 2, x // 2 : (x + CROP) // 2], cv2.CV_32F)).mean())
        for y, x in grid
    ]
    ranked = [grid[i] for i in np.argsort(detail)[::-1]]
    n = min(count, len(grid))
    rest = ranked[n * 2 // 3 :]
    return ranked[: n * 2 // 3] + [rest[i] for i in rng.choice(len(rest), n - n * 2 // 3, replace=False)]


def gain(clean, noisy):
    """Brightness match over the clean frame's brighter, unclipped parts: in shadows, black clipping lifts noise."""
    g = green(clean)
    bright = (g > np.percentile(g, 60)) & (g < min(np.percentile(g, 99), 0.9))
    return float(g[bright].mean() / max(green(noisy)[bright].mean(), 1e-6))


def build(name, clean_path, noisy_paths, meta):
    out = OUT / f"{name}.npy"
    if out.exists():
        return
    clean = mosaic(clean_path())
    if clean is None:
        print(name, "not Bayer", flush=True)
        return
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    picks = corners(clean, rng, meta.pop("crops", CROPS))
    frames = [np.stack([clean[y : y + CROP, x : x + CROP] for y, x in picks])]
    kept = []
    for iso, path in noisy_paths:
        noisy = mosaic(path())
        if noisy is None or noisy.shape != clean.shape:
            continue
        dx, dy = shift_between(clean, noisy)
        if max(abs(dx), abs(dy)) > MAX_SHIFT:
            print(name, iso, "shifted", dx, dy, flush=True)
            continue
        g = gain(clean, noisy)
        frames.append(np.stack([noisy[y + dy : y + dy + CROP, x + dx : x + dx + CROP] * g for y, x in picks]))
        kept.append({"iso": iso, "shift": [dx, dy], "gain": round(g, 3)})
    if noisy_paths and not kept:
        return
    np.save(OUT / f"{name}.tmp.npy", np.stack(frames).astype(np.float16))
    (OUT / f"{name}.tmp.npy").rename(out)
    (OUT / f"{name}.json").write_text(json.dumps({**meta, "frames": kept}))
    print(name, kept, flush=True)


def fetched(url, suffix=".arw"):
    def get():
        path = OUT / f"fetch-{hashlib.sha1(url.encode()).hexdigest()[:12]}{suffix}"
        if not path.exists():
            urllib.request.urlretrieve(url, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)
        fetched.paths.append(path)
        return path

    return get


fetched.paths = []


def rawnind_scenes():
    listing = json.load(urllib.request.urlopen(f"{HOST}/datasets/:persistentId/?persistentId={DOI}"))
    found = {}
    for entry in listing["data"]["latestVersion"]["files"]:
        file = entry["dataFile"]
        m = re.match(r"Bayer_(.+?)_(GT_)?ISO(\d+)_sha1=\w+\.arw$", file["filename"])
        if not m:
            continue
        scene = found.setdefault(m.group(1), {"clean": None, "noisy": {}})
        if m.group(2):
            scene["clean"] = scene["clean"] or file["id"]
        else:
            scene["noisy"].setdefault(int(m.group(3)), file["id"])
    for name, s in sorted(found.items()):
        if not s["clean"] or name.lower().startswith(NIND_TEST):
            continue
        picks = []
        for low, high in BANDS:
            isos = sorted(i for i in s["noisy"] if low <= i <= high)
            if isos:
                picks.append(isos[-1])
        # The Dataverse serves about 1 MB/s in all, so each scene brings its two noisiest bands.
        file = lambda i: fetched(f"{HOST}/access/datafile/{i}")
        if picks:
            yield (
                f"rawnind-{name}",
                file(s["clean"]),
                [(i, file(s["noisy"][i])) for i in picks[-2:]],
                {
                    "source": "RawNIND",
                    "license": "CC BY-SA 4.0",
                },
            )


def local_scenes():
    for folder in sorted((NIGHT / "pairs").iterdir()):
        name = folder.name
        if name.startswith(("sid-fuji", "rawnind-")):  # X-Trans; RawNIND comes whole from the Dataverse
            continue
        clean = next(folder.glob("clean_*"))
        noisy = sorted(folder.glob("noisy_*"))
        source = name.split("-")[0].upper()
        yield f"night-{name}", lambda c=clean: c, [(0, lambda n=n: n) for n in noisy], {"source": source}
    for name, (clean, noisy, isos) in IR_SCENES.items():
        folder = NIGHT / "imaging-resource"
        # The low-light series names its ISO in hundreds.
        frames = [(i, folder / noisy.format(i if "INBI" in noisy else i // 100)) for i in isos]
        yield name, lambda c=folder / clean: c, [(i, lambda p=p: p) for i, p in frames], {"source": "Imaging Resource"}


def raise_scenes():
    """RAISE's 1,000 Nikon raws at ISO 800 or less: clean frames only, for made-up noise (research use only)."""
    import csv
    import io
    import zipfile

    with zipfile.ZipFile(DATA / "raise" / "raise1k.zip") as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read(archive.namelist()[0]).decode())))
    for row in rows:
        if int(re.sub(r"\D", "", row["ISO Sensitivity"]) or 10**6) <= 800:
            yield (
                f"raise-{row['File']}",
                fetched(row["NEF"], ".nef"),
                [],
                {
                    "source": "RAISE",
                    "license": "non-commercial research",
                    "crops": 6,
                },
            )


def split(name):
    """About one RawNIND scene in ten is kept out of training to watch for overfitting."""
    return "val" if name.startswith("rawnind-") and hashlib.sha1(name.encode()).digest()[0] % 10 == 0 else "train"


def main(workers, worker, source="pairs"):
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = list(raise_scenes()) if source == "raise" else [*local_scenes(), *rawnind_scenes()]
    for name, clean, noisy, meta in scenes[worker::workers]:
        try:
            build(name, clean, noisy, {**meta, "scene": name, "split": split(name)})
        except (OSError, ValueError, rawpy.LibRawError) as error:  # one bad download or file shouldn't stop the rest
            print(name, "failed", repr(error), flush=True)
        for path in fetched.paths:
            path.unlink(missing_ok=True)
        fetched.paths.clear()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:])
