"""Aligned raw crops of real noisy/clean pairs, the faithful model's training data (train_raw.py).

    uv run raw_pairs.py 4 0          the first of four workers; every worker skips scenes already done
    uv run raw_pairs.py 6 0 raise    the same for RAISE's clean frames
    uv run raw_pairs.py 4 0 pixls    and for raw.pixls.us's
    uv run raw_pairs.py 2 0 own      and for his own photos
    uv run raw_pairs.py 2 0 ladders  and for review sites' ISO ladders (fetch_ladders.py)

Sources: every RawNIND Bayer scene with a frame at ISO 3,200 or more (fetched from the UCLouvain
Dataverse, CC BY-SA 4.0, and deleted once cropped), and the pairs already in data/night-raw: ELD,
SID Sony, RawNIND, and Imaging Resource's Ricoh GR II and GR III lab scenes. Held out: every Ricoh
GR IIIx frame, every Imaging Resource still life (the GR IIIx test scene, shot the same way on each
camera), X-Trans sensors, and the RawNIND scenes evaluate.py's NIND set comes from.

Clean frames alone add scenes for laid-over noise, from three places. His own photos at ISO 200 or
less, listed one host:path a line in data/own/paths.txt and read over ssh: the GR IIIx's own lens and
sensor, and his to train anything on. raw.pixls.us's public-domain (CC0) raws at ISO 400 or less, a
frame from each of hundreds of cameras. And RAISE's Nikon raws, which are for research only: a model
that is sold has to leave them out (train_raw.py --scenes).

Review sites' ISO ladders (fetch_ladders.py) are real pairs from hundreds of cameras: a camera's
lowest ISO against each of its high ones, on a tripod. Most of a site's cameras shot one scene, so a
ladder brings few crops, and Imaging Resource's still life stays out whole, as the test. They are
review samples, for research runs only.

A frame that turns out too noisy or not Bayer leaves <scene>.skip with its ISO and camera, so no
worker fetches it twice.

Each scene becomes data/rawpairs/<scene>.npy, float16 (frames, crops, 512, 512): frame 0 is the
clean mosaic, then come the noisy frames brought to its brightness, then one clean twin for each of
them. A tripod still leaves a noisy frame a fraction of a pixel off the clean one, and a camera that
steadies its sensor turns it a little too, so offsets() fits where each crop sits to a hundredth of a
pixel. The noisy crop is cut at the nearest whole even pixel and stays as the camera made it; its
twin is the clean crop resampled onto it. Mosaics are black-subtracted, divided by the white level
and cut to start on red, so every camera reads as RGGB.
"""

import hashlib
import json
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np
import rawpy

DATA = pathlib.Path(__file__).parent / "data"
OUT = DATA / "rawpairs"
NIGHT = DATA / "night-raw"
OWN = DATA / "own" / "paths.txt"
LADDERS = DATA / "ladders"
SITES = {"ir": "Imaging Resource", "pb": "Photography Blog", "epz": "ePHOTOzine", "dpreview": "DPReview"}
#: Cameras no scene of which trains, by a part of their name on any site: bench_raw.py's unseen cameras.
HELD_OUT = ("gr-iiix", "eos-r6", "z6-ii", "a7-iv", "a6400", "e-m1-mark-iii", "pentax-k-1", "rx100-vii", "nx500")
PIXLS = "https://raw.pixls.us/json/getrepository.php?set=all"
HOST = "https://dataverse.uclouvain.be/api"
DOI = "doi:10.14428/DVN/DEQCIM"
CROP = 512
CROPS = 24
MAX_SHIFT = 16
TILE, ROUNDS, MARGIN = 64, 5, 8
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


def detail(cfa):
    """Green without its fine noise or its slow changes of brightness: what a fit to a fraction of a pixel holds on to."""
    plane = green(cfa)
    return cv2.GaussianBlur(plane, (0, 0), 2) - cv2.GaussianBlur(plane, (0, 0), 8)


def offsets(clean, noisy):
    """The noisy frame's offset from the clean one at any place, to a fraction of a sensor pixel.

    shift_between gets within a few pixels. From there the offset is fitted as a plane over the frame,
    which covers a shift, a turn and a change of scale, in ROUNDS Gauss-Newton steps on tiles of green
    detail, TILE green pixels a side. The slopes are the clean frame's, so the noisy one's noise cannot
    pull the fit. Returns at(y, x), the (dx, dy) for which noisy[y + dy, x + dx] shows what clean[y, x]
    does.
    """
    c, n = detail(clean), detail(noisy)
    h, w = c.shape
    rows, cols = (h - 2 * TILE) // TILE, (w - 2 * TILE) // TILE
    tiles = lambda a: (
        a[TILE : TILE + rows * TILE, TILE : TILE + cols * TILE].reshape(rows, TILE, cols, TILE).sum((1, 3))
    )
    y, x = np.mgrid[:rows, :cols] * TILE + 1.5 * TILE
    place = np.stack([np.ones(rows * cols), x.ravel() / w, y.ravel() / w], 1)
    outer = place[:, :, None] * place[:, None, :]
    dx, dy = shift_between(clean, noisy)
    plane = np.array([[dx / 2, 0, 0], [dy / 2, 0, 0]], np.float64)
    for _ in range(ROUNDS):
        (ax, bx, cx), (ay, by, cy) = plane
        to_clean = np.array([[1 - bx / w, -cx / w, -ax], [-by / w, 1 - cy / w, -ay]])
        moved = cv2.warpAffine(c, to_clean, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
        gx, gy = np.gradient(moved, axis=1), np.gradient(moved, axis=0)
        rest = n - moved
        weigh = lambda a: (tiles(a).ravel()[:, None, None] * outer).sum(0)
        pull = lambda a: (tiles(a).ravel()[:, None] * place).sum(0)
        hessian = np.block([[weigh(gx * gx), weigh(gx * gy)], [weigh(gx * gy), weigh(gy * gy)]])
        plane -= np.linalg.solve(hessian, np.concatenate([pull(gx * rest), pull(gy * rest)])).reshape(2, 3)
    return lambda y, x: tuple(2 * plane @ (1, x / 2 / w, y / 2 / w))


def resampled(clean, y, x, dx, dy):
    """The clean crop at (y, x) as a frame (dx, dy) sensor pixels off shows it, each colour plane on its own.

    Lanczos reads MARGIN pixels past the crop's edge.
    """
    region = clean[y - MARGIN : y + CROP + MARGIN, x - MARGIN : x + CROP + MARGIN]
    out = np.empty_like(region)
    move = np.float32([[1, 0, dx / 2], [0, 1, dy / 2]])
    for i in range(2):
        for j in range(2):
            plane = np.ascontiguousarray(region[i::2, j::2])
            out[i::2, j::2] = cv2.warpAffine(
                plane, move, plane.shape[::-1], flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
            )
    return out[MARGIN:-MARGIN, MARGIN:-MARGIN]


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


def match(clean, noisy):
    """The noisy frame on the clean one's scale, each colour plane by a line through 64 px block means.

    One gain isn't enough: a noisy frame sits on an offset of its own that the file's black level
    misses, a quarter of the shadows' brightness on the GR III at ISO 102,400 and far more in SID and
    ELD's darkest frames. Returns the frame, and each plane's gain and the offset taken off it, on the
    clean frame's scale: the frame plus those offsets is the noisy frame as shot.
    """
    out = np.empty_like(noisy)
    slopes, blacks = [], []
    for i in range(2):
        for j in range(2):
            means = [
                p[: p.shape[0] // 64 * 64, : p.shape[1] // 64 * 64].reshape(p.shape[0] // 64, 64, -1, 64).mean((1, 3))
                for p in (noisy[i::2, j::2], clean[i::2, j::2])
            ]
            unclipped = means[1] < 0.5
            # Fitted from the clean means, which carry no noise to flatten the line.
            slope, offset = np.polyfit(means[1][unclipped], means[0][unclipped], 1)
            out[i::2, j::2] = (noisy[i::2, j::2] - offset) / slope
            slopes.append(1 / float(slope))
            blacks.append(float(offset / slope))
    return out, slopes, blacks


def tags(path):
    """A raw's ISO (0 when it names none) and camera, read from its EXIF by exiv2."""
    keys = {"Exif.Photo.ISOSpeedRatings": "iso", "Exif.Image.Model": "camera"}
    listed = subprocess.run(
        ["exiv2", "-Pkv", *(f"-g{key}" for key in keys), str(path)], check=False, capture_output=True, text=True
    )
    found = {keys[key]: value.strip() for key, _, value in (line.partition(" ") for line in listed.stdout.splitlines())}
    iso = re.match(r"\d+", found.get("iso", ""))
    return {"iso": int(iso.group()) if iso else 0, "camera": found.get("camera", "")}


def twinned(meta):
    """Whether a built scene is up to date: it has no noisy frames, or each has its clean twin."""
    return not meta["frames"] or meta.get("twins", False)


def build(name, clean_path, noisy_paths, meta):
    out, skip = OUT / f"{name}.npy", OUT / f"{name}.skip"
    built = OUT / f"{name}.json"
    if skip.exists() or out.exists() and built.exists() and twinned(json.loads(built.read_text())):
        return
    path = clean_path()
    limit = meta.pop("clean_iso", None)
    if limit:
        meta |= tags(path)
    clean = mosaic(path) if not limit or 0 < meta["iso"] <= limit else None
    if clean is None:
        skip.write_text(json.dumps(meta))
        print(name, "skipped", meta.get("iso", "not Bayer"), flush=True)
        return
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    picks = corners(clean, rng, meta.pop("crops", CROPS))
    frames = [np.stack([clean[y : y + CROP, x : x + CROP] for y, x in picks])]
    twins, kept = [], []
    for iso, path in noisy_paths:
        noisy = mosaic(path())
        if noisy is None or noisy.shape != clean.shape:
            continue
        at = offsets(clean, noisy)
        moves = np.array([at(y + CROP // 2, x + CROP // 2) for y, x in picks])
        whole = (np.round(moves / 2) * 2).astype(int)
        if np.abs(whole).max() > MAX_SHIFT:
            print(name, iso, "shifted", *moves.mean(0).round(1), flush=True)
            continue
        noisy, gains, blacks = match(clean, noisy)
        dark = green(clean) < np.percentile(green(clean), 20)
        lift = float(green(noisy)[dark].mean() / green(clean)[dark].mean())
        # SID and ELD's darkest frames glow unevenly; a model taught on them learns the glow as signal.
        if not 0.8 < lift < 1.25:
            print(name, iso, "shadows off", round(lift, 2), flush=True)
            continue
        cut = np.stack(
            [noisy[y + dy : y + dy + CROP, x + dx : x + dx + CROP] for (y, x), (dx, dy) in zip(picks, whole)]
        )
        twin = np.stack([resampled(clean, y, x, dx, dy) for (y, x), (dx, dy) in zip(picks, moves - whole)])
        # Light that changed between the frames, or a subject that moved, leaves 64 px blocks whose
        # brightness disagrees. A lab ladder is within 3% at nine frames in ten; some reviews' are not.
        means = lambda crops: crops.reshape(len(crops), CROP // 64, 64, CROP // 64, 64).mean((2, 4))
        drift = float(np.median(np.abs(means(cut) - means(twin)) / (means(twin) + 0.01)))
        if drift > 0.05:
            print(name, iso, "drifted", round(drift, 3), flush=True)
            continue
        frames.append(cut)
        twins.append(twin)
        shift = moves.mean(0).round(2).tolist()
        offset = [round(black, 6) for black in blacks]
        kept.append({"iso": iso, "shift": shift, "gain": round(float(np.mean(gains)), 3), "offset": offset})
    if noisy_paths and not kept:
        return
    np.save(OUT / f"{name}.tmp.npy", np.stack(frames + twins).astype(np.float16))
    (OUT / f"{name}.tmp.npy").rename(out)
    # Renamed into place whole: train_raw.py reads these while a build is still writing them.
    (OUT / f"{name}.tmp").write_text(json.dumps({**meta, "frames": kept, "twins": True}))
    (OUT / f"{name}.tmp").rename(OUT / f"{name}.json")
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


def copied(source):
    """The same for a raw on another machine, named host:path."""
    host, path = source.split(":", 1)

    def get():
        local = OUT / f"fetch-{hashlib.sha1(source.encode()).hexdigest()[:12]}{pathlib.Path(path).suffix}"
        fetched.paths.append(local)
        with local.open("wb") as file:
            subprocess.run(["ssh", host, "cat", shlex.quote(path)], stdout=file, check=True, timeout=900)
        # iCloud Drive downloads a file onto the Mac when it is read; evicting leaves the disk as it was.
        # A file just downloaded can stay locked for some seconds.
        if "/Mobile Documents/" in path:
            evict = f"for n in 1 2 3 4 5 6; do brctl evict {shlex.quote(path)} && exit; sleep 10; done; exit 1"
            evicted = subprocess.run(["ssh", host, evict], check=False, capture_output=True)
            if evicted.returncode:
                sys.exit(f"{source} stays downloaded: {evicted.stderr.decode()}")
        return local

    return get


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


def pixls_scenes():
    """raw.pixls.us's public-domain raws of 10 MP or more, a frame from each camera and mode."""
    for _, _, _, megapixels, _, licence, _, link, _ in json.load(urllib.request.urlopen(PIXLS))["data"]:
        url = re.search(r"href='([^']+)'", link).group(1)
        if "publicdomain/zero" in licence and float(megapixels or 0) >= 10:
            yield (
                f"pixls-{url.split('/')[4]}",
                fetched(urllib.parse.quote(url, safe=":/"), pathlib.Path(url).suffix),
                [],
                {"source": "raw.pixls.us", "license": "CC0", "crops": 6, "clean_iso": 400},
            )


def own_scenes():
    """His own photos, wherever data/own/paths.txt says they are."""
    for source in OWN.read_text().splitlines():
        digest = hashlib.sha1(source.encode()).hexdigest()[:6]
        meta = {"source": "Own", "license": "own", "crops": 12, "clean_iso": 200, "path": source}
        yield f"own-{pathlib.Path(source).stem}-{digest}", copied(source), [], meta


def ladder_scenes():
    for folder in sorted(LADDERS.glob("*/*/*/")):
        site, camera, scene = folder.relative_to(LADDERS).parts
        steps = sorted((int(file.stem), file) for file in folder.iterdir() if file.stem.isdigit())
        if (site, scene) == ("ir", "sli") or any(name in camera for name in HELD_OUT) or len(steps) < 2:
            continue
        (base, clean), *rest = steps
        # A fetch that missed files can leave a ladder without its low end.
        noisy = [(iso, lambda f=file: f) for iso, file in rest if iso >= 16 * base]
        if base > 400 or not noisy:
            continue
        # Photography Blog shot a scene of its own for every review, so its ladders are worth more crops.
        meta = {
            "source": SITES[site],
            "license": "review samples",
            "camera": camera,
            "crops": 16 if site == "pb" else 8,
        }
        yield f"ladder-{site}-{camera}-{scene}", lambda c=clean: c, noisy, meta


SOURCES = {
    "pairs": lambda: [*local_scenes(), *rawnind_scenes()],
    "raise": raise_scenes,
    "pixls": pixls_scenes,
    "own": own_scenes,
    "ladders": ladder_scenes,
}


def split(name):
    """About one RawNIND scene in ten is kept out of training to watch for overfitting."""
    return "val" if name.startswith("rawnind-") and hashlib.sha1(name.encode()).digest()[0] % 10 == 0 else "train"


def main(workers, worker, source="pairs"):
    OUT.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(300)
    scenes = list(SOURCES[source]())
    for name, clean, noisy, meta in scenes[worker::workers]:
        try:
            build(name, clean, noisy, {**meta, "scene": name, "split": split(name)})
        # One bad download or file shouldn't stop the rest.
        except (OSError, ValueError, rawpy.LibRawError, subprocess.SubprocessError) as error:
            print(name, "failed", repr(error), flush=True)
        for path in fetched.paths:
            path.unlink(missing_ok=True)
        fetched.paths.clear()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:])
