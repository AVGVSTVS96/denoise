"""Full-colour truth that a sensor measured: PixelShift200's frames, for bench_raw.py's shift test and train_raw.py --truth.

    uv run pixel_shift.py fetch    the raws, 436 files and 37 GB, into data/pixelshift
    uv run pixel_shift.py          every training scene's crops, into data/rawshift

PixelShift200 (Qian et al., ICCP 2022) is 109 scenes from a Sony A7R III on a tripod at ISO 100, each
shot four times with the sensor moved one pixel between shots (SHIFTS, down and across from the
first), so every pixel was in turn under a red filter, a blue one and two green ones. Laid on each
other the four mosaics are a picture with every colour measured at every pixel, at the sharpness
the lens gave: no demosaic made it and nothing was binned, which is what a camera's own frame looks
like and bench_raw.full_colour's truth, twice as sharp, does not. The dataset's 200 training and 10
test images are crops its authors cut from these raws; here a scene is a whole raw, and TEST's five
are the ones those test images came from. The files are linear sensor counts, black at 512 and
white at 16,383, and laid so they come out the same, count for count, as the authors' own.

Four shots take a second or more, and whatever moved meanwhile is wrong in the picture. The two
greens tell: they were measured at the same place in two different shots, so in a scene that held
still they differ by their shot noise and no more. An A7R III pixel holds ELECTRONS at white at
ISO 100, measured from scene 10's greens, which says how much that is. A crop is kept (still) while
at most SHARE of its 32 px blocks have their greens more than APART times that noise apart and none
more than WORST times: leaves in wind, birds and a flickering sign fall out and the rest of their
scene stays in. A quarter of an outdoor scene's blocks can fail, and two in ten thousand of a
studio scene's.

A scene becomes data/rawshift/<scene>.npy, float16 (3, crops, 512, 512): a crop's red, green and blue
where raw_pairs.py's scenes hold their frames, on its scale of black at 0 and white at 1.
"""

import hashlib
import json
import re
import sys
import urllib.request

import numpy as np

import raw_pairs

RAWS = raw_pairs.DATA / "pixelshift"
OUT = raw_pairs.DATA / "rawshift"
FOLDER = "https://drive.google.com/drive/folders/1S1mt9D8IlpSAVm7RTjLheh1hTaltsV_f"
SHIFTS = ((0, 0), (-1, 0), (-1, 1), (0, 1))
TEST = (5, 10, 13, 49, 98)
CROPS = 48
BLOCK = 32
ELECTRONS = 47300
APART, SHARE, WORST = 4, 0.02, 16


def fetch():
    import gdown

    RAWS.mkdir(parents=True, exist_ok=True)
    for file in gdown.download_folder(FOLDER, output=str(RAWS), skip_download=True, quiet=True, use_cookies=False):
        raw = RAWS / file.path.split("/")[-1]
        if raw.suffix == ".ARW" and not raw.exists():
            urllib.request.urlretrieve(f"https://drive.google.com/uc?id={file.id}", raw.with_suffix(".part"))
            raw.with_suffix(".part").rename(raw)


def shot(scene, number=1):
    return RAWS / f"pixelshift_{scene}_{number}.ARW"


def scenes():
    """The scenes with all four shots on disk, by number."""
    numbers = {int(re.search(r"_(\d+)_1", file.name).group(1)) for file in RAWS.glob("pixelshift_*_1.ARW")}
    return sorted(scene for scene in numbers if all(shot(scene, number).exists() for number in (2, 3, 4)))


def measured(scene):
    """A scene's four shots as camera RGB (3, H, W), and how far apart its two greens are at each pixel."""
    frames = [raw_pairs.mosaic(shot(scene, number)) for number in (1, 2, 3, 4)]
    h, w = (side - 4 for side in frames[0].shape)
    picture, apart = np.zeros((3, h, w), np.float32), np.zeros((h, w), np.float32)
    for frame, (dy, dx) in zip(frames, SHIFTS):
        laid = frame[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]
        for i in range(2):
            for j in range(2):
                colour = (0, 1, 1, 2)[2 * ((i + dy) % 2) + (j + dx) % 2]
                picture[colour, i::2, j::2] += laid[i::2, j::2] / (1 + colour % 2)
                if colour == 1:
                    apart[i::2, j::2] = laid[i::2, j::2] - apart[i::2, j::2]
    return picture, apart


def moved(picture, apart, iso):
    """Each BLOCK px block's greens apart, as a multiple of what their shot noise alone leaves: about 1 where nothing moved."""
    h, w = (side // BLOCK * BLOCK for side in apart.shape)
    blocks = lambda plane: plane[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3))
    return blocks(apart**2) / 2 / (blocks(picture[1]).clip(min=1e-4) * iso / 100 / ELECTRONS)


def still(ratios, y, x, side=raw_pairs.CROP):
    """Whether the crop at (y, x) held still: no bird or leaf crossed a block of it, and it did not tremble as a whole."""
    crop = ratios[y // BLOCK : (y + side) // BLOCK, x // BLOCK : (x + side) // BLOCK]
    return (crop > APART).mean() <= SHARE and crop.max() <= WORST


def cut(picture, picks, side=raw_pairs.CROP):
    return np.stack([picture[..., y : y + side, x : x + side] for y, x in picks])


def build(scene):
    name = f"shift-{scene:03d}"
    out = OUT / f"{name}.npy"
    if out.with_suffix(".json").exists():
        return
    picture, apart = measured(scene)
    tags = raw_pairs.tags(shot(scene))
    ratios = moved(picture, apart, tags["iso"])
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    found = raw_pairs.corners(picture[1], rng, CROPS)
    picks = [(y, x) for y, x in found if still(ratios, y, x)]
    if picks:
        np.save(OUT / f"{name}.tmp.npy", cut(picture, picks).swapaxes(0, 1).astype(np.float16))
        (OUT / f"{name}.tmp.npy").rename(out)
    meta = {"scene": name, "split": "train", "frames": [], "source": "PixelShift200", "license": "research", **tags}
    meta |= {"crops": len(picks), "moved": len(found) - len(picks), "apart": round(float(np.median(ratios)), 3)}
    (OUT / f"{name}.tmp").write_text(json.dumps(meta))
    (OUT / f"{name}.tmp").rename(out.with_suffix(".json"))
    print(name, meta, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for scene in scenes():
            if scene not in TEST:
                build(scene)
