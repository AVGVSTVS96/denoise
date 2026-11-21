"""A blind side-by-side sheet of real night photos, for a person to pick the better result by eye.

    uv run blind.py crop PHOTO X Y [NOTE]     add a crop: a raw, and the crop's middle as fractions of its frame
    uv run blind.py add joint-v2=data/models/raw/joint-v2/joint.pth darktable rawforge-jdd dxo=FOLDER
    uv run blind.py first joint-v1:joint-v2   the pairs the page asks before the rest
    uv run blind.py table                     who beat whom, from the picks so far

The photos are private, so nothing of the sheet is in this repo: the crops' list is in data/blind,
and the pictures go to SHEET, a folder a page on the tailnet shows (BLIND_SHEET names another), as
sheet.json and one PNG per crop and entrant under a name that does not say whose it is. The page shows
two entrants' pictures of one crop, in an order of its own, and saves each pick beside the sheet, in PICKS.

A crop is SIDE px of a photo, developed as the app develops it (raw_develop.py): blacks.py's offsets
off, the entrant's camera RGB, then the photo's white balance, matrix, exposure and tone curve. An
entrant is one of three kinds:

    NAME=WEIGHTS.pth   a joint model of ours, through train_raw.denoise() on the whole photo's gain and
                       tiles, of which only the tiles that reach the crop run
    a rivals.py name   a rival that hands back a picture, fed as rivals.py feeds it, on the crop and
                       MARGIN px around it
    NAME=FOLDER        what a program made of the photo, as a linear DNG (kit.py)

add makes whatever pictures are missing, so a new crop or a new entrant is the same one command.
"""

import collections
import functools
import hashlib
import json
import os
import pathlib
import sys

import cv2
import numpy as np
import rawpy
import torch

import blacks
import kit
import raw_develop
import raw_pairs
import rivals
import train_raw

SIDE = 512
MARGIN = 128
SHEET = pathlib.Path(os.environ.get("BLIND_SHEET", pathlib.Path.home() / "Developer/pages/pages/denoise-blind"))
PICKS = SHEET.parents[1] / "data" / f"{SHEET.name}.json"
CROPS = raw_pairs.DATA / "blind" / "crops.json"
SMALLEST_PREVIEW = 1000
Shot = collections.namedtuple("Shot", "photo path mosaic look flip red iso camera rival")


@functools.lru_cache(1)
def shot(photo):
    """A photo as every entrant gets it: its mosaic with the body's black offsets off, and how to develop it."""
    path = raw_pairs.DATA / photo
    with rawpy.imread(str(path)) as raw:
        colours, pattern, flip = raw.color_desc.decode(), raw.raw_pattern, raw.sizes.flip
        black = float(np.mean(raw.black_level_per_channel)) / raw.white_level
        from_xyz = raw.rgb_xyz_matrix[:3].tolist()
    red = next((i, j) for i in range(2) for j in range(2) if colours[pattern[i, j]] == "R")
    mosaic = raw_pairs.mosaic(path)
    mosaic = mosaic - np.tile(blacks.of(path).reshape(2, 2), (mosaic.shape[0] // 2, mosaic.shape[1] // 2))
    iso, camera = blacks.body(path)[1], raw_pairs.tags(path)["camera"]
    return Shot(photo, path, mosaic, raw_develop.look(path), flip, red, iso, camera, rivals.Shot(iso, from_xyz, black))


def crop(photo, x, y, *note):
    crops = json.loads(CROPS.read_text()) if CROPS.exists() else []
    taken = shot(photo)
    inside = lambda start, length: int(np.clip(start, MARGIN, length - SIDE - MARGIN)) // 2 * 2
    top, left = (inside(float(at) * length - SIDE / 2, length) for at, length in zip((y, x), taken.mosaic.shape))
    name = hashlib.sha1(f"{photo} {top} {left}".encode()).hexdigest()[:8]
    place = {"id": name, "photo": photo, "top": top, "left": left}
    crops.append({**place, "camera": taken.camera, "iso": taken.iso, "note": " ".join(note)})
    CROPS.parent.mkdir(parents=True, exist_ok=True)
    CROPS.write_text(json.dumps(crops, indent=1))
    print(name, photo, top, left)


def upright(picture, flip):
    return cv2.rotate(picture, raw_develop.TURNS[flip]) if flip in raw_develop.TURNS else picture


def camera_jpeg(taken, top, left):
    """The camera's own JPEG of a crop, from the preview its raw carries, or None when that is too small to show.

    A DNG says which part of the frame its preview is. Elsewhere a preview about the frame's size is
    taken to be the frame's middle, and a smaller one the whole frame.
    """
    with rawpy.imread(str(taken.path)) as raw:
        height, width = raw.sizes.height, raw.sizes.width
        try:
            preview = raw.extract_thumb()
        except rawpy.LibRawError:
            return None
    if preview.format != rawpy.ThumbFormat.JPEG:
        return None
    picture = cv2.imdecode(np.frombuffer(preview.data, np.uint8), cv2.IMREAD_COLOR)
    tall, wide = picture.shape[:2]
    if wide < SMALLEST_PREVIEW or (wide > tall) != (width > height):
        return None
    if abs(wide - width) < width / 50:
        across, down, scale = (width - wide) / 2, (height - tall) / 2, 1.0
    else:
        across, down = raw_develop.tag(taken.path, "DefaultCropOrigin", [0, 0])
        scale = wide / raw_develop.tag(taken.path, "DefaultCropSize", [width, height])[0]
    x, y = (left + taken.red[1] - across) * scale, (top + taken.red[0] - down) * scale
    onto = np.array([[1 / scale, 0, -x / scale], [0, 1 / scale, -y / scale]])
    return upright(cv2.warpAffine(picture, onto, (SIDE, SIDE), flags=cv2.INTER_CUBIC), taken.flip)


def ours(weights):
    model = train_raw.trained(weights)
    if not train_raw.joint(model):
        sys.exit(f"{weights} hands back a mosaic: the app develops a joint model's picture, so the sheet shows those")
    return lambda taken, window: train_raw.denoise(model, taken.mosaic, window=window)


def rival(name):
    clean_up = rivals.PICTURES[name]()

    def run(taken, window):
        top, left, height, width = window
        around = taken.mosaic[top - MARGIN : top + height + MARGIN, left - MARGIN : left + width + MARGIN]
        mean = float(taken.mosaic.mean())
        rivals.shots()[mean] = taken.rival
        picture = clean_up(torch.from_numpy(around.copy()).cuda()[None], mean)[0].float().cpu().numpy()
        return picture[:, MARGIN:-MARGIN, MARGIN:-MARGIN]

    return run


def program(folder):
    @functools.lru_cache(1)
    def whole(photo):
        taken = shot(photo)
        made = kit.returned(folder, taken.path)
        if not made:
            return None
        picture = torch.from_numpy(kit.placed(kit.linear(made), taken.mosaic, made))[None]
        return kit.fitted(picture, torch.from_numpy(taken.mosaic)[None])[0][0].numpy()

    def run(taken, window):
        top, left, height, width = window
        picture = whole(taken.photo)
        cut = None if picture is None else picture[:, top : top + height, left : left + width]
        return cut if cut is not None and np.isfinite(cut).all() else None

    return run


def entrant(spec):
    """An entrant's name, and what makes its camera RGB (3, SIDE, SIDE) of a window of a photo, or None."""
    name, _, source = spec.partition("=")
    if source and pathlib.Path(source).is_dir():
        return name, program(source)
    return name, ours(source) if source else rival(name)


def hidden(crop, name, suffix=".png"):
    return hashlib.sha1(f"{crop} {name}".encode()).hexdigest()[:12] + suffix


def sheet():
    file = SHEET / "sheet.json"
    return json.loads(file.read_text()) if file.exists() else {"models": [], "first": []}


def save(models, first):
    """sheet.json as the folder stands: every crop, and each entrant's picture of it that exists."""
    there = lambda file: file if (SHEET / file).exists() else None
    shown = lambda cut: {key: cut[key] for key in ("id", "camera", "iso", "note")}
    crops = [
        shown(cut) | {"reference": there(hidden(cut["id"], "camera", ".jpg"))} for cut in json.loads(CROPS.read_text())
    ]
    images = {
        cut["id"]: {name: hidden(cut["id"], name) for name in models if there(hidden(cut["id"], name))} for cut in crops
    }
    found = {"crops": crops, "models": models, "images": images, "first": first}
    (SHEET / "sheet.tmp").write_text(json.dumps(found, indent=1))
    (SHEET / "sheet.tmp").rename(SHEET / "sheet.json")
    if not PICKS.exists():
        PICKS.write_text(json.dumps({"picks": {}}) + "\n")


@torch.no_grad()
def add(specs):
    SHEET.mkdir(parents=True, exist_ok=True)
    crops = sorted(json.loads(CROPS.read_text()), key=lambda cut: cut["photo"])
    for cut in crops:
        reference = SHEET / hidden(cut["id"], "camera", ".jpg")
        picture = None if reference.exists() else camera_jpeg(shot(cut["photo"]), cut["top"], cut["left"])
        if picture is not None:
            cv2.imwrite(str(reference), picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
    models, first = sheet()["models"], sheet()["first"]
    for name, make in map(entrant, specs):
        models = [*models, *([] if name in models else [name])]
        for cut in crops:
            out = SHEET / hidden(cut["id"], name)
            if out.exists():
                continue
            taken = shot(cut["photo"])
            picture = make(taken, (cut["top"], cut["left"], SIDE, SIDE))
            if picture is None:
                print(name, "has nothing for", cut["photo"], flush=True)
                continue
            developed = upright(raw_develop.developed(picture, *taken.look), taken.flip)
            cv2.imwrite(str(out), cv2.cvtColor(developed, cv2.COLOR_RGB2BGR))
            print(name, cut["photo"], cut["note"], flush=True)
        save(models, first)
    save(models, first)


def table():
    """Each pair's picks: how often each of the two was picked, and how often neither."""
    tally = collections.Counter()
    for key, pick in json.loads(PICKS.read_text())["picks"].items():
        tally[(*key.split("|")[1:], pick["pick"])] += 1
    pairs = sorted({(a, b) for a, b, _ in tally})
    wins = collections.Counter()
    for a, b in pairs:
        print(f"{a:>20} {tally[a, b, a]:3} : {tally[a, b, b]:<3} {b:20} same {tally[a, b, 'same']}")
        wins[a] += tally[a, b, a]
        wins[b] += tally[a, b, b]
    print()
    for name, won in wins.most_common():
        shown = sum(count for (a, b, _), count in tally.items() if name in (a, b))
        print(f"{name:>20} picked {won} of {shown}")


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    if command == "crop":
        crop(*rest)
    elif command == "add":
        add(rest)
    elif command == "first":
        save(sheet()["models"], [spec.split(":") for spec in rest])
    else:
        table()
