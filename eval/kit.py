"""DxO PureRAW and Lightroom on bench_raw.py's test: the raws a person runs them over, and what comes back, scored.

    uv run kit.py folder OUT [PHOTO ...]                           the raws to hand over, into OUT
    uv run kit.py dxo=FOLDER joint-v2=WEIGHTS.pth rawforge-jdd ...  scored, into data/out/kit.json

Neither program runs on a server, so a person runs them. folder copies out the still lifes of the
Ricohs and of four unseen cameras, at ISOS and each camera's highest test ISO, named <camera>--<ISO>,
and any other photos asked for (blind.py's) under their own names. What a program made of a raw comes
back in a folder of its own, under the raw's name with whatever the program added to it.

What comes back is a linear DNG: a picture, demosaiced and denoised, on a scale and in a frame of its
own. placed() finds where the raw's mosaic sits in it, in whole pixels: a picture that was
straightened, sharpened into another size or turned does not fit, and stops the run. fitted() then
puts each colour on the test frame's scale, as raw_pairs.match did for the noisy frame itself, so a
program's own black level, gain and white balance cost it nothing. Each colour is fitted as a mix of
the picture's three, so a picture that left the camera's own colour is brought back to it, and the
run prints how much mixing that took: near none for a picture that stayed.

Each entrant is scored on the aligned test's windows of the kit's cameras and ISOs, as bench_raw.py
scores them, in two rows as rivals.py gives a rival that hands back a picture: "<name>" is its own
picture, and "<name>-cfa" that picture sampled back to a mosaic, so it goes through the same bilinear
demosaic as the clean frame and only its denoising is scored. A model of ours or a rival by its
rivals.py name is scored beside them on the same windows, since the board's rows are over more
cameras and ISOs than a person is asked to run.
"""

import functools
import json
import pathlib
import re
import shutil
import sys

import cv2
import numpy as np
import torch

import bench_raw
import raw_pairs

CAMERAS = (
    "gr3",
    "gr3x",
    "unseen--canon-eos-r6",
    "unseen--sony-alpha-ilce-a7-iv",
    "unseen--nikon-z6-ii",
    "unseen--pentax-k-1-ii",
)
ISOS = (3200, 12800, 51200)
BOARD = raw_pairs.DATA / "out" / "kit.json"
LINEAR_RAW = 34892
FIELD = 2048
BLOCK = 64


def ladder(camera):
    """A kit camera's raws by ISO, its lowest ISO, the kit's ISOs among them, and the name its files go by."""
    if camera in bench_raw.CAMERAS:
        path, base, isos = bench_raw.ricoh(camera), 100, bench_raw.ISOS
        name = bench_raw.CAMERAS[camera].split("--")[0]
    else:
        name = camera.split("--")[1]
        path, base, isos = bench_raw.ladder(bench_raw.STILL_LIFES / name / "sli")
    return path, base, [iso for iso in isos if iso in ISOS or iso == max(isos)], name


def folder(out, photos):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        path, _, isos, name = ladder(camera)
        for iso in isos:
            shutil.copy2(path(iso), out / f"{name}--{iso}{path(iso).suffix}")
    for photo in map(pathlib.Path, photos):
        shutil.copy2(photo, out / photo.name)
    print(len(list(out.iterdir())), "files in", out)


def returned(folder, raw):
    """What a program made of a raw: the one DNG in folder under the raw's name with whatever was added to it."""
    named = re.compile(re.escape(pathlib.Path(raw).stem) + r"(?![0-9a-z])", re.IGNORECASE)
    found = [file for file in pathlib.Path(folder).glob("*.[dD][nN][gG]") if named.match(file.name)]
    return found[0] if len(found) == 1 else None


def linear(path):
    """A linear DNG's picture (3, H, W) as stored, black at 0 and white at 1: no white balance, no crop."""
    import tifffile

    with tifffile.TiffFile(path) as file:
        pages = [page for first in file.pages for page in (first, *(first.pages or ()))]
        raws = [page for page in pages if page.photometric == LINEAR_RAW and page.samplesperpixel == 3]
        if not raws:
            sys.exit(f"{path} holds no linear picture: it is a mosaic, or was not saved as a linear DNG")
        page = max(raws, key=lambda page: page.imagelength * page.imagewidth)
        level = lambda tag, missing: np.asarray(page.tags[tag].value if tag in page.tags else missing, np.float32)
        black, white = level("BlackLevel", 0).reshape(-1), level("WhiteLevel", 65535).reshape(-1)
        return ((page.asarray().astype(np.float32) - black) / (white - black)).transpose(2, 0, 1)


def placed(picture, mosaic, path=""):
    """A picture (3, H, W) cut to its raw's mosaic (H, W), pixel on pixel, and nothing where it does not reach.

    Both are compared by their greens over a field at their middles, which finds how far apart the
    two frames start. A whole number of pixels is a crop; anything else is a picture that was moved.
    """
    side = min(FIELD, *mosaic.shape, *picture.shape[1:]) // 2 * 2
    start = lambda shape: [(length - side) // 4 * 2 for length in shape]
    (my, mx), (py, px) = start(mosaic.shape), start(picture.shape[1:])
    seen = bench_raw.bilinear(torch.from_numpy(mosaic[my : my + side, mx : mx + side].copy())[None])[0, 1].numpy()
    smooth = lambda field: cv2.GaussianBlur(np.ascontiguousarray(field, np.float32), (0, 0), 1.5)
    theirs = smooth(picture[1, py : py + side, px : px + side])
    (dx, dy), found = cv2.phaseCorrelate(smooth(seen), theirs, cv2.createHanningWindow((side, side), cv2.CV_32F))
    if max(abs(dx - round(dx)), abs(dy - round(dy))) > 0.2 or found < 0.1:
        sys.exit(f"{path} is not on its raw's pixels ({dx:.2f}, {dy:.2f} px off, match {found:.2f}): corrections on?")
    down, across = py - my + round(dy), px - mx + round(dx)
    out = np.full((3, *mosaic.shape), np.nan, np.float32)
    rows = slice(max(0, -down), min(mosaic.shape[0], picture.shape[1] - down))
    columns = slice(max(0, -across), min(mosaic.shape[1], picture.shape[2] - across))
    theirs = slice(rows.start + down, rows.stop + down), slice(columns.start + across, columns.stop + across)
    out[:, rows, columns] = picture[(slice(None), *theirs)]
    return out


def mix(pictures, frames):
    """What puts pictures (N, 3, H, W) on the scale of the noisy mosaics (N, H, W) they were made of: (3, 4),
    each colour as the best mix of the picture's three and a lift.

    Fitted through the means of BLOCK px blocks, each colour against the mosaic at its own sites. A
    picture in the camera's own colour needs a gain and a lift a colour, and what it takes from the
    other two says how far from that colour it is.
    """
    sites = {0: [(0, 0)], 1: [(0, 1), (1, 0)], 2: [(1, 1)]}
    blocks = lambda planes: torch.nn.functional.avg_pool2d(planes, BLOCK // 2).flatten(-2).transpose(-1, -2)
    rows = []
    for places in sites.values():
        theirs = torch.cat([blocks(pictures[:, :, i::2, j::2]).reshape(-1, 3) for i, j in places]).double()
        ours = torch.cat([blocks(frames[:, None, i::2, j::2]).reshape(-1) for i, j in places]).double()
        fair = (torch.isfinite(theirs).all(1) & (ours < 0.5)).cpu().numpy()
        known = np.concatenate([theirs.cpu().numpy(), np.ones((len(ours), 1))], 1)[fair]
        rows.append(np.linalg.lstsq(known, ours.cpu().numpy()[fair], rcond=None)[0])
    return np.array(rows, np.float32)


def fitted(pictures, frames):
    """Pictures with each colour on its noisy mosaics' scale, and the share of that colour taken from the other two."""
    matrix = torch.from_numpy(mix(pictures, frames)).to(pictures)
    out = torch.einsum("ck,nkhw->nchw", matrix[:, :3], pictures) + matrix[:, 3].view(1, 3, 1, 1)
    return out, float(1 - matrix[:, :3].diagonal().abs().sum() / matrix[:, :3].abs().sum())


@functools.cache
def corners(camera):
    """Where a kit camera's test windows were cut: its clean frame and their top-left corners, as bench_raw.build."""
    path, base, _, _ = ladder(camera)
    clean = raw_pairs.mosaic(path(base))
    count = bench_raw.WINDOWS // (1 if camera in bench_raw.CAMERAS else 2)
    return clean, raw_pairs.corners(clean, np.random.default_rng(0), count)


def cutter(camera, iso):
    """A test's noisy frame at an ISO as its raw holds it, and what cuts the test's windows out of a frame that size."""
    clean, picks = corners(camera)
    noisy = raw_pairs.mosaic(ladder(camera)[0](iso))
    dx, dy = raw_pairs.shift_between(clean, noisy)
    side = raw_pairs.CROP
    cut = lambda frame: torch.from_numpy(
        np.stack([frame[..., y + dy : y + dy + side, x + dx : x + dx + side] for y, x in picks])
    )
    return noisy, cut


def program(folder):
    """A cleaner out of a folder of linear DNGs: (camera, ISO, the test's noisy windows) -> pictures, or None."""

    def clean_up(camera, iso, noisy, mean):
        made = returned(folder, f"{ladder(camera)[3]}--{iso}")
        if not made:
            return None
        mosaic, cut = cutter(camera, iso)
        reds = lambda frames: frames[:, 0::2, 0::2].flatten().float()
        if torch.corrcoef(torch.stack([reds(cut(mosaic)), reds(noisy.cpu())]))[0, 1] < 0.999:
            sys.exit(f"{camera} at ISO {iso}: the test's windows are not where bench_raw.build cut them")
        pictures = cut(placed(linear(made), mosaic, made)).to(noisy)
        if not torch.isfinite(pictures).all():
            sys.exit(f"{made} does not reach every test window: cropped?")
        result, mixed = fitted(pictures, noisy)
        print(f"{made.name}: {mixed:.1%} of each colour comes from the other two", flush=True)
        return result

    return clean_up


def entrant(spec):
    """An entrant's name and its cleaner: a folder of DNGs, a model of ours, or a rival by its rivals.py name."""
    import rivals
    import train_raw

    name, _, source = spec.partition("=")
    if source and pathlib.Path(source).is_dir():
        return name, program(source)
    clean_up = train_raw.cleaner(train_raw.trained(source)) if source else rivals.rows(name)[1]
    return name, lambda camera, iso, noisy, mean: clean_up(noisy, mean)


@torch.no_grad()
def scores(camera, clean_up):
    """{row: {ISO: [PSNR, LPIPS]}} on a kit camera's aligned test, for the entrant's picture and for its mosaic."""
    frames, isos, means, wb, matrix, twins, offsets, _ = bench_raw.windows(camera)
    found = {}
    for iso in ladder(camera)[2]:
        k = isos.index(iso)
        result = clean_up(camera, iso, frames[k + 1], means[k])
        if result is None:
            continue
        cut = [frames[0], frames[k + 1]], [iso], [means[k]], wb, matrix, twins[k : k + 1], offsets[k : k + 1], None
        rows = {"": result} | ({"-cfa": bench_raw.mosaiced(result)} if result.dim() == 4 else {})
        for row, shown in rows.items():
            aligned = bench_raw.score_one(lambda *_, shown=shown: shown, cut, with_lpips=True)[1]
            found.setdefault(row, {})[iso] = aligned[iso]
    return found


def main(specs):
    table = json.loads(BOARD.read_text()) if BOARD.exists() else {}
    for name, clean_up in map(entrant, specs):
        for camera in CAMERAS:
            for row, by_iso in scores(camera, clean_up).items():
                table.setdefault(camera, {})[name + row] = by_iso
                BOARD.with_suffix(".tmp").write_text(json.dumps(table, indent=1))
                BOARD.with_suffix(".tmp").rename(BOARD)
    show(json.loads(BOARD.read_text()))


def show(table):
    """Each camera's rows, PSNR / LPIPS by ISO, then every row's mean over the cells all rows have."""
    for camera, rows in table.items():
        isos = ladder(camera)[2]
        print(f"\n{camera}  PSNR / LPIPS\n{'':24}" + "".join(f"{iso:>14}" for iso in isos))
        for name, by_iso in rows.items():
            cells = [by_iso.get(str(iso)) for iso in isos]
            print(f"{name:24}" + "".join(f"{c[0]:8.2f} {c[1]:.3f}" if c else f"{'':14}" for c in cells))
    names = sorted({name for rows in table.values() for name in rows})
    cells = {name: {(c, i): v for c, rows in table.items() for i, v in rows.get(name, {}).items()} for name in names}
    shared = set.intersection(*(set(found) for found in cells.values())) if cells else set()
    print(f"\nmean over the {len(shared)} camera and ISO cells every row has")
    for name in names:
        psnr, lpips = np.mean([cells[name][cell] for cell in shared], axis=0) if shared else (float("nan"),) * 2
        print(f"{name:24}{psnr:8.2f} {lpips:.3f}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["folder"]:
        folder(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1:])
