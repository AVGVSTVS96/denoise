"""A fixed test for raw denoisers: Imaging Resource's still life, ISO 3,200 and up, on many cameras.

    uv run bench_raw.py build                                          cut the windows once, into data/bench
    uv run bench_raw.py faithful=data/models/raw/faithful/faithful.pth ...    score models next to the noisy frames,
                                                                               added to data/out/bench.json

Each camera gives 32 windows of 512 px, the same at every ISO, against its ISO 100 frame. A cleaned
mosaic and the clean one are developed alike (bilinear demosaic, the camera's white balance and colour
matrix, gamma 2.2) and compared as pictures: PSNR, and LPIPS, the perceptual distance (lower is better).
No model trains on this scene. The GR III shares the GR IIIx's sensor, so it picks between runs
(train_raw.py logs it); the GR IIIx stays the test.

Two groups of cameras from fetch_ladders.py's still lifes widen it past the Ricohs, 16 windows each:
"unseen" are cameras raw_pairs.py keeps out of training whole, the test of a model meant for any
camera, and "seen" are cameras whose other scenes do train, for picking between runs. A group scores
as one camera would: each ISO's mean over the cameras that have it.
"""

import functools
import json
import sys

import numpy as np
import rawpy
import torch

import raw_pairs

BENCH = raw_pairs.DATA / "bench"
CAMERAS = {"gr3": "ricoh-gr-iii--GR3hSLI{:06d}NR0.DNG", "gr3x": "ricoh-gr-iiix--GR3xhSLI{:06d}NR0.DNG"}
ISOS = (3200, 6400, 12800, 25600, 51200, 102400)
WINDOWS = 32
STILL_LIFES = raw_pairs.LADDERS / "ir"
SEEN = [
    "canon-eos-5d-mark-iv",
    "canon-eos-80d",
    "canon-eos-rp",
    "canon-eos-m50",
    "nikon-d750",
    "nikon-d850",
    "nikon-d5600",
    "nikon-z7",
    "sony-alpha-ilce-a7-iii",
    "sony-alpha-ilce-a7r-iv",
    "sony-alpha-ilce-a6500",
    "sony-cyber-shot-dsc-rx100-iv",
    "olympus-om-d-e-m5-ii",
    "panasonic-lumix-dmc-gx8",
    "pentax-k-70",
    "samsung-nx1",
]
#: Pixels dropped from each side of a window before scoring, where a model has no neighbours to look at.
EDGE = 16


def groups():
    """Each group's cameras, as the still-life folders on disk."""
    folders = sorted(folder for folder in STILL_LIFES.glob("*/sli") if any(folder.iterdir()))
    held = lambda folder: any(name in folder.parent.name for name in raw_pairs.HELD_OUT)
    return {
        "unseen": [folder for folder in folders if held(folder)],
        "seen": [folder for folder in folders if folder.parent.name in SEEN],
    }


def build(camera, path, base=100, isos=ISOS, count=WINDOWS):
    clean = raw_pairs.mosaic(path(base))
    picks = raw_pairs.corners(clean, np.random.default_rng(0), count)
    cut = lambda frame, dx=0, dy=0: np.stack(
        [frame[y + dy : y + dy + raw_pairs.CROP, x + dx : x + dx + raw_pairs.CROP] for y, x in picks]
    )
    frames, means = [cut(clean)], []
    for iso in isos:
        noisy = raw_pairs.mosaic(path(iso))
        dx, dy = raw_pairs.shift_between(clean, noisy)
        noisy, _ = raw_pairs.match(clean, noisy)
        frames.append(cut(noisy, dx, dy))
        means.append(float(noisy.mean()))
        print(camera, iso, "shift", dx, dy, flush=True)
    with rawpy.imread(str(path(base))) as raw:
        wb = np.asarray(raw.camera_whitebalance[:3], np.float32)
        matrix = np.asarray(raw.color_matrix[:, :3], np.float32)
    BENCH.mkdir(parents=True, exist_ok=True)
    frames = np.stack(frames).astype(np.float16)
    np.savez(BENCH / f"{camera}.npz", frames=frames, means=means, wb=wb / wb[1], matrix=matrix, isos=isos)


def build_group(group, folder):
    steps = {int(file.stem): file for file in folder.iterdir() if file.stem.isdigit()}
    isos = [iso for iso in ISOS if iso in steps]
    if raw_pairs.mosaic(steps[min(steps)]) is not None and isos:
        build(f"{group}--{folder.parent.name}", steps.get, min(steps), isos, WINDOWS // 2)


def develop(mosaic, wb, matrix):
    """RGGB mosaics (N, H, W) as linear sRGB (N, 3, H, W): bilinear demosaic, white balance, colour matrix."""
    sites = mosaic.new_zeros(3, *mosaic.shape[1:])
    sites[0, 0::2, 0::2] = sites[1, 0::2, 1::2] = sites[1, 1::2, 0::2] = sites[2, 1::2, 1::2] = 1
    cross = mosaic.new_tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    full = mosaic.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    planes = torch.nn.functional.pad(mosaic[:, None] * sites, (1, 1, 1, 1), mode="reflect")
    rgb = torch.nn.functional.conv2d(planes, torch.stack([full, cross, full])[:, None], groups=3)
    return torch.einsum("ij,njhw->nihw", matrix, rgb * wb.view(1, 3, 1, 1))


def windows(camera):
    """Frames (1 + ISOs, windows, 512, 512) with the clean one first, their ISOs and means, and the colour."""
    data = np.load(BENCH / f"{camera}.npz")
    return (
        torch.from_numpy(data["frames"].astype(np.float32)).cuda(),
        data["isos"].tolist(),
        data["means"].tolist(),
        torch.from_numpy(data["wb"]).cuda(),
        torch.from_numpy(data["matrix"]).cuda(),
    )


#: The Ricohs stay on the GPU between a run's checks; a group's cameras come and go.
kept = functools.cache(windows)


@functools.cache
def perceptual():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cuda()


@torch.no_grad()
def score(clean_up, camera, with_lpips=False):
    """{ISO: [PSNR, LPIPS]} for clean_up(mosaics (N, H, W), their frame's mean) -> cleaned mosaics."""
    if camera not in CAMERAS:
        each = [score_one(clean_up, windows(file.stem), with_lpips) for file in sorted(BENCH.glob(f"{camera}--*.npz"))]
        had = {iso: [by_iso[iso] for by_iso in each if iso in by_iso] for iso in ISOS}
        return {iso: np.mean(scores, axis=0).tolist() for iso, scores in had.items() if scores}
    return score_one(clean_up, kept(camera), with_lpips)


def score_one(clean_up, cut, with_lpips):
    frames, isos, means, wb, matrix = cut
    linear = develop(frames[0], wb, matrix)
    exposure = 0.108 / linear.median()
    picture = lambda lin: ((lin * exposure).clamp(0, 1) ** (1 / 2.2))[..., EDGE:-EDGE, EDGE:-EDGE]
    truth, scores = picture(linear), {}
    for iso, noisy, mean in zip(isos, frames[1:], means):
        result = picture(develop(clean_up(noisy, mean), wb, matrix))
        psnr = -10 * torch.log10(((result - truth) ** 2).mean(dim=(1, 2, 3)))
        distance = perceptual()(result * 2 - 1, truth * 2 - 1) if with_lpips else torch.full((1,), float("nan"))
        scores[iso] = [psnr.mean().item(), distance.mean().item()]
    return scores


def main(models):
    import train_raw

    board = raw_pairs.DATA / "out" / "bench.json"
    table = json.loads(board.read_text()) if board.exists() else {}
    for camera in (*CAMERAS, *groups()):
        table.setdefault(camera, {})["noisy"] = score(lambda noisy, mean: noisy, camera, with_lpips=True)
        for spec in models:
            name, weights = spec.split("=")
            model = train_raw.network().cuda().eval()
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            table[camera][name] = score(train_raw.cleaner(model), camera, with_lpips=True)
        print(f"\n{camera}  PSNR / LPIPS\n{'':24}" + "".join(f"{iso:>14}" for iso in ISOS) + f"{'mean':>14}")
        for name, by_iso in table[camera].items():
            cells = [*by_iso.values(), np.mean(list(by_iso.values()), axis=0)]
            print(f"{name:24}" + "".join(f"{p:8.2f} {d:.3f}" for p, d in cells))
    board.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        for camera, pattern in CAMERAS.items():
            build(camera, lambda iso, pattern=pattern: raw_pairs.NIGHT / "imaging-resource" / pattern.format(iso))
        for group, folders in groups().items():
            for folder in folders:
                build_group(group, folder)
    else:
        main(sys.argv[1:])
