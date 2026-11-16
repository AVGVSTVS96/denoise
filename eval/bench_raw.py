"""A fixed test for raw denoisers: the Ricoh GR III and GR IIIx still-life ladders, ISO 3,200 to 102,400.

    uv run bench_raw.py build                                          cut the windows once, into data/bench
    uv run bench_raw.py faithful=data/models/raw/faithful/faithful.pth ...    score models next to the noisy frames,
                                                                               added to data/out/bench.json

Each camera gives 32 windows of 512 px, the same at every ISO, against its ISO 100 frame. A cleaned
mosaic and the clean one are developed alike (bilinear demosaic, the camera's white balance and colour
matrix, gamma 2.2) and compared as pictures: PSNR, and LPIPS, the perceptual distance (lower is better).
No model trains on this scene. The GR III shares the GR IIIx's sensor, so it picks between runs
(train_raw.py logs it); the GR IIIx stays the test.
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
#: Pixels dropped from each side of a window before scoring, where a model has no neighbours to look at.
EDGE = 16


def build(camera, pattern):
    path = lambda iso: raw_pairs.NIGHT / "imaging-resource" / pattern.format(iso)
    clean = raw_pairs.mosaic(path(100))
    picks = raw_pairs.corners(clean, np.random.default_rng(0), WINDOWS)
    cut = lambda frame, dx=0, dy=0: np.stack(
        [frame[y + dy : y + dy + raw_pairs.CROP, x + dx : x + dx + raw_pairs.CROP] for y, x in picks]
    )
    frames, means = [cut(clean)], []
    for iso in ISOS:
        noisy = raw_pairs.mosaic(path(iso))
        dx, dy = raw_pairs.shift_between(clean, noisy)
        noisy, _ = raw_pairs.match(clean, noisy)
        frames.append(cut(noisy, dx, dy))
        means.append(float(noisy.mean()))
        print(camera, iso, "shift", dx, dy, flush=True)
    with rawpy.imread(str(path(100))) as raw:
        wb = np.asarray(raw.camera_whitebalance[:3], np.float32)
        matrix = np.asarray(raw.color_matrix[:, :3], np.float32)
    BENCH.mkdir(parents=True, exist_ok=True)
    np.savez(
        BENCH / f"{camera}.npz", frames=np.stack(frames).astype(np.float16), means=means, wb=wb / wb[1], matrix=matrix
    )


def develop(mosaic, wb, matrix):
    """RGGB mosaics (N, H, W) as linear sRGB (N, 3, H, W): bilinear demosaic, white balance, colour matrix."""
    sites = mosaic.new_zeros(3, *mosaic.shape[1:])
    sites[0, 0::2, 0::2] = sites[1, 0::2, 1::2] = sites[1, 1::2, 0::2] = sites[2, 1::2, 1::2] = 1
    cross = mosaic.new_tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    full = mosaic.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    planes = torch.nn.functional.pad(mosaic[:, None] * sites, (1, 1, 1, 1), mode="reflect")
    rgb = torch.nn.functional.conv2d(planes, torch.stack([full, cross, full])[:, None], groups=3)
    return torch.einsum("ij,njhw->nihw", matrix, rgb * wb.view(1, 3, 1, 1))


@functools.cache
def windows(camera):
    """Frames (1 + ISOs, windows, 512, 512) with the clean one first, each noisy frame's mean, and the colour."""
    data = np.load(BENCH / f"{camera}.npz")
    return (
        torch.from_numpy(data["frames"].astype(np.float32)).cuda(),
        data["means"].tolist(),
        torch.from_numpy(data["wb"]).cuda(),
        torch.from_numpy(data["matrix"]).cuda(),
    )


@functools.cache
def perceptual():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cuda()


@torch.no_grad()
def score(clean_up, camera, with_lpips=False):
    """{ISO: [PSNR, LPIPS]} for clean_up(mosaics (N, H, W), their frame's mean) -> cleaned mosaics."""
    frames, means, wb, matrix = windows(camera)
    linear = develop(frames[0], wb, matrix)
    exposure = 0.108 / linear.median()
    picture = lambda lin: ((lin * exposure).clamp(0, 1) ** (1 / 2.2))[..., EDGE:-EDGE, EDGE:-EDGE]
    truth, scores = picture(linear), {}
    for iso, noisy, mean in zip(ISOS, frames[1:], means):
        result = picture(develop(clean_up(noisy, mean), wb, matrix))
        psnr = -10 * torch.log10(((result - truth) ** 2).mean(dim=(1, 2, 3)))
        distance = perceptual()(result * 2 - 1, truth * 2 - 1) if with_lpips else torch.full((1,), float("nan"))
        scores[iso] = [psnr.mean().item(), distance.mean().item()]
    return scores


def main(models):
    import train_raw

    board = raw_pairs.DATA / "out" / "bench.json"
    table = json.loads(board.read_text()) if board.exists() else {}
    for camera in CAMERAS:
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
            build(camera, pattern)
    else:
        main(sys.argv[1:])
