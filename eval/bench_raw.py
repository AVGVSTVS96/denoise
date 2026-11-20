"""A fixed test for raw denoisers: Imaging Resource's still life, ISO 3,200 and up, on many cameras.

    uv run bench_raw.py build                                          cut the windows once, into data/bench
    uv run bench_raw.py faithful=data/models/raw/faithful/faithful.pth ...    score models next to the noisy frames,
                                                                               added to data/out/bench.json

A model is a faithful one, mosaic in and mosaic out, or a joint one, which demosaics too and hands
back camera RGB (train_raw.py --joint). On the tests with real noise a joint model's picture is
sampled back to a mosaic first, so both kinds are scored on their denoising alone, alike.

Each camera gives 32 windows of 512 px, the same at every ISO, against its ISO 100 frame. A cleaned
mosaic and the clean one are developed alike (bilinear demosaic, the camera's white balance and colour
matrix, gamma 2.2) and compared as pictures: PSNR, and LPIPS, the perceptual distance (lower is better).
No model trains on this scene. The GR III shares the GR IIIx's sensor, so it picks between runs
(train_raw.py logs it); the GR IIIx stays the test.

Two groups of cameras from fetch_ladders.py's still lifes widen it past the Ricohs, 16 windows each:
"unseen" are cameras raw_pairs.py keeps out of training whole, the test of a model meant for any
camera, and "seen" are cameras whose other scenes do train, for picking between runs. A group scores
as one camera would: each ISO's mean over the cameras that have it.

Every test scores three times. The first score, under the test's own name, is against the ISO 100
window as it was cut: whole even pixels from the noisy one, by raw_pairs.shift_between, which leaves
most cameras a fraction of a pixel off and some two pixels, and no denoiser can score past that. The
second, "<test>-aligned", is against the same window resampled onto the noisy one (raw_pairs.offsets),
and is the one to read; the first stays so old runs compare. In both the noisy frame's black offset
was taken off with the clean frame's help (raw_pairs.match), which a real photo never gets. The
third, "<test>-as-shot", puts that offset back before the denoiser sees the frame and scores against
the resampled window: what a photo straight from the camera would get.

"exact" has no such limit at all: the unseen cameras' ISO 100 frames binned three to one, which
leaves a ninth of their noise, under noise made up at the levels measured from each camera's own high
ISO frames. What it lacks is the camera's real noise, so it ranks models and shows what a test costs,
and "<test>-aligned" stays the test of real photos.

All of those develop both sides with the same bilinear demosaic, so none can tell a good demosaic
from a poor one. "<test>-colour" can: each camera's lowest ISO frame binned to full colour at half the
size (full_colour), which is a picture no demosaic touched, sampled back to a mosaic and put under
made-up noise as "exact" is. A model that demosaics is scored on its own picture, and a model that
hands back a mosaic twice: through the bilinear demosaic, and as "<name>-menon" through a good
classical one (demosaic.py), white-balanced first. Its "unseen" cameras are the "exact" ones, every scene of theirs. It has
no real noise either: a real noisy frame has no full-colour truth at its own size.

    uv run bench_raw.py colour                                         cut only those, into data/bench/colour
"""

import functools
import json
import sys

import cv2
import numpy as np
import rawpy
import torch

import raw_pairs

BENCH = raw_pairs.DATA / "bench"
CAMERAS = {"gr3": "ricoh-gr-iii--GR3hSLI{:06d}NR0.DNG", "gr3x": "ricoh-gr-iiix--GR3xhSLI{:06d}NR0.DNG"}
ISOS = (3200, 6400, 12800, 25600, 51200, 102400)
WINDOWS = 32
STILL_LIFES = raw_pairs.LADDERS / "ir"
XYZ_FROM_SRGB = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    np.float32,
)
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


def ricoh(camera):
    """Where a Ricoh's still life at an ISO is."""
    return lambda iso: raw_pairs.NIGHT / "imaging-resource" / CAMERAS[camera].format(iso)


def colour(path):
    """A raw's white balance against green, and its matrix from camera colour to sRGB."""
    with rawpy.imread(str(path)) as raw:
        wb = np.asarray(raw.camera_whitebalance[:3], np.float32)
        matrix = np.asarray(raw.color_matrix[:, :3], np.float32)
        if not matrix.any():
            # Only some raw formats carry that matrix; for the rest LibRaw knows the camera's colour against XYZ.
            to_camera = np.asarray(raw.rgb_xyz_matrix[:3], np.float32) @ XYZ_FROM_SRGB
            matrix = np.linalg.inv(to_camera / to_camera.sum(axis=1, keepdims=True))
    return wb / wb[1], matrix


def save(name, **arrays):
    """A test's windows, renamed into place whole: the PC scores from this folder while a build runs."""
    (BENCH / name).parent.mkdir(parents=True, exist_ok=True)
    with (BENCH / f"{name}.tmp").open("wb") as file:
        np.savez(file, **arrays)
    (BENCH / f"{name}.tmp").rename(BENCH / f"{name}.npz")


def build(camera, path, base=100, isos=ISOS, count=WINDOWS):
    clean = raw_pairs.mosaic(path(base))
    picks = raw_pairs.corners(clean, np.random.default_rng(0), count)
    cut = lambda frame, dx=0, dy=0: np.stack(
        [frame[y + dy : y + dy + raw_pairs.CROP, x + dx : x + dx + raw_pairs.CROP] for y, x in picks]
    )
    frames, twins, means, offsets = [cut(clean)], [], [], []
    for iso in isos:
        noisy = raw_pairs.mosaic(path(iso))
        dx, dy = raw_pairs.shift_between(clean, noisy)
        noisy, _, blacks = raw_pairs.match(clean, noisy)
        offsets.append(blacks)
        frames.append(cut(noisy, dx, dy))
        at = raw_pairs.offsets(clean, noisy)
        left = np.array([at(y, x)[0] for y, x in picks]) - (dx, dy)
        whole = (np.round(left / 2) * 2).astype(int)
        twins.append(
            np.stack(
                [
                    raw_pairs.resampled(clean, y - wy, x - wx, fx, fy)
                    for (y, x), (wx, wy), (fx, fy) in zip(picks, whole, left - whole)
                ]
            )
        )
        means.append(float(noisy.mean()))
        print(camera, iso, "shift", dx, dy, "left", round(float(np.hypot(*left.T).mean()), 2), flush=True)
    wb, matrix = colour(path(base))
    stacked = lambda crops: np.stack(crops).astype(np.float16)
    save(
        camera,
        frames=stacked(frames),
        twins=stacked(twins),
        means=means,
        offsets=np.array(offsets, np.float32),
        wb=wb,
        matrix=matrix,
        isos=isos,
    )


def ladder(folder):
    """A still-life folder's raws by ISO, its lowest ISO, and the test's ISOs among them."""
    steps = {int(file.stem): file for file in folder.iterdir() if file.stem.isdigit()}
    return steps.get, min(steps), [iso for iso in ISOS if iso in steps]


def build_group(group, folder):
    path, base, isos = ladder(folder)
    if raw_pairs.mosaic(path(base)) is not None and isos:
        build(f"{group}--{folder.parent.name}", path, base, isos, WINDOWS // 2)


def binned(cfa):
    """A mosaic a third the size: each pixel the mean of the nine of its colour around every third one.

    Still RGGB with every colour where it belongs, and a third of the noise.
    """
    h, w = ((side - 2) // 6 * 2 for side in cfa.shape)
    return sum(cfa[a : a + 3 * h : 3, b : b + 3 * w : 3] for a in (0, 2, 4) for b in (0, 2, 4)) / 9


def noise_levels(clean, noisy):
    """A noisy frame's shot and read noise on the clean one's scale: variance = shot * brightness + read.

    From 8 px blocks of each colour plane where the clean frame is flat, their variance taken between
    neighbours so a slope does not count, and their brightness from the clean frame.
    """
    means, variances = [], []
    h, w = (side // 16 * 8 for side in clean.shape)
    blocks = lambda a: a[:h, :w].reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    for i in range(2):
        for j in range(2):
            flat, rough = clean[i::2, j::2], blocks(noisy[i::2, j::2])
            smooth = cv2.GaussianBlur(flat, (0, 0), 3)
            slope = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
            mean = blocks(flat).mean((1, 2))
            busy = blocks(slope).max((1, 2)) / (mean + 0.002)
            keep = (busy < np.percentile(busy, 25)) & (mean > 0.001) & (mean < 0.7)
            means.append(mean[keep])
            variances.append((np.diff(rough, axis=2).var((1, 2)) + np.diff(rough, axis=1).var((1, 2)))[keep] / 4)
    mean, variance = np.concatenate(means), np.concatenate(variances)
    order = np.argsort(mean)
    medians = lambda values: [np.median(part) for part in np.array_split(values[order], 24)]
    shot, read = np.polyfit(medians(mean), medians(variance), 1)
    return max(float(shot), 1e-9), max(float(read), 0)


def build_exact(camera, path, base, isos):
    """Windows whose clean frame is the truth: the ISO 100 frame binned, under each high ISO's measured noise."""
    full = raw_pairs.mosaic(path(base))
    if full is None or not isos:
        return
    clean = binned(full)
    side = raw_pairs.CROP
    cut = lambda frame: np.stack(
        [
            frame[y : y + side, x : x + side]
            for y in range(0, frame.shape[0] - side + 1, side)
            for x in range(0, frame.shape[1] - side + 1, side)
        ]
    )
    frames, means = under_noise(camera, clean, full, path, isos, cut)
    wb, matrix = colour(path(base))
    frames = np.stack([cut(clean), *frames]).astype(np.float16)
    save(f"exact--{camera}", frames=frames, means=means, wb=wb, matrix=matrix, isos=isos)


def under_noise(camera, clean, full, path, isos, cut):
    """Windows of a clean mosaic under each high ISO's noise, as measured on the frame it was made from, and their means."""
    rng = np.random.default_rng(0)
    frames, means = [], []
    for iso in isos:
        shot, read = noise_levels(full, raw_pairs.match(full, raw_pairs.mosaic(path(iso)))[0])
        noisy = rng.poisson(clean.clip(min=0) / shot) * shot + rng.standard_normal(clean.shape) * read**0.5
        frames.append(cut(noisy.astype(np.float32)))
        means.append(float(noisy.mean()))
        print(camera, iso, "shot", f"{shot:.3g}", "read", f"{read**0.5:.3g}", flush=True)
    return frames, means


def full_colour(mosaics):
    """RGGB mosaics (N, H, W) as camera RGB at half the size (N, 3, H/2, W/2), every colour measured, none guessed.

    Each 2x2 cell becomes one pixel at the cell's middle. Its four samples sit half a sensor pixel off
    that middle, each its own way, so each colour plane is first read a quarter of its own pixel
    along (Lanczos) and the two greens are averaged. Taken as they are the colours would sit apart
    by that much, a fringe on every edge that a model would learn to draw.
    """
    taps = torch.arange(-4, 5, dtype=mosaics.dtype, device=mosaics.device)
    lanczos = lambda at: torch.sinc(at) * torch.sinc(at / 4) * (at.abs() < 4)
    onto_middle = lambda side: lanczos(taps - (0.25 - side / 2)) / lanczos(taps - (0.25 - side / 2)).sum()
    kernels = torch.stack([torch.outer(onto_middle(i), onto_middle(j)) for i in (0, 1) for j in (0, 1)])
    planes = torch.stack([mosaics[:, i::2, j::2] for i in (0, 1) for j in (0, 1)], 1)
    planes = torch.nn.functional.pad(planes, (4, 4, 4, 4), mode="reflect")
    red, green, also_green, blue = torch.nn.functional.conv2d(planes, kernels[:, None], groups=4).unbind(1)
    return torch.stack([red, (green + also_green) / 2, blue], 1)


def mosaiced(picture):
    """Camera RGB (N, 3, H, W) as the RGGB mosaic a sensor would have made of it; a mosaic comes back as it is."""
    if picture.dim() == 3:
        return picture
    mosaic = picture[:, 1].clone()
    mosaic[:, 0::2, 0::2] = picture[:, 0, 0::2, 0::2]
    mosaic[:, 1::2, 1::2] = picture[:, 2, 1::2, 1::2]
    return mosaic


def as_mosaic(clean_up):
    """clean_up with a picture it hands back sampled to a mosaic, so only its denoising is scored."""
    return lambda mosaics, mean: mosaiced(clean_up(mosaics, mean))


def build_colour(name, path, base, isos, count):
    """Windows whose truth is full colour: the lowest ISO frame binned (full_colour), and its mosaic under noise."""
    full = raw_pairs.mosaic(path(base))
    if full is None or not isos:
        return
    truth = full_colour(torch.from_numpy(full)[None])
    truth = truth[..., : truth.shape[-2] // 2 * 2, : truth.shape[-1] // 2 * 2]
    clean = mosaiced(truth)[0].numpy()
    picks = raw_pairs.corners(clean, np.random.default_rng(0), count)
    cut = lambda frame: np.stack([frame[..., y : y + raw_pairs.CROP, x : x + raw_pairs.CROP] for y, x in picks])
    frames, means = under_noise(name, clean, full, path, isos, cut)
    wb, matrix = colour(path(base))
    save(
        f"colour/{name}",
        truth=cut(truth[0].numpy()).astype(np.float16),
        frames=np.stack(frames).astype(np.float16),
        means=means,
        wb=wb,
        matrix=matrix,
        isos=isos,
    )


def held_out():
    """Every still-life and other lab folder of the cameras no scene of which trains."""
    folders = sorted(STILL_LIFES.glob("*/*"))
    return [f for f in folders if any(name in f.parent.name for name in raw_pairs.HELD_OUT) and any(f.iterdir())]


def build_colours():
    for camera in CAMERAS:
        build_colour(camera, ricoh(camera), 100, ISOS, WINDOWS // 2)
    for folder in groups()["seen"]:
        build_colour(f"seen--{folder.parent.name}", *ladder(folder), WINDOWS // 4)
    for folder in held_out():
        build_colour(f"unseen--{folder.parent.name}-{folder.name}", *ladder(folder), WINDOWS // 4)


def bilinear(mosaic):
    """RGGB mosaics (N, H, W) as camera RGB (N, 3, H, W) by the plainest demosaic."""
    sites = mosaic.new_zeros(3, *mosaic.shape[1:])
    sites[0, 0::2, 0::2] = sites[1, 0::2, 1::2] = sites[1, 1::2, 0::2] = sites[2, 1::2, 1::2] = 1
    cross = mosaic.new_tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    full = mosaic.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    planes = torch.nn.functional.pad(mosaic[:, None] * sites, (1, 1, 1, 1), mode="reflect")
    return torch.nn.functional.conv2d(planes, torch.stack([full, cross, full])[:, None], groups=3)


def develop(result, wb, matrix, demosaic=None):
    """A result as linear sRGB (N, 3, H, W) under the white balance and colour matrix.

    RGGB mosaics (N, H, W) go through the bilinear demosaic, or through demosaic, which gets them
    white-balanced as a raw developer would hand them over: a demosaic that leans on the colours
    moving together needs grey to be grey. Camera RGB (N, 3, H, W), from a model that demosaics or
    from full_colour, has none left to do.
    """
    balance = wb.view(1, 3, 1, 1)
    if demosaic and result.dim() == 3:
        return torch.einsum(
            "ij,njhw->nihw", matrix, demosaic(result * mosaiced(balance.expand(1, 3, *result.shape[1:])))
        )
    rgb = bilinear(result) if result.dim() == 3 else result
    return torch.einsum("ij,njhw->nihw", matrix, rgb * balance)


def windows(camera):
    """Frames (1 + ISOs, windows, 512, 512) with the clean one first, their ISOs and means, the colour,
    and each noisy frame's clean twin and black offsets (ISOs, 4), which a test under made-up noise goes
    without. A full-colour test's clean frame is its truth, camera RGB (windows, 3, 512, 512)."""
    data = np.load(BENCH / f"{camera}.npz")
    on_gpu = lambda name: torch.from_numpy(data[name].astype(np.float32)).cuda() if name in data else None
    return (
        on_gpu("frames") if "truth" not in data else [on_gpu("truth"), *on_gpu("frames")],
        data["isos"].tolist(),
        data["means"].tolist(),
        on_gpu("wb"),
        on_gpu("matrix"),
        on_gpu("twins"),
        on_gpu("offsets"),
    )


#: The Ricohs stay on the GPU between a run's checks; a group's cameras come and go.
kept = functools.cache(windows)


@functools.cache
def perceptual():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cuda()


@torch.no_grad()
def scores(clean_up, camera, with_lpips=False, demosaic=None):
    """Three {ISO: [PSNR, LPIPS]} for clean_up(mosaics (N, H, W), their frame's mean) -> cleaned mosaics.

    The first is against the clean windows as cut, the second against each noisy frame's clean twin,
    the third the same with the noisy frame left on its black offset, as shot. clean_up may hand back
    camera RGB (N, 3, H, W) instead, which is scored as the picture it is. A full-colour test,
    "colour/<camera>", has one score, three times. demosaic is what a cleaned mosaic goes through
    in place of the bilinear one (see develop).
    """
    if camera.split("/")[-1] in CAMERAS:
        return score_one(clean_up, kept(camera), with_lpips, demosaic)
    names = [file.relative_to(BENCH).with_suffix("").as_posix() for file in sorted(BENCH.glob(f"{camera}--*.npz"))]
    each = [score_one(clean_up, windows(name), with_lpips, demosaic) for name in names]
    had = [{iso: [by_iso[iso] for by_iso in kind if iso in by_iso] for iso in ISOS} for kind in zip(*each)]
    return [{iso: np.mean(found, axis=0).tolist() for iso, found in kind.items() if found} for kind in had]


def score(clean_up, camera, with_lpips=False):
    """The score against the clean windows as cut: every run before the twins logged this one."""
    return scores(clean_up, camera, with_lpips)[0]


def score_one(clean_up, cut, with_lpips, demosaic=None):
    frames, isos, means, wb, matrix, twins, offsets = cut
    linear = develop(frames[0], wb, matrix)
    exposure = 0.108 / linear.median()
    picture = lambda lin: ((lin * exposure).clamp(0, 1) ** (1 / 2.2))[..., EDGE:-EDGE, EDGE:-EDGE]

    def against(result, truth):
        psnr = -10 * torch.log10(((result - truth) ** 2).mean(dim=(1, 2, 3)))
        distance = perceptual()(result * 2 - 1, truth * 2 - 1) if with_lpips else torch.full((1,), float("nan"))
        return [psnr.mean().item(), distance.mean().item()]

    as_cut, aligned, as_shot = {}, {}, {}
    for k, (iso, noisy, mean) in enumerate(zip(isos, frames[1:], means)):
        result = picture(develop(clean_up(noisy, mean), wb, matrix, demosaic))
        as_cut[iso] = aligned[iso] = as_shot[iso] = against(result, picture(linear))
        if twins is not None:
            truth = picture(develop(twins[k], wb, matrix))
            black = offsets[k].view(2, 2).repeat(noisy.shape[-2] // 2, noisy.shape[-1] // 2)
            shot = clean_up(noisy + black, mean + offsets[k].mean().item())
            aligned[iso] = against(result, truth)
            as_shot[iso] = against(picture(develop(shot, wb, matrix, demosaic)), truth)
    return as_cut, aligned, as_shot


def main(specs):
    import demosaic
    import train_raw

    board = raw_pairs.DATA / "out" / "bench.json"
    table = json.loads(board.read_text()) if board.exists() else {}
    models = {name: train_raw.trained(weights) for name, weights in (spec.split("=") for spec in specs)}
    cleaners = {"noisy": lambda noisy, mean: noisy}
    cleaners |= {name: train_raw.cleaner(model) for name, model in models.items()}
    faithful = [name for name, model in models.items() if not train_raw.joint(model)]

    def show(tests):
        for test in tests:
            print(f"\n{test}  PSNR / LPIPS\n{'':24}" + "".join(f"{iso:>14}" for iso in ISOS) + f"{'mean':>14}")
            for name, by_iso in table[test].items():
                cells = [*by_iso.values(), np.mean(list(by_iso.values()), axis=0)]
                print(f"{name:24}" + "".join(f"{p:8.2f} {d:.3f}" for p, d in cells))

    for camera in (*CAMERAS, *groups(), "exact"):
        tests = (camera,) if camera == "exact" else (camera, f"{camera}-aligned", f"{camera}-as-shot")
        for name, clean_up in cleaners.items():
            for test, by_iso in zip(tests, scores(as_mosaic(clean_up), camera, with_lpips=True)):
                table.setdefault(test, {})[name] = by_iso
        show(tests)
    for camera in (*CAMERAS, *groups()):
        rows = table.setdefault(f"{camera}-colour", {})
        for name, clean_up in cleaners.items():
            rows[name] = scores(clean_up, f"colour/{camera}", with_lpips=True)[0]
            if name in faithful:
                rows[f"{name}-menon"] = scores(clean_up, f"colour/{camera}", True, demosaic.menon)[0]
        show([f"{camera}-colour"])
    board.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        for camera in CAMERAS:
            build(camera, ricoh(camera))
        for group, folders in groups().items():
            for folder in folders:
                build_group(group, folder)
        build_exact("gr3x", ricoh("gr3x"), 100, ISOS)
        for folder in held_out():
            build_exact(f"{folder.parent.name}-{folder.name}", *ladder(folder))
        build_colours()
    elif sys.argv[1:] == ["colour"]:
        build_colours()
    else:
        main(sys.argv[1:])
