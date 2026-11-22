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

Every test scores four times. The first score, under the test's own name, is against the ISO 100
window as it was cut: whole even pixels from the noisy one, by raw_pairs.shift_between, which leaves
most cameras a fraction of a pixel off and some two pixels, and no denoiser can score past that. The
second, "<test>-aligned", is against the same window resampled onto the noisy one (raw_pairs.offsets),
and is the one to read; the first stays so old runs compare. In both the noisy frame's black offset
was taken off with the clean frame's help (raw_pairs.match), which a real photo never gets. The
third, "<test>-as-shot", puts that offset back before the denoiser sees the frame and scores against
the resampled window: what a photo straight from the camera would get. The fourth,
"<test>-as-shot-fixed", is that photo with the offsets its body is known by taken off first
(blacks.py, looked up from the whole noisy frame's EXIF when the windows are cut), which is how
raw_denoise.py runs a photo; a body blacks.py does not know scores as it does as shot.

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

"shift" is a full-colour test that was not built our way: PixelShift200's five test scenes, a Sony
A7R III's, with every colour measured at every pixel by four shots a pixel apart (pixel_shift.py).
Its truth is at the sharpness the lens gave, where full_colour's is binned and twice as sharp, and
neither we nor a rival made it. The noise is made up as the "<test>-colour" tests make theirs, at
the levels measured on the same camera's still life, and it is scored as they are.

    uv run bench_raw.py shift                                          cut only those

A full-colour test also says how much of the truth's fine detail a result holds and how much it made
up, with no network to judge by (detail): a run that trains on LPIPS is not picked by LPIPS. Its rows
carry six numbers after PSNR and LPIPS: the detail kept, the detail invented, the grain in flat areas
and the truth's own, both in 8-bit levels, and the error spectrum's peak and its highest value at the
two-pixel limit, each over its median: the two are the same number when a two-pixel pattern is the
largest thing wrong.

"night" is the test of what the model is for: data/night-raw's dark scenes, ELD's, SID's Sony ones
and RawNIND's, each a real noisy frame against a long exposure, 8 windows a pair. Its windows are in
the noisy frame's own units, the ones denoise() meets, with the clean frame brought to them by the
gains raw_pairs.match fitted, so a model runs at the gain denoise() gives the photo and its tiles.
A pair scores twice: "night" with the noisy frame's black offset fitted from the clean one and taken
off, and "night-as-shot" with it left on; the gap is what a black offset costs at that gain. Both
sides stop at the lower of the two frames' white levels, since a long exposure clips lamps its short
one holds. Rows are by the gain the photo needs to reach train_raw.LEVEL, named by the least of each
range in GAINS: under 2, 2 to 8, 8 to 64, and over 64, which is past what the model takes. Some pairs
are scenes built for training too (raw_pairs.py's night-<pair> and rawnind-<scene>): those score
apart, as "night-trained", whichever run is scored. data/bench/night/held.json names every pair's
scene, for train_raw.py --held to leave out; a run that does has still started from weights that met them.
Each pair's own scores go to data/out/night.json, as [PSNR, LPIPS, as-shot PSNR, as-shot LPIPS].

    uv run bench_raw.py night                                          cut its windows, into data/bench/night

"<test>-colour-tiles" asks whether the same pixels come out the same in another tile or at another
gain, on the full-colour tests cut again as 2 windows of 1,024 px a camera, the app's tile: a row is
the PSNR of each window cleaned as one tile, as four tiles of 512 px, and as one tile at half and at
double the gain, all on the same pixels, EDGE in from every 512 px tile's sides. Those rows are four
PSNRs, not a PSNR and an LPIPS, so they go to data/out/tiles.json and stay off the scoreboard, and
the check costs four passes over its windows, so it runs when asked for by name.

    uv run bench_raw.py tiles                                          cut those windows, into data/bench/wide
    uv run bench_raw.py tiles joint-v2=data/models/raw/joint-v2/joint.pth

"colour", "shift" or "night" before the models scores that test alone:

    uv run bench_raw.py night joint-v2=data/models/raw/joint-v2/joint.pth
"""

import functools
import json
import sys

import cv2
import numpy as np
import rawpy
import torch

import blacks
import pixel_shift
import raw_pairs

BENCH = raw_pairs.DATA / "bench"
CAMERAS = {"gr3": "ricoh-gr-iii--GR3hSLI{:06d}NR0.DNG", "gr3x": "ricoh-gr-iiix--GR3xhSLI{:06d}NR0.DNG"}
ISOS = (3200, 6400, 12800, 25600, 51200, 102400)
WINDOWS = 32
STILL_LIFES = raw_pairs.LADDERS / "ir"
SHIFT_LADDER = STILL_LIFES / "sony-alpha-ilce-a7r-iii" / "sli"
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
PAIRS = raw_pairs.NIGHT / "pairs"
NIGHT_WINDOWS = 8
GAINS = (1, 2, 8, 64)
WIDE_WINDOWS = 2
FINE, FLATTEST = 1.5, 0.3


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
    frames, twins, means, offsets, fixes = [cut(clean)], [], [], [], []
    for iso in isos:
        noisy = raw_pairs.mosaic(path(iso))
        dx, dy = raw_pairs.shift_between(clean, noisy)
        noisy, gains, offset = raw_pairs.match(clean, noisy)
        offsets.append(offset)
        fixes.append(blacks.of(path(iso)) * gains)
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
        fixes=np.array(fixes, np.float32),
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


def full_colour(mosaics, bin=2):
    """RGGB mosaics (N, H, W) as camera RGB at half the size (N, 3, H/2, W/2), every colour measured, none guessed.

    Each 2x2 cell becomes one pixel at the cell's middle. Its four samples sit half a sensor pixel off
    that middle, each its own way, so each colour plane is first read a quarter of its own pixel
    along (Lanczos) and the two greens are averaged. Taken as they are the colours would sit apart
    by that much, a fringe on every edge that a model would learn to draw.

    A cell holds one red and one blue sample, so that picture keeps most of its frame's noise. With
    bin 4 each pixel is the mean of four such cells, a quarter of the size: half the noise again in
    every colour, and less of the colour moire a lens draws finer than one colour's samples sit.
    """
    taps = torch.arange(-4, 5, dtype=mosaics.dtype, device=mosaics.device)
    lanczos = lambda at: torch.sinc(at) * torch.sinc(at / 4) * (at.abs() < 4)
    onto_middle = lambda side: lanczos(taps - (0.25 - side / 2)) / lanczos(taps - (0.25 - side / 2)).sum()
    kernels = torch.stack([torch.outer(onto_middle(i), onto_middle(j)) for i in (0, 1) for j in (0, 1)])
    planes = torch.stack([mosaics[:, i::2, j::2] for i in (0, 1) for j in (0, 1)], 1)
    planes = torch.nn.functional.pad(planes, (4, 4, 4, 4), mode="reflect")
    red, green, also_green, blue = torch.nn.functional.conv2d(planes, kernels[:, None], groups=4).unbind(1)
    picture = torch.stack([red, (green + also_green) / 2, blue], 1)
    return picture if bin == 2 else torch.nn.functional.avg_pool2d(picture, bin // 2)


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


def build_colour(name, path, base, isos, count, side=raw_pairs.CROP, folder="colour"):
    """Windows whose truth is full colour: the lowest ISO frame binned (full_colour), and its mosaic under noise."""
    full = raw_pairs.mosaic(path(base))
    if full is None or not isos:
        return
    truth = full_colour(torch.from_numpy(full)[None])
    truth = truth[..., : truth.shape[-2] // 2 * 2, : truth.shape[-1] // 2 * 2]
    clean = mosaiced(truth)[0].numpy()
    picks = raw_pairs.corners(clean, np.random.default_rng(0), count, side)
    cut = lambda frame: np.stack([frame[..., y : y + side, x : x + side] for y, x in picks])
    frames, means = under_noise(name, clean, full, path, isos, cut)
    wb, matrix = colour(path(base))
    save(
        f"{folder}/{name}",
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


def build_shifts():
    """The shift test's windows: each of pixel_shift.TEST's scenes as measured, and its mosaic under the camera's noise.

    A window where something moved between the four shots is left out (pixel_shift.still).
    """
    path, base, isos = ladder(SHIFT_LADDER)
    full = raw_pairs.mosaic(path(base))
    for scene in pixel_shift.TEST:
        picture, apart = pixel_shift.measured(scene)
        ratios = pixel_shift.moved(picture, apart, raw_pairs.tags(pixel_shift.shot(scene))["iso"])
        clean = mosaiced(torch.from_numpy(picture)[None])[0].numpy()
        found = raw_pairs.corners(clean, np.random.default_rng(0), WINDOWS // 2)
        picks = [(y, x) for y, x in found if pixel_shift.still(ratios, y, x)]
        cut = lambda frame: pixel_shift.cut(frame, picks)
        frames, means = under_noise(f"shift {scene}", clean, full, path, isos, cut)
        wb, matrix = colour(pixel_shift.shot(scene))
        save(
            f"shift--{SHIFT_LADDER.parent.name}-{scene}",
            truth=cut(picture).astype(np.float16),
            frames=np.stack(frames).astype(np.float16),
            means=means,
            wb=wb,
            matrix=matrix,
            isos=isos,
        )


def build_wide():
    """The full-colour tests again as windows of the app's tile, for tiles()."""
    wide = WIDE_WINDOWS, raw_pairs.WIDE, "wide"
    for camera in CAMERAS:
        build_colour(camera, ricoh(camera), 100, ISOS, *wide)
    for folder in groups()["seen"]:
        build_colour(f"seen--{folder.parent.name}", *ladder(folder), *wide)
    for folder in held_out():
        build_colour(f"unseen--{folder.parent.name}-{folder.name}", *ladder(folder), *wide)


def scene(pair):
    """The training scene raw_pairs.py makes of a night pair: RawNIND's under the pair's name, the others' night-<pair>."""
    return pair if pair.startswith("rawnind-") else f"night-{pair}"


def trains(pair):
    """Whether a night pair's scene was built for training. One whose noisy frames have no clean twins
    counts too: today's runs leave it out, and the runs they started from drew from it."""
    built = raw_pairs.OUT / f"{scene(pair)}.json"
    return built.exists() and json.loads(built.read_text())["split"] == "train"


def build_night(folder):
    """A night pair's windows, cut as raw_pairs.build cuts a training pair's and left in the noisy frame's own units.

    raw_pairs.match puts the noisy frame on the clean one's scale, each colour plane by its own gain;
    dividing by those gains again leaves the noisy frame as shot less its black offset, and brings the
    clean windows and twins down to it. A pair whose frames moved apart, or do not line up, stays out.
    The file is a test's (windows) with one noisy frame, so its ISO and mean are single numbers, and it
    carries what rivals.py needs to know of the camera.
    """
    clean_path, noisy_path = next(folder.glob("clean_*")), next(folder.glob("noisy_*"))
    clean, shot = raw_pairs.mosaic(clean_path), raw_pairs.mosaic(noisy_path)
    if clean is None or shot is None or clean.shape != shot.shape:
        print(folder.name, "left out: not Bayer", flush=True)
        return
    picks = raw_pairs.corners(clean, np.random.default_rng(0), NIGHT_WINDOWS)
    noisy, gains, blacks = raw_pairs.match(clean, shot)
    at = raw_pairs.offsets(clean, noisy)
    moves, shares = (np.array(found) for found in zip(*(at(y, x) for y, x in picks)))
    whole = (np.round(moves / 2) * 2).astype(int)
    lined_up = min(np.median(shares), shares.min() / np.median(shares))
    if np.abs(whole).max() > raw_pairs.MAX_SHIFT or lined_up < raw_pairs.SHARE:
        print(
            folder.name, "left out: moved", *moves.mean(0).round(1), "lined up", round(float(lined_up), 2), flush=True
        )
        return
    side = raw_pairs.CROP
    whites = [min(1.0, gain - black) for gain, black in zip(gains, blacks)]
    twins = [raw_pairs.resampled(clean, y, x, dx, dy) for (y, x), (dx, dy) in zip(picks, moves - whole)]
    crops = [
        [clean[y : y + side, x : x + side] for y, x in picks],
        [noisy[y + dy : y + dy + side, x + dx : x + dx + side] for (y, x), (dx, dy) in zip(picks, whole)],
        raw_pairs.capped(np.stack(twins), whites),
    ]
    scale = np.tile(np.reshape(gains, (2, 2)), (side // 2, side // 2))
    as_cut, own, twin = ((np.stack(crop) / scale).astype(np.float16) for crop in crops)
    mean = float(np.mean([noisy[i::2, j::2].mean() / gains[2 * i + j] for i in range(2) for j in range(2)]))
    with rawpy.imread(str(noisy_path)) as raw:
        black = float(np.mean(raw.black_level_per_channel)) / raw.white_level
        camera_from_xyz = raw.rgb_xyz_matrix[:3]
    wb, matrix = colour(clean_path)
    save(
        f"night/{folder.name}",
        frames=np.stack([as_cut, own]),
        twins=twin[None],
        mean=mean,
        offsets=np.divide(blacks, gains, dtype=np.float32)[None],
        whites=np.divide(whites, gains, dtype=np.float32)[None],
        wb=wb,
        matrix=matrix,
        iso=raw_pairs.tags(noisy_path)["iso"],
        trains=trains(folder.name),
        camera_from_xyz=camera_from_xyz,
        black_over_white=black,
    )
    print(
        folder.name,
        "mean",
        f"{mean:.3g}",
        "lined up",
        round(float(lined_up), 2),
        "trains" * trains(folder.name),
        flush=True,
    )


def build_nights():
    for folder in sorted(PAIRS.iterdir()):
        build_night(folder)
    held = sorted(scene(file.stem) for file in (BENCH / "night").glob("*.npz"))
    (BENCH / "night" / "held.json").write_text(json.dumps(held))


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
    and each noisy frame's clean twin, its black offsets (ISOs, 4) and the ones blacks.py knows its body
    by, on the same scale, which a test under made-up noise goes without. A full-colour test's clean
    frame is its truth, camera RGB (windows, 3, 512, 512)."""
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
        on_gpu("fixes"),
    )


#: The Ricohs stay on the GPU between a run's checks; a group's cameras come and go.
kept = functools.cache(windows)


@functools.cache
def perceptual():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cuda().requires_grad_(False)


def fine(picture):
    """A picture's fine part: the picture less itself blurred by a bell of FINE px."""
    reach = int(3 * FINE)
    taps = torch.arange(-reach, reach + 1, dtype=picture.dtype, device=picture.device)
    bell = torch.exp(-(taps**2) / (2 * FINE**2))
    blur = (torch.outer(bell, bell) / bell.sum() ** 2).expand(picture.shape[1], 1, -1, -1)
    padded = torch.nn.functional.pad(picture, (reach,) * 4, mode="reflect")
    return picture - torch.nn.functional.conv2d(padded, blur, groups=picture.shape[1])


def detail(result, truth):
    """What became of the truth's fine detail in a result, pictures (N, 3, H, W), with no network to judge by.

    kept is how much of the truth's fine part the result holds: its own fine part projected on the
    truth's, 1 when all of it is there. invented is the energy of what is left of the result's fine
    part after that, as a share of the truth's: detail and noise the truth does not have. Then the
    grain where the truth is flattest, the result's and the truth's own, as the spread of their fine
    parts in 8-bit levels: a result near the truth's has copied the noise its frame left in it. Last
    the error's power spectrum, its peak and its highest value at the two-pixel limit, both over its
    median: equal when the largest thing wrong is a pattern two pixels wide.
    """
    made, real = fine(result), fine(truth)
    kept = (made * real).sum() / (real * real).sum()
    invented = (made - kept * real).pow(2).sum() / real.pow(2).sum()
    grey = torch.nn.functional.pad(truth.mean(1, keepdim=True), (4,) * 4, mode="reflect")
    pooled = lambda values: torch.nn.functional.avg_pool2d(values, 9, 1)
    rough = (pooled(grey * grey) - pooled(grey) ** 2).flatten()
    flat = (rough <= rough.kthvalue(int(FLATTEST * len(rough))).values).view_as(truth[:, :1])
    grain = lambda part: part.mean(1, keepdim=True)[flat].std() * 255
    spectrum = torch.fft.fft2((result - truth).mean(1)).abs().pow(2).mean(0)
    spectrum[0, 0] = 0
    down, across = (side // 2 for side in spectrum.shape)
    two_pixel = torch.stack([spectrum[0, across], spectrum[down, 0], spectrum[down, across]]).max()
    found = kept, invented, grain(made), grain(real), spectrum.max() / spectrum.median(), two_pixel / spectrum.median()
    return [value.item() for value in found]


def over_cameras(each):
    """A group's score as one camera's: each ISO's mean over the cameras that have it."""
    had = {iso: [by_iso[iso] for by_iso in each if iso in by_iso] for iso in ISOS}
    return {iso: np.mean(found, axis=0).tolist() for iso, found in had.items() if found}


def cuts(camera):
    """The files of a test, "<camera>" or "<folder>/<camera>": a Ricoh's one, or each camera's of a group."""
    if camera.split("/")[-1] in CAMERAS:
        return [camera] * (BENCH / f"{camera}.npz").exists()
    return [file.relative_to(BENCH).with_suffix("").as_posix() for file in sorted(BENCH.glob(f"{camera}--*.npz"))]


@torch.no_grad()
def scores(clean_up, camera, with_lpips=False, demosaic=None, with_detail=False):
    """Four {ISO: [PSNR, LPIPS]} for clean_up(mosaics (N, H, W), their frame's mean) -> cleaned mosaics.

    The first is against the clean windows as cut, the second against each noisy frame's clean twin,
    the third the same with the noisy frame left on its black offset, as shot, and the fourth as shot
    with the offsets blacks.py knows the body by taken off. clean_up may hand back
    camera RGB (N, 3, H, W) instead, which is scored as the picture it is. A full-colour test,
    "colour/<camera>", has one score, four times. demosaic is what a cleaned mosaic goes through
    in place of the bilinear one (see develop). with_detail adds detail()'s six numbers to each row.
    """
    if camera.split("/")[-1] in CAMERAS:
        return score_one(clean_up, kept(camera), with_lpips, demosaic, with_detail)
    each = [score_one(clean_up, windows(name), with_lpips, demosaic, with_detail) for name in cuts(camera)]
    return [over_cameras(kind) for kind in zip(*each)]


def score(clean_up, camera, with_lpips=False):
    """The score against the clean windows as cut: every run before the twins logged this one."""
    return scores(clean_up, camera, with_lpips)[0]


def score_one(clean_up, cut, with_lpips, demosaic=None, with_detail=False, whites=None):
    """scores() for one file's windows. whites (ISOs, 4) is where each colour plane of a night pair's
    truth stops: a result is sampled to a mosaic and stopped there too before it is looked at."""
    frames, isos, means, wb, matrix, twins, offsets, fixes = cut
    linear = develop(frames[0], wb, matrix)
    exposure = 0.108 / linear.median()
    picture = lambda lin: ((lin * exposure).clamp(0, 1) ** (1 / 2.2))[..., EDGE:-EDGE, EDGE:-EDGE]

    def against(result, truth):
        psnr = -10 * torch.log10(((result - truth) ** 2).mean(dim=(1, 2, 3)))
        distance = (
            torch.cat([perceptual()(a * 2 - 1, b * 2 - 1).flatten() for a, b in zip(result.split(4), truth.split(4))])
            if with_lpips
            else torch.full((1,), float("nan"))
        )
        return [psnr.mean().item(), distance.mean().item(), *(detail(result, truth) if with_detail else ())]

    def shown(result, k):
        if whites is not None:
            white = whites[k].view(2, 2).repeat(result.shape[-2] // 2, result.shape[-1] // 2)
            result = torch.minimum(mosaiced(result), white)
        return picture(develop(result, wb, matrix, demosaic))

    def left_on(noisy, mean, black, truth, k):
        lift = black.view(2, 2).repeat(noisy.shape[-2] // 2, noisy.shape[-1] // 2)
        return against(shown(clean_up(noisy + lift, mean + black.mean().item()), k), truth)

    as_cut, aligned, as_shot, fixed = {}, {}, {}, {}
    for k, (iso, noisy, mean) in enumerate(zip(isos, frames[1:], means)):
        result = shown(clean_up(noisy, mean), k)
        as_cut[iso] = aligned[iso] = as_shot[iso] = fixed[iso] = against(result, picture(linear))
        if twins is not None:
            truth = picture(develop(twins[k], wb, matrix))
            aligned[iso] = against(result, truth)
            as_shot[iso] = fixed[iso] = left_on(noisy, mean, offsets[k], truth, k)
            if fixes is not None and fixes[k].any():
                fixed[iso] = left_on(noisy, mean, offsets[k] - fixes[k], truth, k)
    return as_cut, aligned, as_shot, fixed


def at_gain(clean_up, times):
    """clean_up as it runs at times its gain: the same windows, shown to it that much brighter."""
    return lambda mosaics, mean: clean_up(mosaics * times, mean) / times


def quarters(windows):
    """Windows of 1,024 px (N, ..., H, W) as their four tiles of 512 px each (4N, ..., H/2, W/2)."""
    side = raw_pairs.CROP
    return torch.cat([windows[..., y : y + side, x : x + side] for y in (0, side) for x in (0, side)])


@torch.no_grad()
def tiles(clean_up, camera):
    """{ISO: four PSNRs} of a full-colour test's 1,024 px windows: cleaned as one tile, as four tiles of
    512 px, and as one tile at half and at double the gain, all on the pixels EDGE in from a 512 px tile's sides."""
    return over_cameras([tiles_one(clean_up, windows(name)) for name in cuts(f"wide/{camera}")])


def tiles_one(clean_up, cut):
    (truth, *frames), isos, means, wb, matrix, *_ = cut
    linear = develop(quarters(truth), wb, matrix)
    exposure = 0.108 / linear.median()
    picture = lambda lin: ((lin * exposure).clamp(0, 1) ** (1 / 2.2))[..., EDGE:-EDGE, EDGE:-EDGE]
    psnr = lambda result: (
        (-10 * torch.log10(((picture(develop(result, wb, matrix)) - picture(linear)) ** 2).mean(dim=(1, 2, 3))))
        .mean()
        .item()
    )
    return {
        iso: [
            psnr(quarters(clean_up(noisy, mean))),
            psnr(clean_up(quarters(noisy), mean)),
            *(psnr(quarters(at_gain(clean_up, times)(noisy, mean))) for times in (0.5, 2)),
        ]
        for iso, noisy, mean in zip(isos, frames, means)
    }


def needs(gain):
    """The range of GAINS a photo that needs a gain falls in, by the range's least."""
    return max(low for low in GAINS if low <= max(gain, GAINS[0]))


def pair(file):
    """A night pair's file as score_one takes a test's windows, one ISO of them, then where its colour
    planes stop, its ISO, its mean and whether it is a training scene."""
    data = np.load(file)
    on_gpu = lambda name: torch.from_numpy(data[name].astype(np.float32)).cuda()
    iso, mean = int(data["iso"]), float(data["mean"])
    cut = on_gpu("frames"), [iso], [mean], on_gpu("wb"), on_gpu("matrix"), on_gpu("twins"), on_gpu("offsets"), None
    return cut, on_gpu("whites"), iso, mean, bool(data["trains"])


@torch.no_grad()
def night(name, clean_up):
    """The night test's rows for a cleaner: {test: {least gain of a range: [PSNR, LPIPS]}}, each the mean
    over that range's pairs. A picture it hands back is sampled to a mosaic and stopped at the pair's
    white, as its truth is. The pairs that are no training scene make "night" and "night-as-shot", the
    rest "night-trained" and its as-shot. Each pair's scores are kept under the name in
    data/out/night.json, with the gain it needs to reach train_raw.LEVEL once its black offset is
    off, and whether it is a training scene."""
    import train_raw

    kept_at = raw_pairs.DATA / "out" / "night.json"
    by_pair = json.loads(kept_at.read_text()) if kept_at.exists() else {}
    tests = {}
    for file in sorted((BENCH / "night").glob("*.npz")):
        cut, whites, iso, mean, trained = pair(file)
        _, fitted, as_shot, _ = score_one(clean_up, cut, with_lpips=True, whites=whites)
        gain = train_raw.LEVEL / max(mean, 1e-9)
        rows = by_pair.get(file.stem, {}).get("rows", {}) | {name: [*fitted[iso], *as_shot[iso]]}
        by_pair[file.stem] = {"gain": gain, "trains": trained, "rows": rows}
        for ending, row in (("", fitted[iso]), ("-as-shot", as_shot[iso])):
            test = f"night{'-trained' * trained}{ending}"
            tests.setdefault(test, {}).setdefault(needs(gain), []).append(row)
    kept_at.write_text(json.dumps(by_pair, indent=1))
    return {test: {low: np.mean(rows[low], axis=0).tolist() for low in sorted(rows)} for test, rows in tests.items()}


def main(specs, only=""):
    import demosaic
    import train_raw

    board, wide = (raw_pairs.DATA / "out" / file for file in ("bench.json", "tiles.json"))
    table, tile_table = (json.loads(file.read_text()) if file.exists() else {} for file in (board, wide))
    models = {name: train_raw.trained(weights) for name, weights in (spec.split("=") for spec in specs)}
    cleaners = {"noisy": lambda noisy, mean: noisy}
    cleaners |= {name: train_raw.cleaner(model) for name, model in models.items()}
    tiled = {"noisy": cleaners["noisy"]}
    tiled |= {name: train_raw.cleaner(model, tiled=True) for name, model in models.items()}
    faithful = [name for name, model in models.items() if not train_raw.joint(model)]

    def show(rows, title, cell=lambda row: f"{row[0]:8.2f} {row[1]:.3f}", mean=True):
        heads = [*rows["noisy"], *["mean"] * mean]
        print(f"\n{title}\n{'':24}" + "".join(f"{head:>20}" for head in heads))
        for name, by_iso in rows.items():
            cells = [*by_iso.values(), *[np.mean(list(by_iso.values()), axis=0)] * mean]
            print(f"{name:24}" + "".join(f"{cell(row):>20}" for row in cells))

    for camera in (*CAMERAS, *groups(), "exact") if not only else ():
        kinds = () if camera == "exact" else ("aligned", "as-shot", "as-shot-fixed")
        tests = (camera, *(f"{camera}-{kind}" for kind in kinds))
        for name, clean_up in cleaners.items():
            for test, by_iso in zip(tests, scores(as_mosaic(clean_up), camera, with_lpips=True)):
                table.setdefault(test, {})[name] = by_iso
        for test in tests:
            show(table[test], f"{test}  PSNR / LPIPS")
    colours = {f"{camera}-colour": f"colour/{camera}" for camera in (*CAMERAS, *groups())} | {"shift": "shift"}
    for test, cut in colours.items():
        if only not in ("", "shift" if test == "shift" else "colour"):
            continue
        rows = table.setdefault(test, {})
        for name, clean_up in cleaners.items():
            rows[name] = scores(clean_up, cut, True, with_detail=True)[0]
            if name in faithful:
                rows[f"{name}-menon"] = scores(clean_up, cut, True, demosaic.menon, True)[0]
        show(rows, f"{test}  PSNR / LPIPS")
        show(
            {name: by_iso for name, by_iso in rows.items() if len(next(iter(by_iso.values()))) > 2},
            f"{test}  detail kept / invented / grain in flat areas",
            lambda row: f"{row[2]:.3f} {row[3]:.3f} {row[4]:.2f}",
        )
    for camera in (*CAMERAS, *groups()) if only == "tiles" else ():
        if not cuts(f"wide/{camera}"):
            continue
        rows = tile_table.setdefault(f"{camera}-colour-tiles", {})
        rows |= {name: tiles(clean_up, camera) for name, clean_up in cleaners.items()}
        show(
            rows,
            f"{camera}-colour-tiles  PSNR as one tile of 1,024 px / as four of 512 px / spread over half, one and double the gain",
            lambda row: f"{row[0]:.2f} {row[1]:.2f} {np.ptp([row[0], *row[2:]]):.2f}",
        )
        wide.write_text(json.dumps(tile_table, indent=1))
    if only in ("", "night") and any((BENCH / "night").glob("*.npz")):
        for name, clean_up in tiled.items():
            for test, rows in night(name, clean_up).items():
                table.setdefault(test, {})[name] = rows
        for test in sorted(test for test in table if test.startswith("night")):
            show(table[test], f"{test}  PSNR / LPIPS by the least gain a photo needs", mean=False)
    board.write_text(json.dumps(table, indent=1))


def build_all():
    for camera in CAMERAS:
        build(camera, ricoh(camera))
    for group, folders in groups().items():
        for folder in folders:
            build_group(group, folder)
    build_exact("gr3x", ricoh("gr3x"), 100, ISOS)
    for folder in held_out():
        build_exact(f"{folder.parent.name}-{folder.name}", *ladder(folder))
    build_colours()
    build_shifts()
    build_wide()
    build_nights()


BUILDS = {
    "build": build_all,
    "colour": build_colours,
    "shift": build_shifts,
    "tiles": build_wide,
    "night": build_nights,
}

if __name__ == "__main__":
    only, specs = (sys.argv[1], sys.argv[2:]) if sys.argv[1] in BUILDS else ("", sys.argv[1:])
    if specs:
        main(specs, only)
    else:
        BUILDS[only]()
