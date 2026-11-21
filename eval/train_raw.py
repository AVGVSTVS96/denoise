"""Train the faithful model: NAFNet (width 32) on raw Bayer mosaics, noisy in, clean mosaic out.

    uv run train_raw.py                                   from scratch, or resumes data/models/raw/last.pth
    uv run train_raw.py --out gr3 --init data/models/raw/faithful/faithful.pth \
        --profile data/noise/gr3.json --steps 20000 --lr 1e-4       one camera's fine-tune

The network sees the mosaic packed into four half-size planes (R, G, G, B), so a phone runs it at a
quarter of the pixels. Its body starts from the SIDD sRGB weights; the first and last layers are new.
--width 64 trains the wider NAFNet, four times the work for a picture, for when quality matters more.
Batches come from raw_pairs.py's crops: real noisy frames, each against the clean twin resampled onto
it, and clean frames with noise laid over them.
A third of the draws go to the scenes that have real noisy frames, however many clean ones there are.
Without --profile that noise is either made up (shot noise, read noise and row banding, from a wide
range of cameras) or a window of the Ricoh GR III's real noise (noise_bank.py). A profile narrows the
run to one camera: every clean frame gets that camera's noise, its real noise fields if the profile
has them beside it and else noise made up from its measured levels (calibrate_raw.py), and only the
camera's own scenes (named <profile>-..., as ir-gr3-inbi is for gr3.json) keep their real noisy
frames. Each crop is scaled to a set brightness first, the same way denoise() scales a photo, so dark
night frames look like any other.

Four settings bring the made-up noise closer to a real camera's, each on its own so runs compare:
--levels data/noise/cameras.json draws two thirds of its levels as whole rows, one camera at one ISO
as fit_noise.py measured it, in place of levels drawn apart; --offsets leaves each colour plane on the black offset
a real frame sits on, for the network to take off, in made-up noise and in the real noisy frames,
which raw_pairs.py stores with theirs taken off; --tails gives the read noise a real sensor's long
tails and rounds the result to the raw's steps; --darks data/noise/darks.json swaps part of the read
noise for windows of real dark frames (dark_bank.py).

--joint 16 trains the joint model instead, which denoises and demosaics in one pass: packed RGGB in,
camera RGB at the mosaic's full size out, the picture a user looks at. It is the same network with
twelve planes in and out, each colour at each of a 2x2 cell's four places, so --init takes a
faithful model's weights and starts as that model with every cell's colours spread flat. A mosaic
never holds its own full-colour truth, so each step adds that many crops whose truth is made:
the clean crop binned to full colour at half its size (bench_raw.full_colour), sampled back to a
mosaic, with noise laid over it, and the loss on all three colours of every pixel. The crops of
--batch go through as before, real noisy frames among them, at their own size and with the faithful
loss on the picture sampled where the camera measured: binning a real noisy frame would average its
noise away, and its clean twin is a mosaic too. A joint run saves joint.pth and logs
bench_raw.py's full-colour scores beside the others, which are of its picture sampled back to a mosaic.

A step hands the GPU its work without stopping for it: the crops of both kinds go up in one piece,
which is the one place the CPU waits, and the small tables the loss and the joint network need stay
on the GPU.

The weights that get scored and saved are a running average of the trained ones, taken every fourth
step, which is steadier than any single step. Every check logs the held-out RawNIND score and
bench_raw.py's scores for picking between runs: the GR III's, its mean and each ISO from 3,200 to
102,400, and the mean over the cameras whose other scenes train. The "aligned" ones are against clean
windows resampled onto the noisy ones, the ones to pick by; the others are what older runs logged.
The "as shot" ones leave each noisy frame on its black offset, as a photo comes from the camera, and
the "as shot fixed" ones take off what blacks.py knows the body's offsets to be, as raw_denoise.py does.
run.json beside the log keeps the run's settings.
"""

import argparse
import fnmatch
import functools
import json
import math
import pathlib
import random
import threading
import time

import numpy as np
import torch

import bench_raw
import evaluate
import raw_pairs

PAIRS = evaluate.DATA / "rawpairs"
NOISE = evaluate.DATA / "noise"
MODELS = evaluate.MODELS / "raw"
#: Mean a crop or photo is scaled to before the network sees it.
LEVEL = 0.1
MAX_GAIN = 64.0
SYNTHETIC_SHARE = 0.4
#: How much of the laid-over noise is the GR III's real noise: one camera's, in a model for any camera.
BANK_SHARE = 0.2
#: The fewest electrons a pixel at LEVEL gets under laid-over noise. His night photo averages 17 and
#: half of it sits under 6; the GR IIIx's ISO 102,400 frame averages 9.
FLOOR = 2
#: How often a scene is drawn, by the start of its name. The review sites' ladders are hundreds of
#: cameras on a handful of scenes, so together they get about as many draws as the other real pairs,
#: which are far fewer cameras on far more scenes.
WEIGHTS = {"ladder-": 0.5}
#: The share of draws that go to scenes with real noisy frames, however many clean-only scenes join.
PAIR_SHARE = 1 / 3
#: Crops kept in memory, and how many of them a second are swapped for fresh ones from disk. The
#: scenes outgrew the disk cache, and a read per sample kept a shared server's disk busy.
POOL, READS = 1024, 4
#: His camera's colour, for the part of the loss that looks at the picture: white balance ranges that
#: cover daylight, tungsten and street light, and the GR III's matrix from camera colour to sRGB.
RED, BLUE = (1.5, 2.8), (1.4, 3.2)
TO_SRGB = ((1.703, -0.719, 0.016), (-0.188, 1.563, -0.375), (0.004, -0.406, 1.402))
EMA = 0.996
TABLE_SHARE = 2 / 3
DARK_SHARE = 0.25
BELL = 0.14
OFFSET_SHARED, OFFSET_OWN = 0.3, 0.1
TAILS = (0.0, 0.2)
READ_COUNTS = (1, 300)
MOST_BANDING = 0.3
#: The joint network's twelve planes from the mosaic's four: red, green and blue, each at a 2x2 cell's
#: four places in reading order. A colour goes where it was measured and the cell's mean elsewhere.
SPREAD = (
    *[(1.0, 0.0, 0.0, 0.0)] * 4,
    (0.0, 0.5, 0.5, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.5, 0.5, 0.0),
    *[(0.0, 0.0, 0.0, 1.0)] * 4,
)


@functools.cache
def held(numbers, device):
    """A table of numbers on a device, sent there once: every sending waits for the GPU to finish what it holds."""
    return torch.tensor(numbers, device=device)


def network(width=32, joint=False):
    from nafnet import CONFIGS, NAFNet

    return NAFNet(CONFIGS[f"nafnet-sidd-w{width}"], img_channel=12 if joint else 4)


def joint(model):
    """Whether a network is a joint one, camera RGB out, and not a faithful one, mosaic out."""
    return model.intro.in_channels == 12


def trained(weights):
    """A saved model, faithful or joint, at whatever width it was trained."""
    state = torch.load(weights, map_location="cpu")
    model = network(state["intro.weight"].shape[0], state["intro.weight"].shape[1] == 12)
    model.load_state_dict(state)
    return model.cuda().eval()


def widened(state):
    """A faithful model's weights as a joint network's, which then gives that model's mosaic spread by SPREAD."""
    spread = torch.tensor(SPREAD)
    first = torch.einsum("omyx,mk->okyx", state["intro.weight"], torch.linalg.pinv(spread))
    last = torch.einsum("km,moyx->koyx", spread, state["ending.weight"])
    return {**state, "intro.weight": first, "ending.weight": last, "ending.bias": spread @ state["ending.bias"]}


def fed(model, mosaics):
    """What a network takes of mosaics (N, H, W): their four planes, spread to twelve for a joint one."""
    planes = pack(mosaics)
    return torch.nn.functional.conv2d(planes, held(SPREAD, planes.device)[:, :, None, None]) if joint(model) else planes


def made(model, planes):
    """What a network's planes are: cleaned mosaics (N, H, W), or camera RGB (N, 3, H, W) from a joint one."""
    return torch.nn.functional.pixel_shuffle(planes, 2) if joint(model) else unpack(planes)


def pack(mosaic):
    return torch.stack(
        [mosaic[..., 0::2, 0::2], mosaic[..., 0::2, 1::2], mosaic[..., 1::2, 0::2], mosaic[..., 1::2, 1::2]], -3
    )


def unpack(planes):
    n, _, h, w = planes.shape
    out = planes.new_empty(n, 2 * h, 2 * w)
    out[:, 0::2, 0::2], out[:, 0::2, 1::2], out[:, 1::2, 0::2], out[:, 1::2, 1::2] = planes.unbind(1)
    return out


def tone(x):
    """Roughly how a developed picture spaces brightness, so shadows count as much as highlights.

    Mirrored below zero: noise leaves a dark pixel negative as often as not, and a curve that is flat
    there would give the network nothing to learn from.
    """
    return x.sign() * ((x.abs() + 1e-3) ** (1 / 2.2) - 1e-3 ** (1 / 2.2))


def loss_of(result, clean):
    """How far the cleaned mosaic is from the clean one, as the picture they develop into.

    Half on the mosaic under a white balance, where detail lives, and half on its 2x2 cells as sRGB:
    the colour matrix is what blows left-over colour noise up into blotches.
    """
    n = clean.shape[0]
    draw = lambda low, high: torch.empty(n, device=clean.device).uniform_(low, high)
    red, green, blue = draw(*RED), torch.ones(n, device=clean.device), draw(*BLUE)
    balance = torch.stack([red, green, green, blue], 1).view(n, 4, 1, 1)
    a, b = pack(result) * balance, pack(clean) * balance
    cells = lambda p: torch.einsum(
        "ij,njhw->nihw", held(TO_SRGB, p.device), torch.stack([p[:, 0], (p[:, 1] + p[:, 2]) / 2, p[:, 3]], 1)
    )
    l1 = torch.nn.functional.l1_loss
    return (l1(tone(a), tone(b)) + l1(tone(cells(a)), tone(cells(b)))) / 2


def picture_loss(result, truth):
    """loss_of for camera RGB (N, 3, H, W) with every colour known at every pixel.

    Half on the three colours under a white balance, half on them as sRGB.
    """
    n = truth.shape[0]
    draw = lambda low, high: torch.empty(n, device=truth.device).uniform_(low, high)
    balance = torch.stack([draw(*RED), torch.ones(n, device=truth.device), draw(*BLUE)], 1).view(n, 3, 1, 1)
    a, b = result * balance, truth * balance
    srgb = lambda p: torch.einsum("ij,njhw->nihw", held(TO_SRGB, p.device), p)
    l1 = torch.nn.functional.l1_loss
    return (l1(tone(a), tone(b)) + l1(tone(srgb(a)), tone(srgb(b)))) / 2


def cleaner(model):
    """bench_raw.score's view of a model: mosaics (N, H, W) in, scaled as denoise() does, and out
    cleaned mosaics, or camera RGB (N, 3, H, W) from a joint model."""

    def clean_up(mosaics, mean):
        gain = min(max(LEVEL / mean, 1), MAX_GAIN)
        with torch.autocast("cuda", torch.bfloat16):
            parts = [made(model, model(fed(model, part * gain)).float()) for part in mosaics.split(8)]
        return torch.cat(parts) / gain

    return clean_up


@torch.no_grad()
def denoise(model, mosaic, tile=1024, overlap=64):
    """A whole normalised RGGB mosaic (H, W float32 numpy) through the model, in overlapping tiles.

    Every tile keeps only its middle; the last row and column of tiles sit flush with the far edge.
    Returns the cleaned mosaic, or camera RGB (3, H, W) from a joint model.
    """
    gain = float(np.clip(LEVEL / max(float(mosaic.mean()), 1e-6), 1, MAX_GAIN))
    padded = np.pad(mosaic * gain, overlap, "reflect")
    out = np.zeros((3, *padded.shape) if joint(model) else padded.shape, padded.dtype)
    starts = lambda side: sorted({*range(0, side - tile, tile - 2 * overlap), side - tile})
    for y in starts(padded.shape[0]):
        for x in starts(padded.shape[1]):
            piece = torch.from_numpy(np.ascontiguousarray(padded[y : y + tile, x : x + tile])).cuda()[None]
            with torch.autocast("cuda", torch.bfloat16):
                result = made(model, model(fed(model, piece)).float())[0].cpu().numpy()
            core = np.s_[..., overlap:-overlap, overlap:-overlap]
            out[..., y : y + tile, x : x + tile][core] = result[core]
    return out[..., overlap:-overlap, overlap:-overlap] / gain


class Crops:
    """Endless batches of (clean, noisy) mosaics, flipped and turned with the colours kept on RGGB.

    One thread reads: it fills the pool, then swaps READS crops a second. A crop is drawn many times
    before it leaves, each time under another window, flip and noise.
    """

    def __init__(self, scenes, size, seed, real, as_shot):
        self.size, self.real, self.as_shot, self.rng, self.pool = size, real, as_shot, random.Random(seed), []
        self.use(scenes)
        threading.Thread(target=self.read, daemon=True).start()

    def use(self, scenes):
        weights = [next((w for start, w in WEIGHTS.items() if s.startswith(start)), 1) for s in scenes]
        paired = [bool(frames) for frames in scenes.values()]
        total = {kind: sum(w for w, p in zip(weights, paired) if p == kind) for kind in (True, False)}
        weights = [w * (PAIR_SHARE if p else 1 - PAIR_SHARE) / total[p] for w, p in zip(weights, paired)]
        self.scenes, self.draw = scenes, (list(scenes), weights)

    def read(self):
        rng = random.Random(self.rng.random())
        while True:
            scene = rng.choices(*self.draw)[0]
            path = PAIRS / f"{scene}.npy"
            stack = np.load(path, mmap_mode="r")
            frames, crops, side, _ = stack.shape
            crop = rng.randrange(crops)
            noisy = frames // 2
            synthetic = not noisy or not fnmatch.fnmatch(scene, self.real) or rng.random() < SYNTHETIC_SHARE
            frame = 0 if synthetic else rng.randrange(1, noisy + 1)
            picks = [0] if synthetic else [frame + noisy, frame]
            # Read by offset: through the memory map the kernel fetches about three times what a crop holds.
            starts = [stack.offset + (i * crops + crop) * side * side * stack.itemsize for i in picks]
            planes = [np.fromfile(path, stack.dtype, side * side, offset=start) for start in starts]
            pair = np.stack(planes).reshape(-1, side, side)
            if self.as_shot and not synthetic:
                black = self.scenes[scene][frame - 1].get("offset", [0.0] * 4)
                pair[1] += np.tile(np.asarray(black, pair.dtype).reshape(2, 2), (side // 2, side // 2))
            fresh = pair, synthetic
            if len(self.pool) < POOL:
                self.pool.append(fresh)
            else:
                self.pool[rng.randrange(POOL)] = fresh
            time.sleep(1 / READS if len(self.pool) == POOL else 0.05)

    def batch(self, n):
        while len(self.pool) < POOL // 4:
            time.sleep(1)
        pairs, kinds = [], []
        for _ in range(n):
            pair, synthetic = self.rng.choice(self.pool)
            side = pair.shape[-1]
            flip_y, flip_x, turn = (self.rng.random() < 0.5 for _ in range(3))
            # A flipped mosaic starts on red again one pixel in, so the window's corner takes that parity.
            y = self.rng.randrange((side - self.size) // 2) * 2 + flip_y
            x = self.rng.randrange((side - self.size) // 2) * 2 + flip_x
            if flip_y:
                pair = pair[:, ::-1]
            if flip_x:
                pair = pair[:, :, ::-1]
            pair = pair[:, y : y + self.size, x : x + self.size]
            if turn:
                pair = pair.transpose(0, 2, 1)
            pairs.append(np.ascontiguousarray(pair[[0, -1]]))
            kinds.append(synthetic)
        return torch.from_numpy(np.stack(pairs)).cuda().float(), torch.tensor(kinds, device="cuda")


def spread(n, device, low, high):
    """n draws between low and high, even on a log scale."""
    return torch.exp(torch.empty(n, device=device).uniform_(math.log(low), math.log(high)))


def windows(fields, n, size):
    """n windows of mosaics (frames, H, W), each from a frame picked at random and starting on red.

    Returns which frames, and the windows.
    """
    frame = torch.randint(len(fields), (n,), device=fields.device)
    along = torch.arange(size, device=fields.device)
    span = lambda side: torch.randint((side - size) // 2, (n,), device=fields.device)[:, None] * 2 + along
    rows, columns = span(fields.shape[1]), span(fields.shape[2])
    return frame, fields[frame[:, None, None], rows[:, :, None], columns[:, None, :]].float()


def sites(planes, like):
    """Values per colour plane (N, 4) laid out as mosaics the shape of like (N, H, W)."""
    n, h, w = like.shape
    return unpack(planes.view(n, 4, 1, 1).expand(n, 4, h // 2, w // 2))


def bank_noise(clean, bank, offsets):
    """The GR III's real noise on clean mosaics (N, H, W) already at LEVEL brightness.

    Each crop gets a window of one ISO's unit-strength field, scaled by that ISO's shot and read noise
    at the brightness under it. The photo is taken as shot 1-16x darker than LEVEL and brought up, as
    a dim night frame is, but never past FLOOR. With offsets it keeps the black offset that ISO's
    frame sat on.
    """
    fields, shot, read, offset = bank
    n = clean.shape[0]
    iso, unit = windows(fields, n, clean.shape[-1])
    darkest = LEVEL / (FLOOR * shot[iso].mean(1))
    dim = torch.minimum(spread(n, clean.device, 1, 16), darkest).clamp(min=1).view(n, 1)
    variance = clean.clamp(min=0) * sites(shot[iso] * dim, clean) + sites(read[iso] * dim, clean) ** 2
    return clean + unit * variance.sqrt() + sites(offset[iso] * dim * offsets, clean)


def noise_bank(profile):
    """A camera's noise fields on the GPU, and its shot noise, read noise and black offset per ISO and colour plane."""
    levels = json.loads(profile.read_text())["iso"].values()
    fields = torch.from_numpy(np.load(profile.with_suffix(".npy"))).cuda()
    part = lambda kind: torch.tensor([level.get(kind, [0.0] * 4) for level in levels], device="cuda")
    return fields, part("shot"), part("read"), part("offset")


def dark_bank(path):
    """Real dark frames at unit strength on the GPU, each with its black offset per colour plane over its read noise."""
    lifts = [frame["offset"] for frame in json.loads(path.read_text())]
    return torch.from_numpy(np.load(path.with_suffix(".npy"))).cuda(), torch.tensor(lifts, device="cuda")


def measured(path):
    """A file's rows of noise levels: fit_noise.py's table of cameras, or one camera's profile by ISO."""
    found = json.loads(pathlib.Path(path).read_text())
    return found if isinstance(found, list) else list(found["iso"].values())


def table(rows):
    """Rows of noise levels on the GPU: shot, read, offset and tail per colour plane (rows, 4), the rest (rows,).

    A row is one camera at one ISO, in normalised raw units. One that lacks a part gets none of it: no
    banding, no offset, a bell curve, no rounding. Banding is what a row's lines carry beyond its
    columns, since noise that is only smeared shows in both, and no more than MOST_BANDING of the read
    noise: on a nearly clean frame the measured banding is mostly what was left of the scene.
    """
    planes = lambda value: value if isinstance(value, list) else [value] * 4
    across = lambda row: max(row.get("row", 0.0) ** 2 - row.get("column", 0.0) ** 2, 0.0) ** 0.5
    banding = lambda row: min(across(row), MOST_BANDING * float(np.mean(row["read"])))
    parts = {
        "shot": [planes(row["shot"]) for row in rows],
        "read": [planes(row["read"]) for row in rows],
        "offset": [planes(row.get("offset", 0.0)) for row in rows],
        "row": [banding(row) for row in rows],
        "tail": [planes(row.get("tail", BELL)) for row in rows],
        "step": [row.get("step", 0.0) for row in rows],
    }
    return {part: torch.tensor(values, device="cuda") for part, values in parts.items()}


def drawn(n, device, levels, share):
    """n rows of noise levels for crops at LEVEL brightness: a share of them a table's, the rest made up.

    A table's row is taken whole: the photo is taken as shot 1-16x darker than LEVEL and brought up, as
    a dim night frame is, but never past FLOOR. A made-up row has every part drawn on its own: from
    3,000 electrons at LEVEL down to FLOOR, with half to eight electrons of read noise (the GR IIIx has
    about 3), an offset of a fraction of the read noise, part shared by the colour planes and part each
    plane's own, and read noise of 1 to 300 raw counts. The ladders the table is fitted on start
    around ISO 800, so the made-up rows are what covers a nearly clean photo.
    """
    shot = spread(n, device, LEVEL / 3000, LEVEL / FLOOR)
    read = shot * spread(n, device, 0.5, 8)
    lift = torch.randn(n, 1, device=device) * OFFSET_SHARED + torch.randn(n, 4, device=device) * OFFSET_OWN
    made_up = {
        "shot": shot[:, None].expand(n, 4),
        "read": read[:, None].expand(n, 4),
        "offset": read[:, None] * lift,
        "row": read * torch.empty(n, device=device).uniform_(0, MOST_BANDING),
        "tail": torch.empty(n, 1, device=device).uniform_(*TAILS).expand(n, 4),
        "step": read / spread(n, device, *READ_COUNTS),
    }
    if not levels:
        return made_up
    pick = torch.randint(len(levels["shot"]), (n,), device=device)
    row = {part: values[pick] for part, values in levels.items()}
    dim = torch.minimum(spread(n, device, 1, 16), LEVEL / (FLOOR * row["shot"].mean(1))).clamp(min=1)
    each = lambda values: (n, *[1] * (values.dim() - 1))
    row = {part: values if part == "tail" else values * dim.view(each(values)) for part, values in row.items()}
    measured = torch.rand(n, device=device) < share
    return {part: torch.where(measured.view(each(values)), values, made_up[part]) for part, values in row.items()}


def long_tailed(like, tail):
    """Unit-strength noise the shape of like (N, H, W) from Tukey's lambda distribution, ELD's read noise.

    tail (N, 4) is each colour plane's lambda: BELL is close to a bell curve, lower has longer tails.
    Held between -0.2 and 0.25: a fit that asks for less tail than that has met rounding, not read noise.
    """
    lam = sites(tail.clamp(-0.2, 0.25), like)
    lam = torch.where(lam.abs() < 0.01, torch.full_like(lam, 0.01), lam)
    variance = 2 / lam**2 * (1 / (1 + 2 * lam) - torch.exp(2 * torch.lgamma(lam + 1) - torch.lgamma(2 * lam + 2)))
    u = torch.rand_like(like).clamp(1e-6, 1 - 1e-6)
    return (u**lam - (1 - u) ** lam) / lam / variance.sqrt()


def made_up_noise(clean, row, offsets, tails, darks):
    """Shot, read and row noise on clean mosaics (N, H, W) already at LEVEL brightness, at drawn()'s levels.

    With tails the read noise is long-tailed and the result rounded to the raw's steps, with offsets
    each colour plane keeps its black offset, and with a dark bank some crops get a window of a real
    dark frame, at the row's read level, in place of made-up read noise, banding and offset.
    """
    n, device = clean.shape[0], clean.device
    shot, read = sites(row["shot"], clean), sites(row["read"], clean)
    unit = long_tailed(clean, row["tail"]) if tails else torch.randn_like(clean)
    dark = unit * read + torch.randn(n, clean.shape[1], 1, device=device) * row["row"].view(n, 1, 1)
    lift = row["offset"]
    if darks:
        fields, lifts = darks
        frame, real = windows(fields, n, clean.shape[-1])
        swapped = torch.rand(n, device=device) < DARK_SHARE
        dark = torch.where(swapped.view(n, 1, 1), real * read, dark)
        lift = torch.where(swapped.view(n, 1), lifts[frame] * row["read"], lift)
    noisy = torch.poisson(clean.clamp(min=0) / shot) * shot + dark + sites(lift * offsets, clean)
    step = row["step"].view(n, 1, 1) * tails
    return torch.where(step > 0, torch.round(noisy / step.clamp(min=1e-12)) * step, noisy)


def gains(clean):
    """Scale each crop to LEVEL, from six times darker to twice brighter.

    denoise() scales a whole photo to LEVEL, and a night photo's tiles are mostly darker than its mean.
    """
    mean = clean.mean(dim=(1, 2)).clamp(min=1e-6)
    return (LEVEL / mean * spread(len(mean), mean.device, 1 / 6, 2)).clamp(1 / 4, MAX_GAIN).view(-1, 1, 1)


def scenes(split, pattern):
    """Each scene's name, and its json's list of real noisy frames, empty for a scene of clean frames only.

    A scene whose noisy frames have no clean twins yet, built by an older raw_pairs.py, stays out.
    """
    found, stale = {}, 0
    for meta in sorted(PAIRS.glob("*.json")):
        info = json.loads(meta.read_text())
        if info["split"] == split and fnmatch.fnmatch(meta.stem, pattern) and meta.with_suffix(".npy").exists():
            if raw_pairs.twinned(info):
                found[meta.stem] = info["frames"]
            else:
                stale += 1
    if stale:
        print(f"{stale} {split} scenes left out for want of clean twins: rerun raw_pairs.py", flush=True)
    return found


@torch.no_grad()
def validate(model, val):
    """Mean PSNR on the toned mosaic, over every held-out real noisy frame's first four crops."""
    model.eval()
    scores = []
    for scene in val:
        stack = np.load(PAIRS / f"{scene}.npy", mmap_mode="r")
        twins = stack.shape[0] // 2
        for frame in range(1, twins + 1):
            clean = torch.from_numpy(stack[frame + twins, :4, 64:448, 64:448].astype(np.float32)).cuda()
            noisy = torch.from_numpy(stack[frame, :4, 64:448, 64:448].astype(np.float32)).cuda()
            g = (LEVEL / clean.mean(dim=(1, 2)).clamp(min=1e-6)).clamp(1 / 4, MAX_GAIN).view(-1, 1, 1)
            with torch.autocast("cuda", torch.bfloat16):
                out = bench_raw.mosaiced(made(model, model(fed(model, noisy * g)).float()))
            error = ((tone(out).clamp(0, 1) - tone(clean * g).clamp(0, 1)) ** 2).mean(dim=(1, 2))
            scores += (-10 * torch.log10(error)).tolist()
    model.train()
    return float(np.mean(scores))


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = MODELS / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(vars(args)))
    camera = pathlib.Path(args.profile or NOISE / "gr3.json")
    bank = noise_bank(camera) if camera.with_suffix(".npy").exists() else None
    levels = table(measured(args.profile or args.levels)) if args.profile or args.levels else None
    darks = dark_bank(pathlib.Path(args.darks)) if args.darks else None
    share, rows = (1, 1) if args.profile else (BANK_SHARE, TABLE_SHARE)
    model = network(args.width, bool(args.joint)).cuda().to(memory_format=torch.channels_last)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        model.load_state_dict(widened(state) if args.joint and state["intro.weight"].shape[1] == 4 else state)
    else:
        sidd = torch.load(evaluate.MODELS / f"NAFNet-SIDD-width{args.width}.pth", map_location="cpu")["params"]
        model.load_state_dict({k: v for k, v in sidd.items() if not k.startswith(("intro", "ending"))}, strict=False)
    average = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0, fused=True)
    warmup = 1000
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            min(1, (s + 1) / warmup) * (0.01 + 0.99 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) / 2)
        ),
    )
    step = 0
    if (out / "last.pth").exists():
        state = torch.load(out / "last.pth", map_location="cpu")
        model.load_state_dict(state["model"])
        average.load_state_dict(state["average"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step = state["step"]
    compiled = torch.compile(model)
    bench_raw.kept("gr3")
    val = scenes("val", args.scenes)
    log = (out / "log.jsonl").open("a")
    running = torch.zeros((), device="cuda")
    train = scenes("train", args.scenes)
    crops = Crops(train, args.size, args.seed + step, f"*{camera.stem}-*" if args.profile else "*", args.offsets)

    def under_noise(clean):
        row = drawn(len(clean), clean.device, levels, rows)
        laid = made_up_noise(clean, row, args.offsets, args.tails, darks)
        if bank:
            real = torch.rand(len(clean), 1, 1, device=clean.device) < share
            laid = torch.where(real, bank_noise(clean, bank, args.offsets), laid)
        return laid

    def cleaned(noisy):
        with torch.autocast("cuda", torch.bfloat16):
            return made(model, compiled(fed(model, noisy).contiguous(memory_format=torch.channels_last)).float())

    while step < args.steps:
        pair, synthetic = crops.batch(args.batch + args.joint)
        clean, noisy = pair[: args.batch, 0], pair[: args.batch, 1]
        synthetic = synthetic[: args.batch].view(-1, 1, 1)
        g = gains(clean)
        clean, noisy = clean * g, noisy * g
        if synthetic.any():
            noisy = torch.where(synthetic, under_noise(clean), noisy)
        loss = loss_of(bench_raw.mosaiced(cleaned(noisy)), clean)
        if args.joint:
            truth = bench_raw.full_colour(pair[args.batch :, 0])
            truth = truth * gains(bench_raw.mosaiced(truth))[:, None]
            loss = loss + picture_loss(cleaned(under_noise(bench_raw.mosaiced(truth))), truth)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        schedule.step()
        step += 1
        if step % 4 == 0:
            average.update_parameters(model)
        running += loss.detach()
        if step % args.check == 0 or step == args.steps:
            clean_up = cleaner(average.module)
            bench, aligned, shot, fixed = bench_raw.scores(bench_raw.as_mosaic(clean_up), "gr3")
            seen, seen_aligned, seen_shot, seen_fixed = bench_raw.scores(bench_raw.as_mosaic(clean_up), "seen")
            mean_psnr = lambda by_iso: round(float(np.mean([psnr for psnr, _ in by_iso.values()])), 3)
            line = {
                "step": step,
                "loss": round(running.item() / args.check, 5),
                "val_psnr": round(validate(average.module, val), 3) if val else float("nan"),
                "gr3_psnr": mean_psnr(bench),
                "gr3_by_iso": [round(psnr, 3) for psnr, _ in bench.values()],
                "seen_psnr": mean_psnr(seen),
                "gr3_aligned": mean_psnr(aligned),
                "seen_aligned": mean_psnr(seen_aligned),
                "gr3_as_shot": mean_psnr(shot),
                "gr3_as_shot_by_iso": [round(psnr, 3) for psnr, _ in shot.values()],
                "seen_as_shot": mean_psnr(seen_shot),
                "gr3_as_shot_fixed": mean_psnr(fixed),
                "gr3_as_shot_fixed_by_iso": [round(psnr, 3) for psnr, _ in fixed.values()],
                "seen_as_shot_fixed": mean_psnr(seen_fixed),
                "lr": schedule.get_last_lr()[0],
                "scenes": len(train),
                "time": round(time.time()),
            }
            if args.joint:
                colour = bench_raw.scores(clean_up, "colour/gr3")[0]
                line |= {
                    "gr3_colour": mean_psnr(colour),
                    "gr3_colour_by_iso": [round(psnr, 3) for psnr, _ in colour.values()],
                    "seen_colour": mean_psnr(bench_raw.scores(clean_up, "colour/seen")[0]),
                }
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + "\n")
            log.flush()
            running.zero_()
            state = {
                "model": model.state_dict(),
                "average": average.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "step": step,
            }
            torch.save(state, out / "last.tmp")
            (out / "last.tmp").rename(out / "last.pth")
            if step % args.keep == 0:
                torch.save(average.module.state_dict(), out / f"step{step}.pth")
        if step % args.rescan == 0:
            # Crops raw_pairs.py finished meanwhile join in.
            train = scenes("train", args.scenes)
            crops.use(train)
    torch.save(average.module.state_dict(), out / ("joint.pth" if args.joint else "faithful.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="faithful")
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=384, help="sensor pixels per side")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--check", type=int, default=2000)
    parser.add_argument("--keep", type=int, default=20000, help="steps between saved snapshots")
    parser.add_argument("--rescan", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init")
    parser.add_argument(
        "--width", type=int, default=32, help="the network's width, 32 or 64: four times the work at 64"
    )
    parser.add_argument("--scenes", default="*")
    parser.add_argument("--profile")
    parser.add_argument("--levels", help="noise levels drawn as whole (camera, ISO) rows of this table")
    parser.add_argument("--offsets", action="store_true", help="noisy frames keep their black offset per colour plane")
    parser.add_argument("--tails", action="store_true", help="long-tailed read noise, rounded to the raw's steps")
    parser.add_argument("--darks", help="a bank of real dark frames for part of the read noise")
    parser.add_argument(
        "--joint", type=int, default=0, help="train the joint model, on this many crops of made truth a step"
    )
    main(parser.parse_args())
