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

Five settings bring the made-up noise closer to a real camera's, each on its own so runs compare:
--levels data/noise/cameras.json draws two thirds of its levels as whole rows, one camera at one ISO
as fit_noise.py measured it, in place of levels drawn apart; --offsets leaves each colour plane on the black offset
a real frame sits on, for the network to take off, in made-up noise and in the real noisy frames,
which raw_pairs.py stores with theirs taken off; --tails gives the read noise a real sensor's long
tails and rounds the result to the raw's steps; --darks data/noise/darks.json swaps part of the read
noise for windows of real dark frames (dark_bank.py); --mixing makes neighbouring pixels' noise move
together as far as fit_noise.py measured it to in the row --levels drew, at the same strength, which
is what a camera that smooths its raw files hands over: the Ricohs and Pentaxes at every ISO, Canons
and Sonys at their highest.

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
bench_raw.py's full-colour scores beside the others, which are of its picture sampled back to a mosaic,
with LPIPS for them, and with how much of the truth's fine detail the picture holds and how much it
made up (bench_raw.detail), as "kept" and "invented": a run that trains on LPIPS is picked by those, not by LPIPS.

Three settings change what the joint crops teach, each on its own so runs compare. A second number
after --size, as in --size 448 896, takes the joint crops at that size from raw_pairs.py's bigger
crops (data/rawbig, clean frames alone) and leaves --batch's crops as they are: 896 px makes a truth
of 448 px, four times a joint crop's picture, so --joint goes down by four for the same work.
--bin 4 bins the truth four to one, which halves the noise its frame left in it; with --size 448 896
that truth is 224 px, as without either. --perceptual 0.05 adds LPIPS, at that weight, between the
picture and its truth, both developed as bench_raw.py looks at them: it is there to make the picture
look closer to the truth where L1 settles for a blur, and the truth is all it can pull towards.

--adversarial 0.01 trains a second, small network beside the first, a patch discriminator (discriminator()),
on those same developed pictures: it learns to tell the truth's 70 px patches from the network's, and
the loss adds, at that weight, how well the network's picture passes for the truth with it. Where
L1 and LPIPS pull towards one truth, this pulls towards anything that looks like a truth, so it can
make detail up: such a model is for a second choice in the app, picked by eye, kept and invented.
The discriminator has its own optimizer, at --lr throughout, and works in float32, since the
differences it lives on are a few thousandths of white. It is in last.pth for a rerun to carry on
with, and nowhere else: joint.pth is the network alone, as without it. "critic" in the log is its
loss, 2 when it cannot tell the two apart and 0 when it always can.

--truth data/rawshift 0.25 takes that share of the joint crops from truth a sensor measured in full
colour, PixelShift200's four shots a pixel apart (pixel_shift.py), in place of truth binned from one
mosaic. It is used as it is, at the joint truth's size: nothing is binned, so it is as sharp as the
lens made it, which is what the network meets in a photo and binned truth, twice as sharp, is not.
It is one camera's 104 scenes, so a quarter and no more. --night lets SID's long exposures train too
(raw_pairs.py sid), which are kept as a split of their own so that runs without it stay as they were.

--text 0.25 lays print into that share of the joint crops' truth before it is sampled to a mosaic
(lettered()): a page of text_bank.py's rendered print, shrunk until its strokes are 1 to 4 px wide,
turned, in one colour darker or lighter than what is under it, and blurred as a lens would. Small
print is the finest detail anyone reads in a photo and the scenes hold little of it, so this is
there to show the network what a stroke looks like. A run without it is what it was.

--soften 0.3 0.9 blurs each joint crop's truth as a lens blurs, by a width drawn per crop between
those two, in px of the truth, before it is sampled to a mosaic (softened()); the loss stays against
the truth as it was. The truth and the mosaic made of it are otherwise equally sharp, so the network
hands back a picture as soft as the lens drew it: this shows it a photo softer than its answer, for
it to give back the band the lens took. A run without it is what it was.

--teacher takes a bigger joint model's weights, --width 64's, and distils it into the one being
trained: frozen, it cleans the same noisy crops of both kinds, and the loss adds picture_loss
against its picture, times --distil. Its picture is steadier than the truth, which keeps most of its
own frame's noise, and on --batch's crops it is the one target for the two colours in three a camera
did not measure. It is not compiled: without gradients it costs the CPU little to hand over, and a
recorded graph would keep its memory beside the trained network's. A run without --teacher is what it was.

--twins WEIGHTS 1 teaches the picture on real pairs in full colour, which no camera measured for them,
and against the clean frame as it was cut, where the loss they had was against the clean twin: that
frame resampled onto the noisy one as a mosaic, each colour plane on its own, which costs it 15 to
70% of its finest detail. A mosaic cannot be moved without that and a picture can, so the network's
picture of the noisy crop is moved onto the clean crop instead (moved()), by the fraction of a pixel
the two sit apart, read back from the clean crop and its stored twin (raw_pairs.apart), the one thing
the twin is still read for. On the moved picture the colour each pixel measured is held to the clean
crop itself (loss_of), and the two it did not to a frozen joint model's picture of the clean crop,
WEIGHTS, at that weight (picture_loss, the measured colour left out of it). The clean crop stops at
the brightest value its noisy crop holds, which is the white the noisy frame clipped at. Crops under
laid-over noise are as they were. Made-up noise never shows the network what a camera that smooths
its raw files does to a scene; its own frames do.

--pairs 0.5 sends that share of --batch's draws to the scenes whose real noisy frames train, in place
of PAIR_SHARE, and with --levels half of those (SMOOTH_SHARE) to the frames of cameras that smooth
their raw (smoothed()): the ones whose row of the table has a pixel's noise moving with a
neighbour's by more than TIED, a twelfth of the real noisy frames. A run without either is what it was.

denoise() scales a whole photo by one gain, so a lamp-lit part of a dark photo reaches the network
tens of times brighter than any crop in training, which a model answers with garbage. Two cures, to
compare: denoise() now holds each tile's gain down until no block of BLOCK px in it is brighter than
BRIGHTEST, and blends tiles where they meet; and --bright 0.25 shows that share of crops as such a part,
at a gain from the top of the usual range up to MAX_GAIN, the noisy one stopped at white. bright()
measures it, and every check logs it at the whole photo's gain: a flat tile at 1 to 640 times LEVEL,
and the brightest tiles of dark photos (highlights.py).

A step hands the GPU its work without stopping for it: the crops of both kinds go up in one piece,
which is the one place the CPU waits, and the small tables the loss and the joint network need stay
on the GPU. Each pass through the network, forward or backward, is recorded once as a CUDA graph and
replayed, because handed over kernel by kernel a pass costs the CPU more time than the GPU takes to
run it, and a joint step makes two of each.

The weights that get scored and saved are a running average of the trained ones, taken every fourth
step, which is steadier than any single step. Every check logs the held-out RawNIND score and
bench_raw.py's scores for picking between runs: the GR III's, its mean and each ISO from 3,200 to
102,400, and the mean over the cameras whose other scenes train. The "aligned" ones are against clean
windows resampled onto the noisy ones, the ones to pick by; the others are what older runs logged.
The "as shot" ones leave each noisy frame on its black offset, as a photo comes from the camera, and
the "as shot fixed" ones take off what blacks.py knows the body's offsets to be, as raw_denoise.py does.
run.json beside the log keeps the run's settings.

PSNR hardly tells two runs apart, so every check also logs what a person can picture of the picture
the model makes itself (bench_raw.dials, quality.py): "gr3_noise_left" on the GR III's real noisy
frames, what is left where the truth is flat as the eye weighs it at 2x, in 8-bit levels, and
"gr3_stops_cleaner", the same in stops against the noisy frame's. A joint run adds, on the
full-colour tests of the GR III and the seen cameras, "detail_kept" and "detail_invented", the share
of the truth's detail the picture holds where the truth is busy and what else it holds there, and
"finest_kept" and "finest_invented", the same of the band 2 to 3 px across. They are not "kept" and
"invented", which stay as they were: one size, over the whole picture.

--held data/bench/night/held.json leaves the scenes that file names out of training: the ones
bench_raw.py's night test scores, which a model has to meet for the first time there.
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
import text_bank

PAIRS = evaluate.DATA / "rawpairs"
NOISE = evaluate.DATA / "noise"
MODELS = evaluate.MODELS / "raw"
#: Mean a crop or photo is scaled to before the network sees it.
LEVEL = 0.1
MAX_GAIN = 64.0
#: A crop in training has its mean between LEVEL / DIM and LEVEL * BRIGHT, and no gain under LEAST_GAIN.
DIM, BRIGHT, LEAST_GAIN = 6, 2, 1 / 4
BLOCK, BRIGHTEST = 128, 1.0
#: The flat tiles bright() tries, in times LEVEL: the last is a blown tile of the darkest photo denoise() takes.
FLATS = (1, 2, 5, 10, 20, 50, 100, 200, 640)
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
TAPS = 7
NEXT_GREEN = (((0, 1, 0, -1), (0, 2, -1, 0)), ((3, 2, 0, 1), (3, 1, 1, 0)))
STROKES, SOFT, TILT, CONTRAST, REACH = (1, 4), (0.5, 1.2), 0.15, (1.5, 16), 15
EDGE = 16
SMOOTH_SHARE, TIED = 0.5, 0.1
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


def moved(picture, move):
    """Camera RGB (N, 3, H, W), each picture moved by its own (dx, dy) px of move (N, 2), so that what
    was at (x, y) is at (x + dx, y + dy), and EDGE px cut from every side.

    The move turns each frequency's phase and leaves its strength alone, so nothing is softened, where
    resampling between pixels takes most from the finest detail. A spectrum takes a picture's edges to
    meet round the back, and what one side rings into the other is in what is cut.
    """
    n, _, h, w = picture.shape
    down = torch.fft.fftfreq(h, device=picture.device).view(1, 1, h, 1)
    across = torch.fft.rfftfreq(w, device=picture.device).view(1, 1, 1, -1)
    turn = torch.exp(-2j * math.pi * (across * move[:, 0].view(n, 1, 1, 1) + down * move[:, 1].view(n, 1, 1, 1)))
    return torch.fft.irfft2(torch.fft.rfft2(picture) * turn, s=(h, w))[..., EDGE:-EDGE, EDGE:-EDGE]


@functools.cache
def sampled(h, w, device):
    """Where an RGGB mosaic (H, W) measured each colour of camera RGB (3, H, W)."""
    at = torch.zeros(3, h, w, dtype=torch.bool, device=device)
    at[0, 0::2, 0::2] = at[1, 0::2, 1::2] = at[1, 1::2, 0::2] = at[2, 1::2, 1::2] = True
    return at


def developed(result, truth):
    """Camera RGB (N, 3, H, W) and its truth, each as the picture bench_raw.score_one looks at, from -1 to 1:
    a white balance, the colour matrix, the truth's median brought to mid-grey, the tone curve."""
    n = truth.shape[0]
    draw = lambda low, high: torch.empty(n, device=truth.device).uniform_(low, high)
    balance = torch.stack([draw(*RED), torch.ones(n, device=truth.device), draw(*BLUE)], 1).view(n, 3, 1, 1)
    srgb = lambda p: torch.einsum("ij,njhw->nihw", held(TO_SRGB, p.device), p * balance)
    a, b = srgb(result), srgb(truth)
    exposure = 0.108 / b.flatten(1).median(1).values.clamp(min=1e-6).view(n, 1, 1, 1)
    look = lambda p: tone(p * exposure).clamp(0, 1) * 2 - 1
    return look(a), look(b)


def perceptual_loss(result, truth):
    """LPIPS between camera RGB (N, 3, H, W) and its truth, each developed()."""
    shown, wanted = developed(result, truth)
    return bench_raw.perceptual()(shown, wanted).mean()


def discriminator():
    """The network --adversarial trains against: pix2pix's patch discriminator, which gives every 70 px
    patch of a developed picture a score, high for the truth's and low for the network's. Each layer is
    held to a spectral norm of 1 in place of batch normalisation, which four pictures a step cannot feed."""
    conv = lambda a, b, stride: torch.nn.utils.parametrizations.spectral_norm(torch.nn.Conv2d(a, b, 4, stride, 1))
    layers = [conv(3, 64, 2), conv(64, 128, 2), conv(128, 256, 2), conv(256, 512, 1)]
    return torch.nn.Sequential(
        *[part for layer in layers for part in (layer, torch.nn.LeakyReLU(0.2, True))], conv(512, 1, 1)
    )


def critic_loss(critic, shown, wanted):
    """The hinge loss a critic learns by: the truth's patches scored over 1, the network's under -1."""
    real, made_up = critic(torch.cat([wanted, shown.detach()])).chunk(2)
    return torch.nn.functional.relu(1 - real).mean() + torch.nn.functional.relu(1 + made_up).mean()


def adversarial_loss(critic, shown):
    """What the network is after in front of a critic: its picture's patches scored as the truth's are."""
    return -critic(shown).mean()


def tile_gain(tile, gain):
    """The gain denoise() gives a tile (H, W) of a photo of that gain, and the app with it (ship_raw.py
    writes BLOCK and BRIGHTEST into the model's manifest): the photo's, held down until no whole block of
    BLOCK px in the tile is brighter than BRIGHTEST, which is where a blown block sits in a photo that
    needs no gain. Never under 1, so such a photo keeps every tile as it is."""
    h, w = (side // BLOCK * BLOCK for side in tile.shape)
    blocks = tile[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3))
    return float(np.clip(BRIGHTEST / max(float(blocks.max()), 1e-6), 1, gain))


def cleaner(model, tiled=False):
    """bench_raw.score's view of a model: mosaics (N, H, W) in, scaled by their photo's mean, and out
    cleaned mosaics, or camera RGB (N, 3, H, W) from a joint model. With tiled each mosaic gets the
    gain denoise() gives a tile."""

    def at(mosaics, gain):
        with torch.autocast("cuda", torch.bfloat16):
            parts = [made(model, model(fed(model, part * gain)).float()) for part in mosaics.split(8)]
        return torch.cat(parts) / gain

    def clean_up(mosaics, mean):
        gain = min(max(LEVEL / mean, 1), MAX_GAIN)
        if tiled:
            return torch.cat([at(tile[None], tile_gain(tile, gain)) for tile in mosaics])
        return at(mosaics, gain)

    return clean_up


@torch.no_grad()
def bright(clean_up):
    """How far clean_up moves bright areas: each of FLATS's flat tiles, and the brightest tiles of dark photos.

    A flat tile is a tile of the darkest photo denoise() takes, the one whose gain is MAX_GAIN, and its
    number is the mean distance of what comes back from the flat, as a share of it. A photo's tile
    (highlights.py) has no clean twin, but a bright area holds little noise against its light, so its
    number is how far the means of 16 px blocks move, as a share of them, over the blocks the photo's
    gain shows brighter than training's brightest crop.
    """
    darkest = LEVEL / MAX_GAIN
    flats = [torch.full((1, 256, 256), level * darkest).cuda() for level in FLATS]
    found = {
        "flats": [
            ((bench_raw.mosaiced(clean_up(flat, darkest)) - flat).abs().mean() / flat.mean()).item() for flat in flats
        ]
    }
    path = bench_raw.BENCH / "bright.npz"
    if path.exists():
        data = np.load(path)
        blocks = lambda mosaic: torch.nn.functional.avg_pool2d(pack(mosaic), 8)
        moved = []
        for tile, mean in zip(torch.from_numpy(data["tiles"].astype(np.float32)).cuda(), data["means"].tolist()):
            seen, made_of = blocks(tile[None]), blocks(bench_raw.mosaiced(clean_up(tile[None], mean)))
            lit = seen * min(max(LEVEL / mean, 1), MAX_GAIN) > BRIGHT * LEVEL
            moved.append(((made_of - seen).abs() / seen)[lit].mean().item())
        found["tiles"] = float(np.mean(moved))
    return found


@torch.no_grad()
def denoise(model, mosaic, tile=1024, overlap=64, window=None):
    """A whole normalised RGGB mosaic (H, W float32 numpy) through the model, in overlapping tiles.

    The photo's mean sets the gain, as a crop's does in training, and tile_gain() holds it down for a
    tile with a light in it. Tiles then differ in gain, so they fade into each other over
    the middle half of what they share; the last row and column of tiles sit flush with the far edge.
    Returns the cleaned mosaic, or camera RGB (3, H, W) from a joint model. With window (y, x, height,
    width) only the tiles that reach it run and only it comes back, the same pixels the whole photo gives.
    """
    top, left, height, width = window or (0, 0, *mosaic.shape)
    gain = float(np.clip(LEVEL / max(float(mosaic.mean()), 1e-6), 1, MAX_GAIN))
    padded = np.pad(mosaic, overlap, "reflect")
    out = np.zeros((3, *padded.shape) if joint(model) else padded.shape, padded.dtype)
    weight = np.zeros(padded.shape, padded.dtype)
    inward = np.minimum(np.arange(tile), np.arange(tile)[::-1]) + 0.5
    fade = np.clip((inward - overlap / 2) / overlap, 0, 1).astype(padded.dtype)
    fade = np.outer(fade, fade)
    every = lambda side: sorted({*range(0, side - tile, tile - 2 * overlap), side - tile})
    starts = lambda side, at, span: [s for s in every(side) if s < at + span + overlap and at + overlap < s + tile]
    for y in starts(padded.shape[0], top, height):
        for x in starts(padded.shape[1], left, width):
            piece = padded[y : y + tile, x : x + tile]
            own = tile_gain(piece, gain)
            shown = torch.from_numpy(np.ascontiguousarray(piece * own)).cuda()[None]
            with torch.autocast("cuda", torch.bfloat16):
                result = made(model, model(fed(model, shown)).float())[0].cpu().numpy()
            out[..., y : y + tile, x : x + tile] += result / own * fade
            weight[y : y + tile, x : x + tile] += fade
    kept = slice(top + overlap, top + overlap + height), slice(left + overlap, left + overlap + width)
    return out[(..., *kept)] / weight[kept]


class Crops:
    """Endless batches of (clean, noisy) mosaics, flipped and turned with the colours kept on RGGB.

    One thread reads: it fills the pool, then swaps READS crops a second. A crop is drawn many times
    before it leaves, each time under another window, flip and noise.

    With twins the clean mosaic of a real noisy crop is its clean frame as it was cut, stopped at the
    brightest value the noisy crop holds, in place of the clean twin, and the crop brings how far it
    sits off that frame as (dx, dy) sensor pixels, flipped and turned with it. With pairs that share
    of the draws goes to the scenes whose real noisy frames train, SMOOTH_SHARE of it to their frames
    in smooth, a set of (scene, ISO), when there are some.
    """

    folder, room, reads, frames, whole = PAIRS, POOL, READS, [0, -1], None

    def __init__(self, scenes, size, seed, real, as_shot, twins=False, pairs=0, smooth=frozenset()):
        self.size, self.real, self.as_shot, self.rng, self.pool = size, real, as_shot, random.Random(seed), []
        self.twins, self.pairs, self.smooth = twins, pairs, smooth
        self.use(scenes)
        threading.Thread(target=self.read, daemon=True).start()

    def use(self, scenes):
        weights = [next((w for start, w in WEIGHTS.items() if s.startswith(start)), 1) for s in scenes]
        paired = [bool(frames) for frames in scenes.values()]
        total = {kind: sum(w for w, p in zip(weights, paired) if p == kind) for kind in (True, False)}
        weights = [w * (PAIR_SHARE if p else 1 - PAIR_SHARE) / total[p] for w, p in zip(weights, paired)]
        draws = [(scene, weight, None) for scene, weight in zip(scenes, weights)]
        self.scenes, self.draw = scenes, tuple(zip(*(self.apportioned(scenes, draws) if self.pairs else draws)))

    def apportioned(self, scenes, draws):
        """The draws with self.pairs of them sent to the scenes whose real noisy frames train: such a
        scene once for its frames in self.smooth, which get SMOOTH_SHARE of that between them, and once
        for its other frames, each kind shared out among its own as the scenes were."""
        kinds = {"clean": [], "smooth": [], "plain": []}
        for scene, weight, _ in draws:
            frames = scenes[scene] if fnmatch.fnmatch(scene, self.real) else []
            smooth = [k for k, frame in enumerate(frames, 1) if (scene, frame["iso"]) in self.smooth]
            plain = [k for k in range(1, len(frames) + 1) if k not in smooth]
            if not frames:
                kinds["clean"].append((scene, weight, None))
            if smooth:
                kinds["smooth"].append((scene, weight, smooth))
            if plain:
                kinds["plain"].append((scene, weight, plain))
        some = SMOOTH_SHARE if kinds["smooth"] and kinds["plain"] else float(bool(kinds["smooth"]))
        shares = {"clean": 1 - self.pairs, "smooth": self.pairs * some, "plain": self.pairs * (1 - some)}
        totals = {kind: sum(weight for _, weight, _ in found) for kind, found in kinds.items()}
        return [
            (scene, weight * shares[kind] / totals[kind], among)
            for kind, found in kinds.items()
            for scene, weight, among in found
        ]

    def read(self):
        rng = random.Random(self.rng.random())
        while True:
            names, weights, among = self.draw
            at = rng.choices(range(len(names)), weights)[0]
            scene, among = names[at], among[at]
            path = self.folder / f"{scene}.npy"
            stack = np.load(path, mmap_mode="r")
            frames, crops, side, _ = stack.shape
            crop = rng.randrange(crops)
            noisy = frames // 2
            synthetic = not noisy or not fnmatch.fnmatch(scene, self.real) or rng.random() < SYNTHETIC_SHARE
            frame = 0 if synthetic else rng.choice(among) if among else rng.randrange(1, noisy + 1)
            twinned = self.twins and not synthetic
            picks = self.whole or (
                [0] if synthetic else [frame + noisy, 0, frame] if twinned else [frame + noisy, frame]
            )
            # Read by offset: through the memory map the kernel fetches about three times what a crop holds.
            starts = [stack.offset + (i * crops + crop) * side * side * stack.itemsize for i in picks]
            planes = [np.fromfile(path, stack.dtype, side * side, offset=start) for start in starts]
            pair = np.stack(planes).reshape(-1, side, side)
            move = (0.0, 0.0)
            if twinned:
                twin, clean, shot = pair
                whites = [shot[i::2, j::2].max() for i in range(2) for j in range(2)]
                clean = raw_pairs.capped(clean[None], whites)[0]
                pair, move = np.stack([clean, shot]), raw_pairs.apart(clean, twin)
            if self.as_shot and not synthetic:
                black = self.scenes[scene][frame - 1].get("offset", [0.0] * 4)
                pair[1] += np.tile(np.asarray(black, pair.dtype).reshape(2, 2), (side // 2, side // 2))
            fresh = pair, synthetic, move
            if len(self.pool) < self.room:
                self.pool.append(fresh)
            else:
                self.pool[rng.randrange(self.room)] = fresh
            time.sleep(1 / self.reads if len(self.pool) == self.room else 0.05)

    def batch(self, n):
        while len(self.pool) < self.room // 4:
            time.sleep(1)
        pairs, kinds, moves = [], [], []
        for _ in range(n):
            pair, synthetic, (dx, dy) = self.rng.choice(self.pool)
            side = pair.shape[-1]
            flip_y, flip_x, turn = (self.rng.random() < 0.5 for _ in range(3))
            # A flipped mosaic starts on red again one pixel in, so the window's corner takes that parity.
            y = self.rng.randrange((side - self.size) // 2) * 2 + flip_y
            x = self.rng.randrange((side - self.size) // 2) * 2 + flip_x
            if flip_y:
                pair, dy = pair[:, ::-1], -dy
            if flip_x:
                pair, dx = pair[:, :, ::-1], -dx
            pair = pair[:, y : y + self.size, x : x + self.size]
            if turn:
                pair, dx, dy = pair.transpose(0, 2, 1), dy, dx
            pairs.append(np.ascontiguousarray(pair[self.frames]))
            kinds.append(synthetic)
            moves.append((dx, dy))
        crops = torch.from_numpy(np.stack(pairs)).cuda().float()
        return crops, torch.tensor(kinds, device="cuda"), torch.tensor(moves, device="cuda")


class BigCrops(Crops):
    """The same from raw_pairs.py's bigger crops: clean frames alone, four times the crop, so a quarter as many kept and read."""

    folder, room, reads, frames = raw_pairs.BIG, POOL // 4, READS / 4, [0]


class TruthCrops(Crops):
    """The same from a folder of truth measured in full colour (pixel_shift.py): a crop's red, green and
    blue where a scene's frames are, all three read, and an eighth as many kept."""

    room, reads, frames, whole = POOL // 8, READS / 8, [0, 1, 2], [0, 1, 2]

    def __init__(self, folder, *settings):
        self.folder = folder
        super().__init__(*settings)


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


def smoothed(rows):
    """The real noisy frames of cameras that smooth their raw, as (scene, ISO): the rows of fit_noise.py's
    table in which a pixel's noise moves with a neighbour's, the green beside it or its own colour's
    next pixel across or down, by more than TIED. A row is one scene's frame, named as raw_pairs.py
    names a review site's ladder."""
    tie = lambda row: max(max(row.get(part, [0.0])) for part in ("beside", "across", "down"))
    return frozenset(
        (f"ladder-{row['site']}-{row['camera']}-{row['scene']}", row["iso"])
        for row in rows
        if "site" in row and tie(row) > TIED
    )


def mixers(rows):
    """Each row's filter (rows, 4, 4, TAPS, TAPS) that mixes white noise over the colour planes into noise
    whose neighbours move together as far as fit_noise.py measured: red and blue with each green that
    touches them, each plane with its next pixel across and down, and nothing else. A red or a blue
    pixel has a green on all four sides: NEXT_GREEN lists, for red and then for blue, its plane, each
    green plane, and how many cells down and across the green on the other side sits, the first being
    in its own cell.

    Those correlations are a covariance between the planes at each lag, and the filter is its square
    root, taken where a root of one is plain, frequency by frequency, over TAPS pixels of a plane a
    side. Correlations no noise could have all at once lose what is impossible there, and every
    plane's taps are brought back to unit strength, so mixing never changes how strong the noise is.
    A row that lacks them gets none.
    """
    part = lambda key, planes: np.array([row.get(key, [0.0] * planes) for row in rows])
    beside, across, down = part("beside", 2), part("across", 4), part("down", 4)
    wanted = np.zeros((len(rows), 4, 4, TAPS, TAPS))
    planes = np.arange(4)
    wanted[:, planes, planes, 0, 0] = 1
    wanted[:, planes, planes, 0, 1] = wanted[:, planes, planes, 0, -1] = across
    wanted[:, planes, planes, 1, 0] = wanted[:, planes, planes, -1, 0] = down
    for sides, alike in zip(NEXT_GREEN, beside.T):
        for colour, green, cells_down, cells_across in sides:
            for y, x in ((0, 0), (cells_down, cells_across)):
                wanted[:, colour, green, y, x] = wanted[:, green, colour, -y, -x] = alike
    size, turn = np.linalg.eigh(np.fft.fft2(wanted).transpose(0, 3, 4, 1, 2))
    root = (turn * np.sqrt(size.clip(min=0))[..., None, :]) @ turn.conj().swapaxes(-1, -2)
    taps = np.fft.fftshift(np.fft.ifft2(root.transpose(0, 3, 4, 1, 2)).real, axes=(-2, -1))
    return (taps / np.sqrt((taps**2).sum((2, 3, 4), keepdims=True))).astype(np.float32)


def mixed(unit, taps):
    """Unit-strength noise (N, H, W), each mosaic's colour planes run through its filter of mixers().

    A plane's edges meet round the back, so the noise at an edge is as strong and as mixed as elsewhere.
    """
    n, h, w = unit.shape
    planes = torch.nn.functional.pad(pack(unit), [TAPS // 2] * 4, "circular").flatten(0, 1)[None]
    return unpack(torch.nn.functional.conv2d(planes, taps.flatten(0, 1), groups=n).view(n, 4, h // 2, w // 2))


def table(rows, mixing=False):
    """Rows of noise levels on the GPU: shot, read, offset and tail per colour plane (rows, 4), the rest (rows,),
    and with mixing each row's filter of mixers().

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
    found = {part: torch.tensor(values, device="cuda") for part, values in parts.items()}
    return found | {"mix": torch.tensor(mixers(rows), device="cuda")} if mixing else found


def drawn(n, device, levels, share):
    """n rows of noise levels for crops at LEVEL brightness: a share of them a table's, the rest made up.

    A table's row is taken whole: the photo is taken as shot 1-16x darker than LEVEL and brought up, as
    a dim night frame is, but never past FLOOR. A made-up row has every part drawn on its own: from
    3,000 electrons at LEVEL down to FLOOR, with half to eight electrons of read noise (the GR IIIx has
    about 3), an offset of a fraction of the read noise, part shared by the colour planes and part each
    plane's own, and read noise of 1 to 300 raw counts. The ladders the table is fitted on start
    around ISO 800, so the made-up rows are what covers a nearly clean photo. A made-up row has no
    mixing of its own and keeps that of the table's row it stands in for, so it draws from the table's
    spread, where most rows have none.
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
    unscaled = lambda part, values: values if part in ("tail", "mix") else values * dim.view(each(values))
    row = {part: unscaled(part, values) for part, values in row.items()}
    measured = torch.rand(n, device=device) < share
    return row | {part: torch.where(measured.view(each(values)), row[part], values) for part, values in made_up.items()}


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
    dark frame, at the row's read level, in place of made-up read noise, banding and offset. A row
    with a mixing filter has its shot and read noise, a dark frame's too, run through it: each
    pixel's is brought to unit strength first and scaled back after, so banding and offset stay as
    they are and no pixel's noise changes strength.
    """
    n, device = clean.shape[0], clean.device
    shot, read = sites(row["shot"], clean), sites(row["read"], clean)
    unit = long_tailed(clean, row["tail"]) if tails else torch.randn_like(clean)
    banding = torch.randn(n, clean.shape[1], 1, device=device) * row["row"].view(n, 1, 1)
    lift = row["offset"]
    if darks:
        fields, lifts = darks
        frame, real = windows(fields, n, clean.shape[-1])
        swapped = torch.rand(n, device=device) < DARK_SHARE
        unit = torch.where(swapped.view(n, 1, 1), real, unit)
        banding = banding * ~swapped.view(n, 1, 1)
        lift = torch.where(swapped.view(n, 1), lifts[frame] * row["read"], lift)
    light = clean.clamp(min=0)
    counted, dark = torch.poisson(light / shot) * shot, unit * read + banding
    if "mix" in row:
        strength = (light * shot + read**2).sqrt().clamp(min=1e-12)
        counted = light + mixed((counted - light + unit * read) / strength, row["mix"]) * strength
        dark = banding
    noisy = counted + dark + sites(lift * offsets, clean)
    step = row["step"].view(n, 1, 1) * tails
    return torch.where(step > 0, torch.round(noisy / step.clamp(min=1e-12)) * step, noisy)


def lettered(truth, pages):
    """Camera RGB (N, 3, H, H) with print laid into each: one of text_bank.py's pages (pages, P, P) of coverage.

    The page is shrunk until its strokes are between STROKES px wide, even on a log scale, and set
    down with its middle anywhere in the crop, at any quarter turn and TILT radians or so off it.
    It is blurred by between SOFT px, so that it is no sharper than a lens draws, and on the page,
    before the shrinking, which would otherwise break thin strokes up; the taps reach REACH px of the
    page. The ink is one colour: the mean of what lies under the letters, between CONTRAST times
    darker or, as often, lighter, each colour by its own share of that, and never past white.
    """
    n, side, device = len(truth), truth.shape[-1], truth.device
    page = pages[torch.randint(len(pages), (n,), device=device)].float() / 255
    shrunk = text_bank.STROKE / spread(n, device, *STROKES)
    soft = torch.empty(n, 1, device=device).uniform_(*SOFT) * shrunk[:, None]
    taps = torch.exp(-(torch.arange(-REACH, REACH + 1, device=device) ** 2) / (2 * soft**2))
    taps = taps / taps.sum(1, keepdim=True)
    page = torch.nn.functional.conv2d(page[None], taps.view(n, 1, 1, -1), padding=(0, REACH), groups=n)
    page = torch.nn.functional.conv2d(page, taps.view(n, 1, -1, 1), padding=(REACH, 0), groups=n)
    turn = torch.randint(4, (n,), device=device) * (math.pi / 2) + torch.randn(n, device=device) * TILT
    spin = torch.stack([turn.cos(), -turn.sin(), turn.sin(), turn.cos()], 1).view(n, 2, 2)
    spin = spin * (shrunk * side / pages.shape[-1]).view(n, 1, 1)
    middle = torch.empty(n, 2, 1, device=device).uniform_(-1, 1)
    placed = torch.cat([spin, -spin @ middle], 2)
    grid = torch.nn.functional.affine_grid(placed, (n, 1, side, side), align_corners=False)
    cover = torch.nn.functional.grid_sample(page.transpose(0, 1), grid, align_corners=False)
    under = (truth * cover).sum((2, 3), keepdim=True) / cover.sum((2, 3), keepdim=True).clamp(min=1e-6)
    share = 1 - torch.rand(n, 1, 1, 1, device=device) * torch.rand(n, 3, 1, 1, device=device)
    contrast = spread(n, device, *CONTRAST).view(n, 1, 1, 1) ** share
    dark = torch.rand(n, 1, 1, 1, device=device) < 0.5
    return torch.lerp(truth, torch.where(dark, under / contrast, (under * contrast).clamp(max=1)), cover)


def softened(truth, low, high):
    """Camera RGB (N, 3, H, W), each blurred as a lens blurs it: a Gaussian between low and high px wide, drawn per crop.

    A tap is the Gaussian's share of one pixel, so an edge between two pixels comes out as that lens
    draws it; taps read off the curve at each pixel's middle would hardly blur under half a pixel.
    The picture is mirrored past its edges.
    """
    n, reach = len(truth), math.ceil(3 * high)
    width = torch.empty(n, 1, device=truth.device).uniform_(low, high)
    sides = torch.arange(-reach, reach + 2, device=truth.device) - 0.5
    taps = torch.erf(sides / (width * math.sqrt(2))).diff(dim=1)
    taps = (taps / taps.sum(1, keepdim=True)).repeat_interleave(3, 0)
    mirrored = torch.nn.functional.pad(truth, (reach,) * 4, mode="reflect").flatten(0, 1)[None]
    across = torch.nn.functional.conv2d(mirrored, taps.view(-1, 1, 1, 2 * reach + 1), groups=3 * n)
    return torch.nn.functional.conv2d(across, taps.view(-1, 1, 2 * reach + 1, 1), groups=3 * n).view_as(truth)


def gains(clean, lit=0.0):
    """Scale each crop to LEVEL, from six times darker to twice brighter.

    denoise() scales a whole photo to LEVEL, and a night photo's tiles are mostly darker than its mean.
    Its lamps are not: with lit, that share of crops gets a gain from the top of that range up to
    MAX_GAIN, as a lit part of a dark photo does, and what the camera clipped then sits at the gain.
    """
    mean = clean.mean(dim=(1, 2)).clamp(min=1e-6)
    gain = (LEVEL / mean * spread(len(mean), mean.device, 1 / DIM, BRIGHT)).clamp(LEAST_GAIN, MAX_GAIN)
    if lit:
        least = (BRIGHT * LEVEL / mean).clamp(1, MAX_GAIN)
        lamp = least * (MAX_GAIN / least) ** torch.rand_like(mean)
        gain = torch.where(torch.rand_like(mean) < lit, lamp, gain)
    return gain.view(-1, 1, 1)


def scenes(split, pattern, folder=PAIRS, left_out=()):
    """Each scene's name, and its json's list of real noisy frames, empty for a scene of clean frames only.

    A scene whose noisy frames have no clean twins yet, built by an older raw_pairs.py, stays out,
    and so does one that left_out names.
    """
    found, stale = {}, 0
    for meta in sorted(folder.glob("*.json")):
        info = json.loads(meta.read_text())
        wanted = info["split"] == split and fnmatch.fnmatch(meta.stem, pattern) and meta.stem not in left_out
        if wanted and meta.with_suffix(".npy").exists():
            if raw_pairs.twinned(info):
                found[meta.stem] = info["frames"]
            else:
                stale += 1
    if stale:
        print(f"{stale} {split} scenes left out for want of clean twins: rerun raw_pairs.py", flush=True)
    return found


def mean_of(by_iso, part, digits):
    """One part of a test's rows, bench_raw.scores's {ISO: [PSNR, LPIPS, ...]}, as its mean over the ISOs."""
    return round(float(np.mean([row[part] for row in by_iso.values()])), digits)


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
    if args.joint and args.size[-1] % (32 * args.bin):
        raise SystemExit("the joint crops' size has to be a multiple of 32 times --bin")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out = MODELS / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(vars(args)))
    camera = pathlib.Path(args.profile or NOISE / "gr3.json")
    bank = noise_bank(camera) if camera.with_suffix(".npy").exists() else None
    if args.mixing and not args.levels:
        raise SystemExit("--mixing mixes as the rows of --levels' table say")
    levels = table(measured(args.profile or args.levels), args.mixing) if args.profile or args.levels else None
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
    if args.adversarial and not args.joint:
        raise SystemExit("--adversarial judges the joint crops' picture, for a --joint run")
    critic = discriminator().cuda() if args.adversarial else None
    critic_optimizer = (
        torch.optim.AdamW(critic.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0, fused=True)
        if critic
        else None
    )
    step = 0
    if (out / "last.pth").exists():
        state = torch.load(out / "last.pth", map_location="cpu")
        model.load_state_dict(state["model"])
        average.load_state_dict(state["average"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step = state["step"]
        if critic:
            critic.load_state_dict(state["critic"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
    compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
    teacher = trained(args.teacher).to(memory_format=torch.channels_last) if args.teacher else None
    if teacher and not (args.joint and joint(teacher)):
        raise SystemExit("--teacher is a joint model's weights, for a --joint run")
    unbinned = float(args.truth[1]) if args.truth else 0
    if not 0 <= args.joint * unbinned <= max(args.joint - 1, 0):
        raise SystemExit("--truth's share has to leave one of --joint's crops to the binned truth")
    if args.text and not args.joint:
        raise SystemExit("--text lays print into the joint crops' truth, for a --joint run")
    if args.soften and not args.joint:
        raise SystemExit("--soften blurs the joint crops' truth, for a --joint run")
    pages = torch.from_numpy(np.load(text_bank.OUT)).cuda() if args.text else None
    twin = trained(args.twins[0]).to(memory_format=torch.channels_last) if args.twins else None
    if twin and not (args.joint and joint(twin)):
        raise SystemExit("--twins is a joint model's weights, for a --joint run")
    bench_raw.kept("gr3")
    val = scenes("val", args.scenes)
    log = (out / "log.jsonl").open("a")
    running, judging = torch.zeros((), device="cuda"), torch.zeros((), device="cuda")
    left_out = frozenset(json.loads(pathlib.Path(args.held).read_text())) if args.held else ()

    def training(*folder):
        found = scenes("train", args.scenes, *folder, left_out=left_out)
        return {**found, **scenes("night", args.scenes, *folder)} if args.night else found

    train = training()
    own = f"*{camera.stem}-*" if args.profile else "*"
    smooth = smoothed(measured(args.levels)) if args.pairs and args.levels else frozenset()
    crops = Crops(train, args.size[0], args.seed + step, own, args.offsets, bool(twin), args.pairs, smooth)
    big = None
    if args.size[1:]:
        wide = training(raw_pairs.BIG)
        if not wide:
            raise SystemExit(f"no crops in {raw_pairs.BIG}: uv run raw_pairs.py 1 0 raise big")
        big = BigCrops(wide, args.size[1], args.seed + step, "*", False)
    truths, lot = None, random.Random(args.seed + step)
    if unbinned:
        folder = pathlib.Path(args.truth[0])
        sharp = scenes("train", args.scenes, folder)
        if not sharp:
            raise SystemExit(f"no crops in {folder}: uv run pixel_shift.py")
        truths = TruthCrops(folder, sharp, args.size[-1] // args.bin, args.seed + step, "*", False)

    def under_noise(clean):
        row = drawn(len(clean), clean.device, levels, rows)
        laid = made_up_noise(clean, row, args.offsets, args.tails, darks)
        if bank:
            real = torch.rand(len(clean), 1, 1, device=clean.device) < share
            laid = torch.where(real, bank_noise(clean, bank, args.offsets), laid)
        return laid

    def at_white(noisy, gain):
        return torch.minimum(noisy, gain) if args.bright else noisy

    def cleaned(noisy):
        with torch.autocast("cuda", torch.bfloat16):
            return made(model, compiled(fed(model, noisy).contiguous(memory_format=torch.channels_last)).float())

    @torch.no_grad()
    def taught(by, noisy):
        with torch.autocast("cuda", torch.bfloat16):
            return made(by, by(fed(by, noisy).contiguous(memory_format=torch.channels_last)).float())

    def twinned(picture, clean, real, moves):
        onto = moved(picture[real], -moves[real])
        inner = clean[real][..., EDGE:-EDGE, EDGE:-EDGE]
        wanted = taught(twin, clean[real])[..., EDGE:-EDGE, EDGE:-EDGE]
        unmeasured = torch.where(sampled(*inner.shape[-2:], inner.device), wanted, onto)
        found = loss_of(bench_raw.mosaiced(onto), inner) + float(args.twins[1]) * picture_loss(unmeasured, wanted)
        if real.all():
            return found
        share = real.float().mean()
        return share * found + (1 - share) * loss_of(bench_raw.mosaiced(picture[~real]), clean[~real])

    while step < args.steps:
        taken = int(args.joint * unbinned + lot.random()) if truths else 0
        pair, synthetic, moves = crops.batch(args.batch if big else args.batch + args.joint - taken)
        alone = big.batch(args.joint - taken)[0][:, 0] if big else pair[args.batch :, 0]
        clean, noisy = pair[: args.batch, 0], pair[: args.batch, 1]
        synthetic = synthetic[: args.batch].view(-1, 1, 1)
        g = gains(clean, args.bright)
        clean, noisy = clean * g, noisy * g
        if synthetic.any():
            noisy = torch.where(synthetic, under_noise(clean), noisy)
        noisy = at_white(noisy, g)
        picture = cleaned(noisy)
        if twin and not synthetic.all():
            loss = twinned(picture, clean, ~synthetic.view(-1), moves[: args.batch])
        else:
            loss = loss_of(bench_raw.mosaiced(picture), clean)
        if teacher:
            loss = loss + args.distil * picture_loss(picture, taught(teacher, noisy))
        if args.joint:
            truth = bench_raw.full_colour(alone, args.bin)
            if taken:
                truth = torch.cat([truth, truths.batch(taken)[0]])
            printed = int(args.joint * args.text + lot.random()) if args.text else 0
            if printed:
                truth = torch.cat([lettered(truth[:printed], pages), truth[printed:]])
            g = gains(bench_raw.mosaiced(truth), args.bright)
            truth = truth * g[:, None]
            soft = softened(truth, *args.soften) if args.soften else truth
            noisy = at_white(under_noise(bench_raw.mosaiced(soft)), g)
            picture = cleaned(noisy)
            loss = loss + picture_loss(picture, truth)
            if teacher:
                loss = loss + args.distil * picture_loss(picture, taught(teacher, noisy))
            if args.perceptual:
                loss = loss + args.perceptual * perceptual_loss(picture, truth)
            if critic:
                shown, wanted = developed(picture, truth)
                critic.requires_grad_(False)
                loss = loss + args.adversarial * adversarial_loss(critic, shown)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if critic:
            critic.requires_grad_(True)
            doubt = critic_loss(critic, shown, wanted)
            critic_optimizer.zero_grad(set_to_none=True)
            doubt.backward()
            critic_optimizer.step()
            judging += doubt.detach()
        schedule.step()
        step += 1
        if step % 4 == 0:
            average.update_parameters(model)
        running += loss.detach()
        if step % args.check == 0 or step == args.steps:
            clean_up = cleaner(average.module)
            bench, aligned, shot, fixed = bench_raw.scores(bench_raw.as_mosaic(clean_up), "gr3")
            seen, seen_aligned, seen_shot, seen_fixed = bench_raw.scores(bench_raw.as_mosaic(clean_up), "seen")
            mean_psnr = lambda by_iso: mean_of(by_iso, 0, 3)
            mean_lpips = lambda by_iso: mean_of(by_iso, 1, 4)
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
            lit = bright(clean_up)
            line |= {"bright_flats": [round(error, 4) for error in lit["flats"]]}
            line |= {"bright_tiles": round(lit["tiles"], 4)} if "tiles" in lit else {}
            look = bench_raw.dials_one(clean_up, bench_raw.kept("gr3"))
            line |= {"gr3_noise_left": mean_of(look, 0, 4), "gr3_stops_cleaner": mean_of(look, 1, 3)}
            if args.joint:
                colour = bench_raw.scores(clean_up, "colour/gr3", with_lpips=True, with_detail=True)[0]
                seen_colour = bench_raw.scores(clean_up, "colour/seen", with_lpips=True, with_detail=True)[0]
                line |= {
                    "gr3_colour": mean_psnr(colour),
                    "gr3_colour_by_iso": [round(row[0], 3) for row in colour.values()],
                    "seen_colour": mean_psnr(seen_colour),
                    "gr3_colour_lpips": mean_lpips(colour),
                    "gr3_colour_lpips_by_iso": [round(row[1], 4) for row in colour.values()],
                    "seen_colour_lpips": mean_lpips(seen_colour),
                    "gr3_colour_kept": mean_of(colour, 2, 4),
                    "gr3_colour_invented": mean_of(colour, 3, 4),
                    "seen_colour_kept": mean_of(seen_colour, 2, 4),
                    "seen_colour_invented": mean_of(seen_colour, 3, 4),
                }
                looks = {
                    "gr3_colour": bench_raw.dials_one(clean_up, bench_raw.kept("colour/gr3")),
                    "seen_colour": bench_raw.dials(clean_up, "colour/seen"),
                }
                parts = ("detail_kept", "finest_kept", "detail_invented", "finest_invented")
                line |= {
                    f"{test}_{part}": mean_of(look, 2 + at, 4)
                    for test, look in looks.items()
                    for at, part in enumerate(parts)
                }
            if critic:
                line |= {"critic": round(judging.item() / args.check, 5)}
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + "\n")
            log.flush()
            running.zero_()
            judging.zero_()
            state = {
                "model": model.state_dict(),
                "average": average.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "step": step,
            }
            if critic:
                state |= {"critic": critic.state_dict(), "critic_optimizer": critic_optimizer.state_dict()}
            torch.save(state, out / "last.tmp")
            (out / "last.tmp").rename(out / "last.pth")
            if step % args.keep == 0:
                torch.save(average.module.state_dict(), out / f"step{step}.pth")
        if step % args.rescan == 0:
            # Crops raw_pairs.py finished meanwhile join in.
            train = training()
            crops.use(train)
            if big:
                big.use(training(raw_pairs.BIG))
    torch.save(average.module.state_dict(), out / ("joint.pth" if args.joint else "faithful.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="faithful")
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[384],
        help="sensor pixels per side; a second number is the joint crops' own, taken from the bigger crops",
    )
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
        "--mixing", action="store_true", help="neighbouring pixels' noise moves together as in --levels' rows"
    )
    parser.add_argument(
        "--joint", type=int, default=0, help="train the joint model, on this many crops of made truth a step"
    )
    parser.add_argument("--bin", type=int, default=2, choices=(2, 4), help="sensor pixels a side to one of the truth")
    parser.add_argument("--perceptual", type=float, default=0, help="weight of LPIPS on the joint crops' picture")
    parser.add_argument(
        "--adversarial", type=float, default=0, help="weight of a patch discriminator's verdict on that picture"
    )
    parser.add_argument("--bright", type=float, default=0, help="share of crops shown as a lit part of a dark photo")
    parser.add_argument(
        "--truth",
        nargs=2,
        metavar=("FOLDER", "SHARE"),
        help="a folder of truth measured in full colour, and the share of the joint crops taken from it unbinned",
    )
    parser.add_argument("--text", type=float, default=0, help="share of the joint crops with print laid into the truth")
    parser.add_argument(
        "--soften",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="the joint crops' truth is blurred by between these px before it is sampled to a mosaic",
    )
    parser.add_argument("--night", action="store_true", help="SID's long exposures train too, the split night")
    parser.add_argument(
        "--twins",
        nargs=2,
        metavar=("WEIGHTS", "WEIGHT"),
        help="real pairs teach the picture against the clean frame as cut, and against these weights' picture of it",
    )
    parser.add_argument(
        "--pairs", type=float, default=0, help="share of --batch's draws from the scenes whose real noisy frames train"
    )
    parser.add_argument("--teacher", help="a bigger joint model's weights, to distil into this one")
    parser.add_argument("--distil", type=float, default=1, help="weight of the loss against the teacher's picture")
    parser.add_argument("--held", help="a list of scenes to leave out of training, as bench_raw.py night writes")
    main(parser.parse_args())
