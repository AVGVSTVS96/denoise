"""Every ladder camera's noise at every ISO, for train_raw.py --levels to draw whole rows from.

    uv run fit_noise.py 3 0        the first of three workers, one file a ladder into data/noise/fit
    uv run fit_noise.py table      gather them into data/noise/cameras.json

A row is one noisy frame against its camera's lowest ISO. Per colour plane: shot and read noise, the
black offset, how long the read noise's tails are (a Tukey lambda: 0.14 is a bell curve, lower is
longer) and the kurtosis of all the noise in the shadows that it was fitted to. For the frame: row
banding, the raw's rounding step, and how far the noise of neighbouring pixels moves together, which a
camera that smooths its raw files leaves behind. All in the noisy frame's own units, black-subtracted
over the white level, as noise_bank.py's are. The cameras bench_raw.py tests on stay out, as they do
in raw_pairs.py.
"""

import json
import sys

import cv2
import numpy as np
import rawpy
import scipy.optimize
import scipy.stats

import raw_pairs

OUT = raw_pairs.DATA / "noise"
FITS = OUT / "fit"
SITES = ((0, 0), (0, 1), (1, 0), (1, 1))
BLOCK = 16
BINS = 24
LAMBDAS = np.linspace(-0.2, 0.5, 141)
KURTOSIS = scipy.stats.tukeylambda.stats(LAMBDAS, moments="k") + 3
BELL = 0.14
FLATTEST = 0.15
FAR, SCENE = 3, 0.05


def blocks(plane, side=BLOCK):
    h, w = plane.shape[0] // side * side, plane.shape[1] // side * side
    return plane[:h, :w].reshape(h // side, side, w // side, side).swapaxes(1, 2).reshape(-1, side, side)


def line(truth, plane):
    """The plane's gain and offset against the clean one, as raw_pairs.match fits them."""
    means = [blocks(p, 64).mean((1, 2)) for p in (truth, plane)]
    unclipped = means[0] < 0.5
    return np.polyfit(means[0][unclipped], means[1][unclipped], 1)


def tail(kurtosis, read_share, shot_excess):
    """The Tukey lambda for read noise that brings all the noise in the shadows to the kurtosis measured there.

    read_share is how much of that noise's variance is read noise, shot_excess the kurtosis shot noise
    adds by itself. Under a quarter read noise the tails say nothing of it and it stays a bell curve;
    under a half it gets the tails a half would need, so a small error is not blown up.
    """
    if read_share < 0.25:
        return BELL
    wanted = 3 + (kurtosis - 3 - shot_excess) / max(read_share, 0.5) ** 2
    return float(np.interp(wanted, KURTOSIS[::-1], LAMBDAS[::-1]))


def step(plane):
    """The raw's rounding step: the commonest gap between the values its darkest pixels take."""
    values = np.unique(plane.ravel()[:: max(plane.size // 400000, 1)])
    gaps = np.diff(values[: len(values) // 4 + 2])
    return float(np.median(gaps))


def exposed(truth, plane):
    """The blocks of a colour plane that are neither black nor clipped."""
    mean = blocks(truth).mean((1, 2))
    return (mean > 0.0005) & (mean < 0.6) & (blocks(plane).max((1, 2)) < 0.9)


def flat(truth, plane):
    """A colour plane where the clean one is flat, in 16 px blocks: their brightness, and what is left of them.

    Left is the plane with the clean one taken off: both frames' noise and none of the scene. Also the
    plane's gain and offset.
    """
    slope, offset = line(truth, plane)
    smooth = blocks(cv2.GaussianBlur(truth, (0, 0), 1.5))
    mean = blocks(truth).mean((1, 2))
    busy = smooth.std((1, 2)) / (mean + 0.002)
    usable = exposed(truth, plane)
    if usable.sum() < 16 * BINS:
        raise ValueError("too few blocks that are neither black nor clipped")
    keep = usable & (busy < np.percentile(busy[usable], 25))
    return slope * mean[keep], blocks(plane - slope * truth)[keep], slope, offset


def curve(signal, left, share):
    """Noise variance against brightness: the medians of both over 24 runs of blocks, darkest first."""
    inside = left.var(2, ddof=1).mean(1) * share
    order = np.argsort(signal)
    medians = lambda values: np.array([np.median(part) for part in np.array_split(values[order], BINS)])
    return medians(signal), medians(inside)


def unit(signal, left):
    """The darkest quarter of the blocks, each line centred and the block brought to unit strength."""
    dark = left[signal < np.percentile(signal, 25)]
    return (dark - dark.mean(2, keepdims=True)) / np.sqrt(dark.var(2, ddof=1).mean(1))[:, None, None]


def plane_levels(truth, plane, share):
    """One colour plane's levels; share is how much of the noise left over is the noisy frame's."""
    signal, left, slope, offset = flat(truth, plane)
    s, v = curve(signal, left, share)
    (shot, read), _ = scipy.optimize.nnls(np.stack([s / v, 1 / v], 1), np.ones(BINS))
    darkest = signal < np.percentile(signal, 25)
    dark, level = left[darkest], float(np.median(signal[darkest]))
    spread = shot * level + read
    band = lambda axis: float(np.mean(dark.mean(axis).var(1, ddof=1) - dark.var(axis, ddof=1).mean(1) / BLOCK) * share)
    kurtosis = float(scipy.stats.kurtosis(unit(signal, left).clip(-12, 12), axis=None, fisher=False))
    return {
        "gain": float(slope),
        "offset": float(offset),
        "shot": float(shot),
        "read": float(np.sqrt(read)),
        "row": band(2),
        "column": band(1),
        "tail": tail(kurtosis, read / spread, shot**3 * level / spread**2),
        "kurtosis": kurtosis,
        "step": step(plane),
        "blocks": len(signal),
        "fit": float(np.median(np.abs(np.log(v / (shot * s + read))))),
    }


def together(clean, noisy, gains):
    """How far the noise of neighbouring pixels moves together, as correlations: red's and blue's with the
    green beside it, and each colour plane's with its next pixel of the same colour across and down.

    Over the flattest blocks of the clean frame that no colour is black or clipped in, each block of
    the noisy frame less the clean one centred on its own mean, so neither the scene nor a black
    offset counts as noise. Nothing for a frame with too few such blocks, and nothing for one whose
    blocks still hold the scene, a frame that sits a part of a pixel off its clean one or was lit
    another way: a camera's smoothing reaches two pixels of a colour at most, and the scene shows as
    pixels FAR apart still moving together by more than SCENE.
    """
    twins, planes = ([frame[i::2, j::2] for i, j in SITES] for frame in (clean, noisy))
    usable = np.all([exposed(twin, plane) for twin, plane in zip(twins, planes)], 0)
    if usable.sum() < 16 * BINS:
        return {}
    left = [blocks(plane - gain * twin) for plane, twin, gain in zip(planes, twins, gains)]
    busy = sum(blocks(twins[plane]).std((1, 2)) for plane in (0, 1, 3))
    flattest = usable & (busy <= np.quantile(busy[usable], FLATTEST))
    red, green, other, blue = noise = [part[flattest] - part[flattest].mean((1, 2), keepdims=True) for part in left]
    alike = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    across = lambda apart: [alike(plane[:, :, :-apart], plane[:, :, apart:]) for plane in noise]
    down = lambda apart: [alike(plane[:, :-apart], plane[:, apart:]) for plane in noise]
    if max(across(FAR) + down(FAR)) > SCENE:
        return {}
    return {"beside": [alike(red, green), alike(blue, other)], "across": across(1), "down": down(1)}


def measure(clean, noisy, share=1.0):
    """A noisy frame's row, or None when it sits too far from the clean one to compare."""
    dx, dy = raw_pairs.shift_between(clean, noisy)
    edge = raw_pairs.MAX_SHIFT
    if max(abs(dx), abs(dy)) > edge:
        return None
    h, w = clean.shape
    clean, noisy = clean[edge : h - edge, edge : w - edge], noisy[edge + dy : h - edge + dy, edge + dx : w - edge + dx]
    planes = [plane_levels(clean[i::2, j::2], noisy[i::2, j::2], share) for i, j in SITES]
    row = {key: [plane[key] for plane in planes] for key in ("shot", "read", "offset", "tail", "kurtosis", "gain")}
    row |= together(clean, noisy, row["gain"])
    banding = lambda key: float(np.sqrt(max(np.mean([plane[key] for plane in planes]), 0)))
    return row | {
        "row": banding("row"),
        "column": banding("column"),
        "step": float(np.median([plane["step"] for plane in planes])),
        "mean": float(noisy.mean()),
        "blocks": min(plane["blocks"] for plane in planes),
        "fit": float(np.mean([plane["fit"] for plane in planes])),
    }


def ladders():
    for folder in sorted(raw_pairs.LADDERS.glob("*/*/*/")):
        site, camera, scene = folder.relative_to(raw_pairs.LADDERS).parts
        steps = sorted((int(file.stem), file) for file in folder.iterdir() if file.stem.isdigit())
        held = (site, scene) == ("ir", "sli") or any(name in camera for name in raw_pairs.HELD_OUT)
        if not held and len(steps) > 1 and steps[0][0] <= 400:
            yield f"{site}--{camera}--{scene}", steps


def fit(name, steps):
    out = FITS / f"{name}.json"
    if out.exists():
        return
    site, camera, scene = name.split("--")
    (base, path), *rest = steps
    clean = raw_pairs.mosaic(path)
    rows = []
    for iso, path in rest if clean is not None else []:
        noisy = raw_pairs.mosaic(path)
        if noisy is None or noisy.shape != clean.shape:
            continue
        try:
            row = measure(clean, noisy, iso / (iso + base))
        except ValueError as error:
            print(name, iso, "failed", error, flush=True)
            continue
        if row:
            rows.append({"camera": camera, "site": site, "scene": scene, "iso": iso, "base": base} | row)
    out.write_text(json.dumps(rows))
    print(name, [row["iso"] for row in rows], flush=True)


def sound(row):
    """Whether a frame's fit can be trusted: it matched the clean frame's light, and every plane has a level."""
    gains = np.array(row["gain"])
    return (
        row["blocks"] >= 2000
        and row["fit"] < 0.25
        and min(row["shot"]) > 0
        and np.ptp(gains) / gains.mean() < 0.2
        and 0.25 < gains.mean() < 4
    )


def table():
    rows = [row for file in sorted(FITS.glob("*.json")) for row in json.loads(file.read_text())]
    kept = [row for row in rows if sound(row)]
    (OUT / "cameras.json").write_text(json.dumps(kept))
    print(len(kept), "of", len(rows), "rows from", len({row["camera"] for row in kept}), "cameras")


def main(workers, worker):
    FITS.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    for name, steps in list(ladders())[worker::workers]:
        try:
            fit(name, steps)
        except (OSError, ValueError, rawpy.LibRawError, np.linalg.LinAlgError) as error:
            print(name, "failed", repr(error), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        table()
    else:
        main(int(sys.argv[1]), int(sys.argv[2]))
