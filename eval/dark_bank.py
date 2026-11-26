"""A camera's own noise from its lens-cap frames, for train_raw.py --darks.

    uv run dark_bank.py FOLDER gr3x ricoh-gr-iii

FOLDER holds one body's lens-cap raws as the camera named them, some twenty at each ISO, all at the
shutter speed of a night photo. A frame's ISO is read from its EXIF, or from a name like dark-<ISO>-<n>.

A dark frame is read noise as the camera makes it: banding, pixels that sit high, long tails, the
black offset the file's black level misses, and whatever the camera's own smoothing did to all of
it. Part of that repeats from frame to frame, the fixed pattern, and a photo can have it taken off
before a network sees it. So for each ISO this writes three things.

The fixed pattern, data/darks/<name>-fixed.npy, float16 (ISOs, H, W) in the frame's normalised units,
as raw_pairs.mosaic cuts a frame: the mean dark frame, as far as it repeats. Its level, each line's
and each column's own and its slow shading are means over thousands of pixels and kept whole. What
is left pixel by pixel is kept by the share of it that shows again when the frames are split in two
halves, since the rest of it is the frames' own noise, a twentieth of it left in a mean of twenty;
a pixel that sits surely high is kept whole.

The noise that does not repeat, data/noise/<name>.npy, float16 (frames, 2048, 2048): the middle of
BANK frames an ISO, each with the fixed pattern of all the other frames taken off, which is what a
photo with the pattern taken off holds, and each colour plane brought to unit strength.

And data/noise/<name>.json. "frames" has a row for each of those: its ISO, each colour plane's
strength ("read") and the offset left, in units of that strength. With the third word, a camera of
fit_noise.py's table with the same sensor, a row also carries that camera's shot noise at the ISO
and how far its neighbouring pixels' noise moves together where the frame is lit, so train_raw.py
can lay shot noise as the camera hands it over under the real dark frame. A dark frame cannot say
either. "iso" has what was measured at each ISO, all over the read noise: how strong the fixed
pattern is, part by part, what a frame's noise is as shot and with the pattern off, the banding
that changes from frame to frame, and how far neighbours move together in the dark.
"""

import collections
import fractions
import json
import pathlib
import re
import subprocess
import sys

import cv2
import numpy as np

import blacks
import fit_noise
import raw_pairs

OUT = raw_pairs.DATA / "noise"
MAPS = raw_pairs.DATA / "darks"
TABLE = OUT / "cameras.json"
SIDE = 2048
BANK = 6
SHADING = 16
SURE = 6
SITES = ((0, 0), (0, 1), (1, 0), (1, 1))
RAWS = (".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".rw2", ".orf", ".pef", ".srw")


def near(plane):
    """A plane's mean and spread without the few pixels that sit far out."""
    middle = np.median(plane)
    rough = 1.4826 * np.median(np.abs(plane - middle))
    kept = plane[np.abs(plane - middle) < 6 * rough]
    return float(kept.mean()), float(kept.std())


def unit(cfa):
    """A dark mosaic with every colour plane centred and at unit strength, and each plane's offset and spread."""
    field, lifts, spreads = np.empty_like(cfa), [], []
    for i, j in SITES:
        mean, spread = near(cfa[i::2, j::2])
        field[i::2, j::2] = (cfa[i::2, j::2] - mean) / spread
        lifts.append(mean / spread)
        spreads.append(spread)
    return field, lifts, spreads


def shot_at(path):
    """A raw's body, ISO and shutter speed."""
    body, iso = blacks.body(path)
    named = re.match(r"dark-(\d+)-", path.name)
    listed = subprocess.run(["exiv2", "-Pv", "-gExposureTime", str(path)], check=False, capture_output=True, text=True)
    time = listed.stdout.split()
    return body, iso or int(named.group(1)) if iso or named else 0, str(fractions.Fraction(time[0])) if time else ""


def borrowed(camera, iso):
    """Shot noise per colour plane and how far neighbours' noise moves together at an ISO, from the rows of a
    camera in fit_noise.py's table: between the ISOs it was measured at, and past them shot noise goes with the ISO."""
    rows = [row for row in json.loads(TABLE.read_text()) if row["camera"] == camera and "beside" in row]
    if not rows:
        raise SystemExit(f"{camera} has no rows with neighbours in {TABLE}")
    isos = sorted({row["iso"] for row in rows})
    mean = lambda key: np.array([np.mean([row[key] for row in rows if row["iso"] == step], 0) for step in isos])
    steps = np.log(isos)
    at = float(np.clip(np.log(iso), steps[0], steps[-1]))
    between = lambda values: np.array([np.interp(at, steps, plane) for plane in values.T])
    shot = np.exp(between(np.log(mean("shot")))) * iso / np.exp(at)
    return {"shot": shot.tolist()} | {key: between(mean(key)).tolist() for key in ("beside", "across", "down")}


def summed(paths):
    """Dark frames' sum, the sum of every other one, and the read noise per colour plane, from each frame
    less the one before it: nothing that repeats is in that."""
    total = half = last = None
    reads = []
    for k, path in enumerate(paths):
        frame = raw_pairs.mosaic(path)
        total = frame.copy() if total is None else total + frame
        if k % 2 == 0:
            half = frame.copy() if half is None else half + frame
        if last is not None:
            reads.append([near((frame - last)[i::2, j::2])[1] / np.sqrt(2) for i, j in SITES])
        last = frame
    return total, half, np.mean(reads, 0)


def parts(plane):
    """A mean dark plane as its level, each line's own, each column's own, the slow shading left, and the rest."""
    level = plane.mean()
    rows = (plane - level).mean(1, keepdims=True)
    columns = (plane - level - rows).mean(0, keepdims=True)
    rest = plane - level - rows - columns
    shading = cv2.GaussianBlur(rest, (0, 0), SHADING)
    return level, rows, columns, shading, rest - shading


def sure(pixels, read, frames):
    """The pixels of a mean of that many frames that sit too far out to be the frames' own noise."""
    return np.abs(pixels) > SURE * read / np.sqrt(frames)


def pattern(plane, frames, read, repeated):
    """What repeats in a mean of that many dark frames: parts() with the pixels kept by the share of their
    variance that repeats, and whole where one sits surely out."""
    level, rows, columns, shading, pixels = parts(plane)
    kept = np.where(sure(pixels, read, frames), 1, repeated / (repeated + read**2 / frames))
    return level + rows + columns + shading + kept * pixels


def repeats(one, other):
    """How much of two halves' same part is in both, as a variance: the frames' own noise is in neither twice."""
    return max(float(np.mean((one - one.mean()) * (other - other.mean()))), 0.0)


def strengths(halves, whole, read, frames):
    """The fixed pattern of one colour plane, part by part, as variances: what the two halves' means share."""
    one, other, both = (parts(mean) for mean in (*halves, whole))
    out = sure(both[4], read, frames)
    found = {name: repeats(one[k], other[k]) for k, name in ((1, "row"), (2, "column"), (3, "shading"))}
    return found | {"pixel": repeats(one[4][~out], other[4][~out]), "sure": float(out.mean()), "black": float(both[0])}


def banding(plane, read, axis):
    """How far a dark plane's lines (axis 1) or columns (axis 0) sit apart beyond what its pixels' noise makes of them."""
    return max(float(plane.mean(axis).var()) - read**2 / plane.shape[axis], 0.0)


def ties(field):
    """How far neighbouring pixels of a unit dark field move together, as fit_noise.py's rows have it."""
    planes = [fit_noise.blocks(field[i::2, j::2].clip(-SURE, SURE)) for i, j in SITES]
    return fit_noise.ties([plane - plane.mean((1, 2), keepdims=True) for plane in planes])


def ladder(paths):
    """One ISO's dark frames: its fixed pattern, the bank's fields with their rows, and what was measured."""
    count = len(paths)
    total, half, read = summed(paths)
    halves = half / ((count + 1) // 2), (total - half) / (count // 2)
    whole = total / count
    plane = lambda cfa, site: cfa[site[0] :: 2, site[1] :: 2]
    fixed = [
        strengths([plane(mean, site) for mean in halves], plane(whole, site), level, count)
        for site, level in zip(SITES, read)
    ]
    repeated = [found["pixel"] for found in fixed]
    over = lambda variances: float(np.sqrt(np.mean(np.array(variances) / read**2)))
    y, x = ((side - SIDE) // 4 * 2 for side in whole.shape)
    fields, rows, seen = [], [], collections.defaultdict(list)
    for path in paths[:: max(count // BANK, 1)][:BANK]:
        frame = raw_pairs.mosaic(path)
        rest = (total - frame) / (count - 1)
        left = np.empty_like(frame)
        for site, level, share in zip(SITES, read, repeated):
            left[site[0] :: 2, site[1] :: 2] = plane(frame, site) - pattern(plane(rest, site), count - 1, level, share)
        for name, cfa in (("as shot", frame), ("pattern off", left)):
            seen[name].append([near(plane(cfa, site))[1] ** 2 for site in SITES])
        for name, axis in (("row", 1), ("column", 0)):
            seen[name].append([banding(plane(left, site), level, axis) for site, level in zip(SITES, read)])
        field, lifts, spreads = unit(left[y : y + SIDE, x : x + SIDE])
        seen["ties"].append(ties(field))
        fields.append(field.clip(-60, 60).astype(np.float16))
        rows.append({"offset": lifts, "read": spreads})
    cut = np.empty_like(whole)
    for site, level, share in zip(SITES, read, repeated):
        cut[site[0] :: 2, site[1] :: 2] = pattern(plane(whole, site), count, level, share)
    together = {key: np.mean([tie[key] for tie in seen["ties"]], 0).tolist() for key in ("beside", "across", "down")}
    return (
        cut.astype(np.float16),
        fields,
        rows,
        {
            "frames": count,
            "read": read.tolist(),
            "black": [found["black"] for found in fixed],
            "fixed": {part: over([found[part] for found in fixed]) for part in ("row", "column", "shading", "pixel")},
            "sure": float(np.mean([found["sure"] for found in fixed])),
            "kept": [float(share / (share + level**2 / count)) for share, level in zip(repeated, read)],
            "noise": {name: over(np.mean(seen[name], 0)) for name in ("as shot", "pattern off")},
            "banding": {name: over(np.mean(seen[name], 0)) for name in ("row", "column")},
            "ties": together,
        },
    )


def main(folder, name, like=None):
    shots = collections.defaultdict(list)
    for path in sorted(pathlib.Path(folder).iterdir()):
        if path.suffix.lower() in RAWS:
            body, iso, shutter = shot_at(path)
            shots[iso].append((path, body, shutter))
    maps, fields, frames, isos = [], [], [], {}
    for iso, shot in sorted(shots.items()):
        paths, bodies, shutters = zip(*shot)
        if not iso or len(paths) < 2:
            print(f"ISO {iso}: {len(paths)} frames left out, an ISO takes two and more", flush=True)
            continue
        cut, bank, rows, found = ladder(paths)
        lent = borrowed(like, iso) if like else {}
        maps.append(cut)
        fields += bank
        frames += [{"camera": bodies[0], "iso": iso} | row | lent for row in rows]
        isos[iso] = found | {"shutter": sorted(set(shutters))}
        fixed, noise, wavy, tied = found["fixed"], found["noise"], found["banding"], found["ties"]
        print(
            f"ISO {iso}: {found['frames']} frames at {', '.join(isos[iso]['shutter'])} s, read {np.mean(found['read']):.3g},"
            f" black {np.mean(found['black']) / np.mean(found['read']):+.2f} of it\n"
            f"  fixed pattern over read: rows {fixed['row']:.3f}, columns {fixed['column']:.3f},"
            f" shading {fixed['shading']:.3f}, pixels {fixed['pixel']:.3f} kept by {np.mean(found['kept']):.2f},"
            f" surely out {100 * found['sure']:.3f}%\n"
            f"  a frame's noise over read: {noise['as shot']:.3f} as shot, {noise['pattern off']:.3f} with the pattern off\n"
            f"  banding that changes, over read: rows {wavy['row']:.3f}, columns {wavy['column']:.3f}\n"
            f"  neighbours in the dark: beside {np.round(tied['beside'], 3).tolist()},"
            f" across {np.round(tied['across'], 3).tolist()}, down {np.round(tied['down'], 3).tolist()}"
            + (f"\n  {like} where lit: beside {np.round(lent['beside'], 3).tolist()}" if like else ""),
            flush=True,
        )
    if not fields:
        raise SystemExit(f"no ISO with two dark frames in {folder}")
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)
    np.save(MAPS / f"{name}-fixed.npy", np.stack(maps))
    np.save(OUT / f"{name}.npy", np.stack(fields))
    (OUT / f"{name}.json").write_text(
        json.dumps({"camera": frames[0]["camera"], "like": like, "frames": frames, "iso": isos}, indent=1)
    )
    print(
        f"{len(fields)} fields of {len(isos)} ISOs: train_raw.py --darks data/noise/{name}.json" + " 0.5" * bool(like)
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
