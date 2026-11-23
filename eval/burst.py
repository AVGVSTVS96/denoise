"""The burst test: what a burst model (train_raw.py --burst) makes of several real frames of one scene.

    uv run burst.py ladders r19-burst=data/models/raw/r19-burst/joint.pth r17-all=data/models/raw/r17-all/joint.pth
    uv run burst.py ladders gr3x gr3 r19-burst=...          on those tests alone, of gr3, gr3x, unseen and seen
    uv run burst.py moving data/models/raw/r19-burst/joint.pth
    uv run burst.py own data/models/raw/r19-burst/joint.pth FOLDER CLEAN

ladders reads small print (small_print.py) in a real burst we hold: a review site's ISO ladder is one
scene shot again at each ISO, a fraction of a pixel apart, so the frame at PICKED with the next one
or two up the ladder is a burst of uneven frames, each noisier than the last. Every model gives its
picture of the picked frame alone, "one frame": a row beside each burst, over the cameras that have
that burst's frames. A burst model then gets the frames as they are, each moved by whole colour
cells until its lines sit nearest the picked frame's (lined_up()) and no finer, "burst". A
one-frame model gets them averaged, each weighed by one over its ISO, with nothing moved, and the
lines read at their weighted place, "merged": the merge a burst model has to beat. Both readers
spell every line at five cuts; right, wrong and missing are the first reader's, as
small_print.told counts them, and "second" is the second reader's letters right. They are summed
over all of a camera's lines and over its lines up to TALLEST px tall in its clean frame, the print
a person stops reading first, and over a group's cameras. Rows go to data/out/burst.json as
{test: {model: {way: {"all": [right, wrong, missing, second, there], "small": [...]}}}}, each
camera under its file's name. Without a test named it runs gr3x, gr3 and unseen.

moving is for what a burst gets wrong: something that moved shows twice. It makes bursts of the
Ricohs' full-colour windows as the trainer does (train_raw.burst), every crop with a patch that
sits several px off in the later frames, under made-up noise at the levels of the camera's frame at
PICKED, and gives how far the picture is from the truth inside those patches, in 8-bit levels of
the developed picture, from the burst and from the picked frame alone. The burst's must not be the
larger; outside the patches it should be the smaller.

own is for a burst of the user's: FOLDER holds its raw files and nothing else, the first by name
the picked frame, and CLEAN is a clean frame of the same scene. Each later frame is moved onto the
picked one by whole colour cells, a block at a time (onto()), and FOLDER/burst gets one.png, the
picked frame alone, merged.png, the lined-up frames averaged and cleaned as one, burst.png, and
clean.png, each developed as the test develops pictures.

It runs on the GPU when there is one and on the CPU when not, there in float32 where the GPU runs
bfloat16, so compare rows made on the same machine.
"""

import json
import pathlib
import sys

import cv2
import numpy as np
import torch

import bench_raw
import demosaic
import raw_pairs
import small_print
import train_raw

OUT = raw_pairs.DATA / "out" / "burst.json"
PICKED, MORE, TALLEST = 12800, (1, 2), 15
BLOCK, REACH = 128, 64


def lined_up(frames, lines):
    """A window's frames (count, H, W), each after the first moved by whole colour cells until its
    lines (count, lines, 4, 2) sit nearest the first's, and how far a line's corner still sits off, px."""
    moved, apart = [frames[0]], 0.0
    for frame, at in zip(frames[1:], lines[1:]):
        cells = np.round((at - lines[0]).reshape(-1, 2).mean(0) / 2).astype(int) * 2
        moved.append(torch.roll(frame, (-int(cells[1]), -int(cells[0])), (0, 1)))
        apart = max(apart, float(np.abs(at - lines[0] - cells).max()))
    return torch.stack(moved), apart


def read(shown, thing, corners, small):
    """A printed thing's letters in an 8-bit picture: right, wrong and missing by the first reader and
    right by the second, each line's the mean over its five cuts, then the letters there; summed over
    all its lines, and over those of small."""
    lines = small_print.LINES[thing]
    spelt = lambda text, strip: (
        *small_print.told(text, small_print.spelt(strip)),
        small_print.told(text, small_print.spelt_again(strip))[0],
        len(small_print.plain(text)),
    )
    counts = np.array(
        [
            np.mean([spelt(text, strip) for strip in small_print.strips(shown, line)], 0)
            for (text, _), line in zip(lines, corners)
        ]
    )
    return np.stack([counts.sum(0), counts[small].sum(0)])


@torch.no_grad()
def camera(file, models):
    """{model: {way: counts (2, 5)}} for one camera's windows, read()'s over its printed things, and how
    far its frames sit apart once lined up; nothing for a camera without a frame at PICKED and the next ISO."""
    things, isos, means, picture = small_print.opened(file)
    if PICKED not in isos or isos.index(PICKED) + 1 >= len(isos):
        return {}, 0.0
    k, rows, furthest = isos.index(PICKED), {}, 0.0
    look = lambda result: small_print.as_bytes(picture(result, None))
    for thing, (frames, _, lines) in things.items():
        small = [n for n, line in enumerate(lines[0]) if np.linalg.norm(line[3] - line[0]) <= TALLEST]
        for name, (clean_up, slots) in models.items():
            alone, found = read(look(clean_up(frames[k + 1 : k + 2], means[k])), thing, lines[k + 1], small), {}
            for more in [more for more in MORE if k + more < len(isos)]:
                used = "+".join(str(iso) for iso in isos[k : k + 1 + more])
                given, places = frames[k + 1 : k + 2 + more], lines[k + 1 : k + 2 + more]
                burst, apart = lined_up(given, places)
                furthest = max(furthest, apart)
                found[f"{used} one frame"] = alone
                if slots > more:
                    found[f"{used} burst"] = read(look(clean_up(burst[None], means[k])), thing, places[0], small)
                if slots == 1:
                    weights = np.float32([PICKED / iso for iso in isos[k : k + 1 + more]])
                    weights /= weights.sum()
                    merged = sum(float(weight) * frame for weight, frame in zip(weights, given))[None]
                    where = sum(float(weight) * place for weight, place in zip(weights, places))
                    found[f"{used} merged"] = read(look(clean_up(merged, means[k])), thing, where, small)
            kept = rows.setdefault(name, {})
            for way, counts in found.items():
                kept[way] = kept.get(way, 0) + counts
    return rows, furthest


def shown(title, rows):
    """A test's rows printed, all lines and the lines up to TALLEST px, as small_print's small lines are."""
    for at, which in enumerate(("all lines", f"lines up to {TALLEST} px tall")):
        heads = ("letters", "read %", "right", "wrong", "missing", "wrong of spelt %", "second read %")
        print(f"\n{title}, {which}\n{'':52}" + "".join(f"{head:>{len(head) + 3}}" for head in heads))
        for name, ways in rows.items():
            for way, counts in sorted(ways.items()):
                right, wrong, missing, second, there = counts[at]
                spelt = 100 * wrong / max(right + wrong, 1e-9)
                cells = (there, 100 * right / there, right, wrong, missing, spelt, 100 * second / there)
                row = "".join(f"{cell:>{len(head) + 3}.1f}" for head, cell in zip(heads, cells))
                print(f"{name:20}{way:32}{row}", flush=True)


def ladders(*specs):
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    every = (*bench_raw.CAMERAS, *bench_raw.groups())
    models = {}
    for name, _, weights in (spec.partition("=") for spec in specs if spec not in every):
        model = small_print.trained(weights)
        models[name] = (train_raw.cleaner(model), train_raw.slots(model))
    table = json.loads(OUT.read_text()) if OUT.exists() else {}
    for test in [test for test in every if test in specs] or ("gr3x", "gr3", "unseen"):
        together, furthest = {}, 0.0
        for name in bench_raw.cuts(f"print/{test}"):
            rows, apart = camera(bench_raw.BENCH / f"{name}.npz", models)
            furthest = max(furthest, apart)
            for model, ways in rows.items():
                for way, counts in ways.items():
                    table.setdefault(name.split("/")[-1], {}).setdefault(model, {})[way] = dict(
                        zip(("all", "small"), counts.tolist())
                    )
                    kept = together.setdefault(model, {})
                    kept[way] = kept.get(way, 0) + counts
            if rows and name.split("/")[-1] != test:
                print(f"{name}: frames lined up to {apart:.2f} px", flush=True)
        if together:
            table.setdefault(test, {}).update(
                {
                    model: {way: dict(zip(("all", "small"), counts.tolist())) for way, counts in ways.items()}
                    for model, ways in together.items()
                }
            )
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(table, indent=1))
            shown(f"{test}, ISO {PICKED} picked, frames lined up to {furthest:.2f} px", together)


def moved(name, clean_up, slots, rng):
    """One Ricoh's full-colour windows as bursts with a moved patch in each: how far the picture is from
    the truth, in 8-bit levels of the developed picture, inside the patches and outside them, from the
    burst and from the picked frame alone."""
    data, levels = (
        np.load(bench_raw.BENCH / "colour" / f"{name}.npz"),
        np.load(small_print.READER.parent / f"{name}.npz"),
    )
    on = lambda key: torch.from_numpy(data[key].astype(np.float32)).to(small_print.DEVICE)
    truth, wb, matrix = on("truth"), on("wb"), on("matrix")
    shot, read_noise = levels["levels"][levels["isos"].tolist().index(PICKED)]
    seen, _, _, patch = train_raw.burst(truth, max(slots, 2), fewer=0, uneven=0, moving=1)
    frames = small_print.made_up(bench_raw.mosaiced(seen), shot, read_noise, rng)
    frames = frames.view(len(truth), -1, *truth.shape[-2:])
    mean = float(frames[:, 0].mean())
    exposure = 0.108 / bench_raw.develop(truth, wb, matrix).median()
    picture = lambda result: (bench_raw.develop(result, wb, matrix) * exposure).clamp(0, 1) ** (1 / 2.2)
    apart = lambda result: 255 * (picture(result) - picture(truth)).abs().mean(1, keepdim=True)
    burst, one = apart(clean_up(frames, mean)), apart(clean_up(frames[:, 0], mean))
    return [float(part.mean()) for part in (burst[patch], one[patch], burst[~patch], one[~patch])]


@torch.no_grad()
def moving(weights):
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    model = small_print.trained(weights)
    clean_up, slots = train_raw.cleaner(model), train_raw.slots(model)
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for name in bench_raw.CAMERAS:
        inside, alone, outside, alone_outside = moved(name, clean_up, slots, rng)
        print(
            f"{name}, ISO {PICKED}'s noise, {slots} frames, a moved patch in every window, 8-bit levels from the truth:"
            f" inside the patches burst {inside:.3f}, one frame {alone:.3f};"
            f" outside them burst {outside:.3f}, one frame {alone_outside:.3f}",
            flush=True,
        )


def onto(picked, other):
    """A whole frame (H, W) moved onto the picked one by whole colour cells, a block of BLOCK px at a time.

    raw_pairs.offsets says where each block sits in the other frame, to a fraction of a pixel, from a
    fit over the whole frame that covers a shift and a turn, up to REACH px: a hand turns the camera
    a little between frames, and a turn is another offset in every corner. A block is copied from
    there, rounded to whole colour cells, so nothing is resampled and a colour stays the colour its
    pixel measured. The other frame comes at the picked one's brightness.
    """
    at, out = raw_pairs.offsets(picked, other), np.empty_like(picked)
    wide = np.pad(other, REACH, "reflect")
    starts = lambda side: sorted({*range(0, side - BLOCK, BLOCK), side - BLOCK})
    about = lambda start, side: int(np.clip(start + (BLOCK - raw_pairs.CROP) // 2, 0, side - raw_pairs.CROP)) // 2 * 2
    for y in starts(picked.shape[0]):
        for x in starts(picked.shape[1]):
            off, _ = at(about(y, picked.shape[0]), about(x, picked.shape[1]))
            dx, dy = (int(np.clip(round(part / 2) * 2, -REACH, REACH)) for part in off)
            out[y : y + BLOCK, x : x + BLOCK] = wide[y + REACH + dy :, x + REACH + dx :][:BLOCK, :BLOCK]
    return out


@torch.no_grad()
def own(weights, folder, clean):
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    folder = pathlib.Path(folder)
    model = small_print.trained(weights)
    paths = sorted(path for path in folder.iterdir() if path.is_file())[: max(train_raw.slots(model), 2)]
    picked, *later = (raw_pairs.mosaic(path) for path in paths)
    burst = np.stack([picked, *(onto(picked, frame) for frame in later)])
    wb, matrix = (torch.from_numpy(part) for part in bench_raw.colour(paths[0]))
    results = {
        "one": train_raw.denoise(model, picked),
        "merged": train_raw.denoise(model, burst.mean(0)),
        "burst": train_raw.denoise(model, burst),
    }
    develop = lambda result, way=None: bench_raw.develop(torch.from_numpy(result)[None], wb, matrix, way)
    pictures = {name: develop(result) for name, result in results.items()}
    exposures = dict.fromkeys(pictures, 0.108 / pictures["one"].median())
    pictures["clean"] = develop(onto(picked, raw_pairs.match(picked, raw_pairs.mosaic(clean))[0]), demosaic.menon)
    exposures["clean"] = 0.108 / pictures["clean"].median()
    (folder / "burst").mkdir(exist_ok=True)
    for name, linear in pictures.items():
        picture = small_print.as_bytes((linear * exposures[name]).clamp(0, 1) ** (1 / 2.2))
        cv2.imwrite(str(folder / "burst" / f"{name}.png"), picture[:, :, ::-1])
        print(folder / "burst" / f"{name}.png", flush=True)


if __name__ == "__main__":
    {"ladders": ladders, "moving": moving, "own": own}[sys.argv[1]](*sys.argv[2:])
