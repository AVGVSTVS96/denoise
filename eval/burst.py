"""The burst test: what a burst model (train_raw.py --burst) makes of several real frames of one scene.

    uv run burst.py ladders r19-burst=data/models/raw/r19-burst/joint.pth r17-all=data/models/raw/r17-all/joint.pth
    uv run burst.py ladders gr3x gr3 r19-burst=...          on those tests alone, of gr3, gr3x, unseen and seen
    uv run burst.py equal r19-burst=data/models/raw/r19-burst/joint.pth r17-all=data/models/raw/r17-all/joint.pth
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

equal reads the same small print in a burst of equal frames, which no ladder holds: the GR IIIx's
clean windows under made-up noise of the strength of its frame at PICKED, FRAMES draws that share
nothing but the scene. Once steady, every frame the clean window as it is, and once with each later
frame up to train_raw.SHAKE px off and turned, moved as the trainer moves them (train_raw.shaken)
and sampled back to a mosaic: a mosaic cannot be moved without resampling, so what is moved is the
clean window through demosaic.menon, and a later frame's samples are that demosaic's. Beside the
two bursts, one made-up frame at PICKED and at each ISO under it: four equal frames are two stops,
so the frame two ISOs down is what a perfect merge would read. The noise is small_print.made_up's,
from one generator, the single frames first in small_print.py's order: a model's one-frame rows are
its "-made" rows there, and the burst's picked frame is the single frame at PICKED. Rows go to
data/out/burst.json under "equal".

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

Every burst a burst model gets, it gets a second time behind a guard (guarded()), as "guarded": a
burst's one fault is to show twice what moved between its frames, and the guard takes such a place
out of the later frames before the model sees them. Where a later frame's mean over NEAR colour
cells a side differs from the picked frame's, in any colour plane, by more than ALLOWED times what
the two frames' shot and read noise gives that mean, the cells it was taken over become the picked
frame's, an empty slot to the model there. It is a mean and not an energy on purpose: print a
pixel apart differs from itself stroke by stroke, up one side and down the other, which a mean
over many strokes hardly sees, and a thing several pixels off moves the mean. The noise is the
frames' own, as bench_raw.noise_levels measured it or as the test laid it, so the guard is for
frames whose noise is known: own does not use it. A place the guard empties does not come out as
one frame alone would have it: NAFNet weighs its channels by their means over the whole tile, so
in a tile that has later frames elsewhere such a place is cleaned as lightly as the rest.

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
PICKED, MORE, TALLEST, FRAMES = 12800, (1, 2), 15, 4
BLOCK, REACH = 128, 64
#: guarded()'s neighbourhood, colour cells a side, and how many times its noise a mean over it may differ.
NEAR, ALLOWED = 5, 6.0


def lined_up(frames, lines):
    """A window's frames (count, H, W), each after the first moved by whole colour cells until its
    lines (count, lines, 4, 2) sit nearest the first's, and how far a line's corner still sits off, px."""
    moved, apart = [frames[0]], 0.0
    for frame, at in zip(frames[1:], lines[1:]):
        cells = np.round((at - lines[0]).reshape(-1, 2).mean(0) / 2).astype(int) * 2
        moved.append(torch.roll(frame, (-int(cells[1]), -int(cells[0])), (0, 1)))
        apart = max(apart, float(np.abs(at - lines[0] - cells).max()))
    return torch.stack(moved), apart


def guarded(bursts, levels):
    """Bursts (N, frames, H, W) with a later frame's colour cells replaced by the picked frame's
    wherever the two differ by more than their noise allows, and the share of those cells replaced.

    levels (frames, 2) is each frame's shot and read noise, variance = shot * brightness + read. A
    frame's four colour planes are each averaged over NEAR cells a side; a later frame is off where
    any plane's average sits more than ALLOWED times the noise of that average from the picked
    frame's, and with it every cell the average was taken over.
    """
    planes, levels = train_raw.pack(bursts), torch.as_tensor(levels, dtype=bursts.dtype, device=bursts.device)
    picked, later = planes[:, :1], planes[:, 1:]
    box = lambda values: torch.nn.functional.avg_pool2d(values.flatten(0, 1), NEAR, 1, NEAR // 2).view(values.shape)
    both = lambda part: (levels[:1, part] + levels[1:, part]).view(1, -1, 1, 1, 1)
    noise = (both(0) * box(picked).clamp(min=0) + both(1)).sqrt() / NEAR
    off = (box(later - picked).abs() > ALLOWED * noise).any(2, keepdim=True).flatten(0, 1).float()
    off = (torch.nn.functional.max_pool2d(off, NEAR, 1, NEAR // 2) > 0).view(later[:, :, :1].shape)
    safe = torch.cat([picked, torch.where(off, picked, later)], 1)
    return train_raw.unpack(safe.flatten(0, 1)).view(bursts.shape), off.float().mean().item()


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
    """{model: {way: counts (2, 5)}} for one camera's windows, read()'s over its printed things, how far
    its frames sit apart once lined up, and the largest share of a burst's later cells the guard
    replaced; nothing for a camera without a frame at PICKED and the next ISO."""
    things, isos, means, picture = small_print.opened(file)
    if PICKED not in isos or isos.index(PICKED) + 1 >= len(isos):
        return {}, 0.0, 0.0
    k, rows, furthest, most, levels = isos.index(PICKED), {}, 0.0, 0.0, np.load(file)["levels"]
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
                    safe, share = guarded(burst[None], levels[k : k + 1 + more])
                    most = max(most, share)
                    found[f"{used} guarded"] = read(look(clean_up(safe, means[k])), thing, places[0], small)
                if slots == 1:
                    weights = np.float32([PICKED / iso for iso in isos[k : k + 1 + more]])
                    weights /= weights.sum()
                    merged = sum(float(weight) * frame for weight, frame in zip(weights, given))[None]
                    where = sum(float(weight) * place for weight, place in zip(weights, places))
                    found[f"{used} merged"] = read(look(clean_up(merged, means[k])), thing, where, small)
            kept = rows.setdefault(name, {})
            for way, counts in found.items():
                kept[way] = kept.get(way, 0) + counts
    return rows, furthest, most


def shown(title, rows):
    """A test's rows printed, all lines and the lines up to TALLEST px, as small_print's small lines are."""
    for at, which in enumerate(("all lines", f"lines up to {TALLEST} px tall")):
        heads = ("letters", "read %", "right", "wrong", "missing", "wrong of spelt %", "second read %")
        print(f"\n{title}, {which}\n{'':52}" + "".join(f"{head:>{len(head) + 3}}" for head in heads))
        for name, ways in rows.items():
            for way, counts in ways.items():
                right, wrong, missing, second, there = counts[at]
                spelt = 100 * wrong / max(right + wrong, 1e-9)
                cells = (there, 100 * right / there, right, wrong, missing, spelt, 100 * second / there)
                row = "".join(f"{cell:>{len(head) + 3}.{head != 'letters':d}f}" for head, cell in zip(heads, cells))
                print(f"{name:20}{way:32}{row}", flush=True)


def entrants(specs):
    """{name: (cleaner, slots)} for each NAME=WEIGHTS."""
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    models = {}
    for name, _, weights in (spec.partition("=") for spec in specs):
        model = small_print.trained(weights)
        models[name] = (train_raw.cleaner(model), train_raw.slots(model))
    return models


def kept(test, rows):
    """A test's rows added to OUT."""
    table = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model, ways in rows.items():
        found = {way: dict(zip(("all", "small"), counts.tolist())) for way, counts in ways.items()}
        table.setdefault(test, {}).setdefault(model, {}).update(found)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=1))


def ladders(*specs):
    every = (*bench_raw.CAMERAS, *bench_raw.groups())
    models = entrants([spec for spec in specs if spec not in every])
    for test in [test for test in every if test in specs] or ("gr3x", "gr3", "unseen"):
        together, furthest, most = {}, 0.0, 0.0
        for name in bench_raw.cuts(f"print/{test}"):
            rows, apart, share = camera(bench_raw.BENCH / f"{name}.npz", models)
            furthest, most = max(furthest, apart), max(most, share)
            kept(name.split("/")[-1], rows)
            for model, ways in rows.items():
                pooled = together.setdefault(model, {})
                for way, counts in ways.items():
                    pooled[way] = pooled.get(way, 0) + counts
            if rows and name.split("/")[-1] != test:
                print(f"{name}: frames lined up to {apart:.2f} px, the guard took up to {share:.1%}", flush=True)
        if together:
            kept(test, together)
            lined = f"frames lined up to {furthest:.2f} px, the guard took up to {most:.1%} of a window's later cells"
            shown(f"{test}, ISO {PICKED} picked, {lined}", together)


@torch.no_grad()
def equal(*specs):
    models = entrants(specs)
    file = small_print.READER.parent / "gr3x.npz"
    things, isos, means, picture = small_print.opened(file)
    data, rng, rows, taken = np.load(file), torch.Generator().manual_seed(0), {}, {}
    balance = torch.from_numpy(data["wb"]).to(small_print.DEVICE).view(1, 3, 1, 1)
    under = isos[: isos.index(PICKED) + 1]
    shot, read_noise = data["levels"][len(under) - 1]
    alone = {
        iso: {thing: small_print.made_up(frames[:1], *levels, rng) for thing, (frames, _, _) in things.items()}
        for iso, levels in zip(under, data["levels"].tolist())
    }
    look = lambda result: small_print.as_bytes(picture(result, None))
    for thing, (frames, _, lines) in things.items():
        small = [n for n, line in enumerate(lines[0]) if np.linalg.norm(line[3] - line[0]) <= TALLEST]
        clean, later = frames[:1], FRAMES - 1
        full = demosaic.menon(clean * bench_raw.mosaiced(balance.expand(1, 3, *clean.shape[1:]))) / balance
        move = ((torch.rand(later, 2, generator=rng) * 2 - 1) * train_raw.SHAKE).to(clean.device)
        turn = ((torch.rand(later, generator=rng) * 2 - 1) * train_raw.TURN).to(clean.device)
        scenes = {
            "steady": clean.expand(later, -1, -1),
            f"within {train_raw.SHAKE:g} px": bench_raw.mosaiced(
                train_raw.shaken(full.expand(later, -1, -1, -1), move, turn)
            ),
        }
        bursts = {
            way: torch.cat([alone[PICKED][thing], small_print.made_up(scene, shot, read_noise, rng)])[None]
            for way, scene in scenes.items()
        }
        safe = {way: guarded(burst, [[shot, read_noise]] * FRAMES) for way, burst in bursts.items()}
        for way, (_, share) in safe.items():
            taken[way] = taken.get(way, 0) + share / len(things)
        for name, (clean_up, slots) in models.items():
            pooled, given = rows.setdefault(name, {}), min(slots, FRAMES)
            found = {f"{iso} one frame": clean_up(alone[iso][thing], means[at]) for at, iso in enumerate(under)}
            for way in bursts if slots > 1 else ():
                found[f"{PICKED} {given} frames {way}"] = clean_up(bursts[way][:, :given], means[len(under) - 1])
                found[f"{PICKED} {given} frames {way}, guarded"] = clean_up(
                    safe[way][0][:, :given], means[len(under) - 1]
                )
            for way, result in found.items():
                pooled[way] = pooled.get(way, 0) + read(look(result), thing, lines[0], small)
    kept("equal", rows)
    shown(f"gr3x, clean windows under made-up noise, bursts at ISO {PICKED}'s strength", rows)
    for way, share in taken.items():
        print(f"{way}: the guard took {share:.1%} of the later frames' cells, a window's mean")


def moved(name, clean_up, slots, rng):
    """One Ricoh's full-colour windows as bursts with a moved patch in each: how far the picture is from
    the truth, in 8-bit levels of the developed picture, inside the patches and then outside them,
    from the burst, from the burst behind the guard and from the picked frame alone; and the share
    of the later frames' cells the guard replaced."""
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
    safe, share = guarded(frames, [[shot, read_noise]] * frames.shape[1])
    ways = [apart(clean_up(given, mean)) for given in (frames, safe, frames[:, 0])]
    return [[float(way[where].mean()) for way in ways] for where in (patch, ~patch)], share


@torch.no_grad()
def moving(weights):
    ((clean_up, slots),) = entrants([f"={weights}"]).values()
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for name in bench_raw.CAMERAS:
        found, share = moved(name, clean_up, slots, rng)
        told = [
            f"{where} burst {burst:.3f}, guarded {safe:.3f}, one frame {one:.3f}"
            for where, (burst, safe, one) in zip(("inside the patches", "outside them"), found)
        ]
        print(
            f"{name}, ISO {PICKED}'s noise, {slots} frames, a moved patch in every window, 8-bit levels from the truth:"
            f" {'; '.join(told)}; the guard took {share:.1%} of the later frames' cells",
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
    {"ladders": ladders, "equal": equal, "moving": moving, "own": own}[sys.argv[1]](*sys.argv[2:])
