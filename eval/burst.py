"""The burst test: what a burst model (train_raw.py --burst) makes of several real frames of one scene.

    uv run burst.py ladders r19-burst=data/models/raw/r19-burst/joint.pth r17-all=data/models/raw/r17-all/joint.pth
    uv run burst.py ladders gr3x gr3 r19-burst=...          on those tests alone, of gr3, gr3x, unseen and seen
    uv run burst.py equal r19-burst=data/models/raw/r19-burst/joint.pth r17-all=data/models/raw/r17-all/joint.pth
    uv run burst.py moving data/models/raw/r19-burst/joint.pth
    uv run burst.py slip
    uv run burst.py own data/models/raw/r19-burst/joint.pth FOLDER CLEAN

ladders reads small print (small_print.py) in a real burst we hold: a review site's ISO ladder is one
scene shot again at each ISO, a fraction of a pixel apart, so the frame at PICKED with the next one
or two up the ladder is a burst of uneven frames, each noisier than the last. Every model gives its
picture of the picked frame alone, "one frame": a row beside each burst, over the cameras that have
that burst's frames. A burst model then gets the frames as an app would hand them over, "burst":
lined up from the noisy frames themselves (lined_up()), each moved by whole colour cells until it
sits nearest the picked frame and no finer, with how far it still sits off, which a model trained
with --apart is told. "swapped" is that picture as it is used, behind the guard (below). A
one-frame model gets the frames averaged, each weighed by one over its ISO, with nothing moved, and
the lines read at their weighted place, "merged": the merge a burst model has to beat. Both readers
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

The offsets a model is told are measured ones in every test, because measured ones are all an app
has: the rows a model is held to are those. Beside them, for a model that is told, the same burst
with the offsets the test knows: in equal the ones it drew, "true offsets", and in ladders, where
nothing is drawn, the frames lined up by where the test found their printed lines, "lines'
offsets", which is how every burst model was read before there was a line-up. slip prints how far
off the line-up is: the Ricohs' full-colour windows as two frames a pixel apart, under made-up
noise at each ISO's levels, the measured offsets against the drawn ones. train_raw.slip() is fitted
to its lines, and the trainer lays that much error on the offsets it tells.

moving is for what a burst gets wrong: something that moved shows twice. It makes bursts of the
Ricohs' full-colour windows as the trainer does (train_raw.burst), every crop with a patch that
sits several px off in the later frames, under made-up noise at the levels of the camera's frame at
PICKED, and gives how far the picture is from the truth inside those patches, in 8-bit levels of
the developed picture, from the burst, from the burst behind the guard and from the picked frame
alone. Behind the guard it must not be the larger; outside the patches it should be the smaller.

own is for a burst of the user's: FOLDER holds its raw files and nothing else, the first by name
the picked frame, and CLEAN is a clean frame of the same scene. Each later frame is moved onto the
picked one by whole colour cells, a block at a time (onto()), and FOLDER/burst gets one.png, the
picked frame alone, merged.png, the lined-up frames averaged and cleaned as one, burst.png, and
clean.png, each developed as the test develops pictures.

The guard is how a burst model's picture is used: a burst's one fault is to show twice what moved
between its frames, so where a later frame differs from the picked one by more than their noise
allows (fired()), the picture is the picked frame's own, from a pass of the model on that frame
alone, faded into the burst's over FADE px so no edge shows (swapped()). A later frame differs
where its mean over NEAR colour cells a side sits further from the picked frame's, in any colour
plane, than ALLOWED times what the two frames' shot and read noise gives that mean. It is a mean
and not an energy on purpose: print a pixel apart differs from itself stroke by stroke, up one side
and down the other, which a mean over many strokes hardly sees, and a thing several pixels off
moves the mean. The noise is the frames' own, as bench_raw.noise_levels measured it or as the test
laid it, so the guard is for frames whose noise is known: own does not use it. The guard is on the
picture and not on the frames because taking such a place out of the later frames does not give
one frame's picture there: NAFNet weighs its channels by their means over the whole tile, so in a
tile that has later frames elsewhere such a place is cleaned as lightly as the rest. Its cost is
the second pass, on every tile where it fires: ladders counts those, of each of TILES px a side.

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
#: fired()'s neighbourhood, colour cells a side, how many times its noise a mean over it may differ,
#: and the px swapped() fades over. The sides of the tiles ladders counts the guard's firing in, px:
#: the app's, and half of it.
NEAR, ALLOWED, FADE = 5, 6.0, 4
TILES = (512, 256)
#: sits()'s band of green detail, colour cells, its rounds, the margin it leaves at a window's edges,
#: in widths of the band's coarse side, and how many times the median a cell may sit off and still count.
FINE, COARSE, ROUNDS, MARGIN, TRIM = 1.0, 4.0, 6, 3, 4


def sits(picked, other):
    """Where a noisy frame (H, W) sits from the picked one, measured from the two: (dx, dy) px at the
    window's middle and a turn about it in radians, as train_raw.sat() lays them out, so that what
    the picked frame holds at a place the other holds that far on.

    The picked frame's green detail between FINE and COARSE colour cells is laid on the other's by
    Gauss-Newton steps from the nearest whole cell, which a phase correlation finds. Noise finer
    than the band is not in it, and under the band two frames' noise is nothing they share, so it
    widens the answer and does not pull it. A step leaves out the cells where the two still differ
    by more than TRIM times the median: something that moved between the frames is not where the
    camera went, and left in it drags the fit by pixels. The detail is laid with Lanczos: OpenCV's
    cubic lays a smooth picture a twentieth short of the fraction asked for, and the fit came out
    that much short.
    """
    green = lambda cfa: raw_pairs.green(np.ascontiguousarray(cfa, np.float32))
    band = lambda plane: cv2.GaussianBlur(plane, (0, 0), FINE) - cv2.GaussianBlur(plane, (0, 0), COARSE)
    a, b = band(green(picked)), band(green(other))
    h, w = a.shape
    cx, cy, edge = (w - 1) / 2, (h - 1) / 2, int(MARGIN * COARSE)
    inner = (slice(edge, -edge),) * 2
    y, x = np.mgrid[:h, :w][(slice(None), *inner)] - np.array([cy, cx]).reshape(2, 1, 1)
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    at = np.array([round(dx), round(dy), 0.0])
    for _ in range(ROUNDS):
        cos, sin = np.cos(at[2]), np.sin(at[2])
        back = np.array(
            [
                [cos, sin, cx - cos * (cx + at[0]) - sin * (cy + at[1])],
                [-sin, cos, cy + sin * (cx + at[0]) - cos * (cy + at[1])],
            ]
        )
        laid = cv2.warpAffine(
            a, back, (w, h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT
        )
        gx, gy = -np.gradient(laid, axis=1)[inner], -np.gradient(laid, axis=0)[inner]
        rest = (b - laid)[inner].ravel()
        pulls = np.stack([gx, gy, gy * x - gx * y]).reshape(3, -1) * (np.abs(rest) < TRIM * np.median(np.abs(rest)))
        at = at + np.linalg.solve(pulls @ pulls.T, pulls @ rest)
    return 2 * at[0], 2 * at[1], at[2]


def measured(frames):
    """Where each later frame of a window's (count, H, W) sits from the first, measured from the noisy
    frames as an app has to: sits()'s (dx, dy, turn) for each (count - 1, 3)."""
    picked = frames[0].cpu().numpy()
    found = [sits(picked, frame.cpu().numpy()) for frame in frames[1:]]
    return torch.tensor(found, dtype=frames.dtype, device=frames.device)


def by_lines(lines):
    """Where each later frame sits from the first by where the test found their printed lines
    (count, lines, 4, 2): the mean of the corners' offsets and no turn (count - 1, 3)."""
    off = (lines[1:] - lines[0]).reshape(len(lines) - 1, -1, 2).mean(1)
    return torch.from_numpy(np.concatenate([off, np.zeros((len(off), 1))], 1)).float()


def lined_up(frames, places):
    """A window's frames (count, H, W) as an app hands a burst over, given where each later one sits
    from the first, places (count - 1, 3) as measured() finds them: each moved by whole colour cells
    until it sits nearest the first and no finer, and where it then sits from the first at each
    colour cell (count - 1, 2, H / 2, W / 2), in colour cells, as train_raw.fed() takes it."""
    places = places.to(frames)
    cells = 2 * (places[:, :2] / 2).round()
    moved = [torch.roll(frame, (-int(dy), -int(dx)), (0, 1)) for frame, (dx, dy) in zip(frames[1:], cells.tolist())]
    apart = train_raw.sat(places[:, :2] - cells, places[:, 2], *frames.shape[1:]) / 2
    return torch.stack([frames[0], *moved]), apart


def fired(bursts, levels):
    """Where the guard fires in bursts (N, frames, H, W): the colour cells (N, 1, H / 2, W / 2) where a
    later frame differs from the picked one by more than their noise allows.

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
    off = (box(later - picked).abs() > ALLOWED * noise).flatten(1, 2).any(1, keepdim=True).float()
    return torch.nn.functional.max_pool2d(off, NEAR, 1, NEAR // 2) > 0


def swapped(many, one, where):
    """A burst's picture (N, 3, H, W) with the picked frame's own, one, where the guard fired
    (fired()): all of one's there, fading into the burst's over FADE px past it."""
    full = where.float().repeat_interleave(2, 2).repeat_interleave(2, 3)
    wide = torch.nn.functional.max_pool2d(full, FADE + 1, 1, FADE // 2)
    share = torch.nn.functional.avg_pool2d(wide, FADE + 1, 1, FADE // 2, count_include_pad=False)
    return many + (one - many) * share


def tiles(where):
    """How often the guard fired in one window (1, 1, H / 2, W / 2): for each of TILES, how many tiles
    of that side the window is and in how many it fired at all, then the share of its cells swapped."""
    counts = []
    for side in TILES:
        any_in = torch.nn.functional.max_pool2d(where.float(), side // 2, side // 2, ceil_mode=True)
        counts += [any_in.numel(), float(any_in.sum())]
    return np.array([*counts, float(where.float().mean()), 1.0])


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
    """{model: {way: counts (2, 5)}} for one camera's windows, read()'s over its printed things; how
    often the guard fired in each burst's windows, {frames: tiles()'s counts summed}; and the furthest
    the line-up's offset at a window's middle sits from the lines', px. Nothing for a camera without a
    frame at PICKED and the next ISO."""
    things, isos, means, picture = small_print.opened(file)
    if PICKED not in isos or isos.index(PICKED) + 1 >= len(isos):
        return {}, {}, 0.0
    k, rows, swaps, furthest, levels = isos.index(PICKED), {}, {}, 0.0, np.load(file)["levels"]
    look = lambda result: small_print.as_bytes(picture(result, None))
    for thing, (frames, _, lines) in things.items():
        small = [n for n, line in enumerate(lines[0]) if np.linalg.norm(line[3] - line[0]) <= TALLEST]
        bursts = {}
        for more in [more for more in MORE if k + more < len(isos)]:
            used = "+".join(str(iso) for iso in isos[k : k + 1 + more])
            given, places = frames[k + 1 : k + 2 + more], lines[k + 1 : k + 2 + more]
            sat_at, read_off = measured(given), by_lines(places)
            (burst, apart), (rolled, shift) = lined_up(given, sat_at), lined_up(given, read_off)
            where = fired(burst[None], levels[k : k + 1 + more])
            swaps[used] = swaps.get(used, 0) + tiles(where)
            furthest = max(furthest, float((sat_at.cpu() - read_off)[:, :2].abs().max()))
            bursts[used] = (more, given, places, burst[None], apart[None], rolled[None], shift[None], where)
        for name, (clean_up, slots, told) in models.items():
            one, found = clean_up(frames[k + 1 : k + 2], means[k]), {}
            alone = read(look(one), thing, lines[k + 1], small)
            for used, (more, given, places, burst, apart, rolled, shift, where) in bursts.items():
                found[f"{used} one frame"] = alone
                if slots > more:
                    many = clean_up(burst, means[k], apart)
                    found[f"{used} burst"] = read(look(many), thing, places[0], small)
                    found[f"{used} swapped"] = read(look(swapped(many, one, where)), thing, places[0], small)
                    by = clean_up(rolled, means[k], shift)
                    found[f"{used} burst, lines' offsets"] = read(look(by), thing, places[0], small)
                if slots == 1:
                    weights = np.float32([PICKED / iso for iso in isos[k : k + 1 + more]])
                    weights /= weights.sum()
                    merged = sum(float(weight) * frame for weight, frame in zip(weights, given))[None]
                    place = sum(float(weight) * place for weight, place in zip(weights, places))
                    found[f"{used} merged"] = read(look(clean_up(merged, means[k])), thing, place, small)
            kept = rows.setdefault(name, {})
            for way, counts in found.items():
                kept[way] = kept.get(way, 0) + counts
    return rows, swaps, furthest


def shown(title, rows):
    """A test's rows printed, all lines and the lines up to TALLEST px, as small_print's small lines are."""
    for at, which in enumerate(("all lines", f"lines up to {TALLEST} px tall")):
        heads = ("letters", "read %", "right", "wrong", "missing", "wrong of spelt %", "second read %")
        print(f"\n{title}, {which}\n{'':60}" + "".join(f"{head:>{len(head) + 3}}" for head in heads))
        for name, ways in rows.items():
            for way, counts in ways.items():
                right, wrong, missing, second, there = counts[at]
                spelt = 100 * wrong / max(right + wrong, 1e-9)
                cells = (there, 100 * right / there, right, wrong, missing, spelt, 100 * second / there)
                row = "".join(f"{cell:>{len(head) + 3}.{head != 'letters':d}f}" for head, cell in zip(heads, cells))
                print(f"{name:20}{way:40}{row}", flush=True)


def entrants(specs):
    """{name: (cleaner, slots, whether it is told how far its frames sit apart)} for each NAME=WEIGHTS."""
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    models = {}
    for name, _, weights in (spec.partition("=") for spec in specs):
        model = small_print.trained(weights)
        models[name] = (train_raw.cleaner(model), train_raw.slots(model), hasattr(model, "apart"))
    return models


def kept(test, rows):
    """A test's rows added to OUT."""
    table = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model, ways in rows.items():
        found = {way: dict(zip(("all", "small"), counts.tolist())) for way, counts in ways.items()}
        table.setdefault(test, {}).setdefault(model, {}).update(found)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=1))


def fires(swaps):
    """tiles()'s counts, summed over windows, in words."""
    told = []
    for used, counts in swaps.items():
        sides = [
            f"{hit:.0f} of {there:.0f} tiles of {side} px ({hit / there:.0%})"
            for side, there, hit in zip(TILES, counts[0:-2:2], counts[1:-2:2])
        ]
        told.append(f"{used}: {', '.join(sides)}, {counts[-2] / counts[-1]:.1%} of the picture swapped")
    return "; ".join(told)


def ladders(*specs):
    every = (*bench_raw.CAMERAS, *bench_raw.groups())
    models = entrants([spec for spec in specs if spec not in every])
    for test in [test for test in every if test in specs] or ("gr3x", "gr3", "unseen"):
        together, fired_in, furthest = {}, {}, 0.0
        for name in bench_raw.cuts(f"print/{test}"):
            rows, swaps, apart = camera(bench_raw.BENCH / f"{name}.npz", models)
            furthest = max(furthest, apart)
            kept(name.split("/")[-1], rows)
            for model, ways in rows.items():
                pooled = together.setdefault(model, {})
                for way, counts in ways.items():
                    pooled[way] = pooled.get(way, 0) + counts
            for used, counts in swaps.items():
                fired_in[used] = fired_in.get(used, 0) + counts
            if rows and name.split("/")[-1] != test:
                print(f"{name}: the line-up up to {apart:.2f} px from the lines; the guard fired in {fires(swaps)}")
        if together:
            kept(test, together)
            shown(f"{test}, ISO {PICKED} picked, the line-up up to {furthest:.2f} px from the lines", together)
            print(f"{test}: the guard fired in {fires(fired_in)}", flush=True)


@torch.no_grad()
def equal(*specs):
    models = entrants(specs)
    file = small_print.READER.parent / "gr3x.npz"
    things, isos, means, picture = small_print.opened(file)
    data, rng, rows, taken, off = np.load(file), torch.Generator().manual_seed(0), {}, {}, {}
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
            "steady": (clean.expand(later, -1, -1), torch.zeros_like(move), torch.zeros_like(turn)),
            f"within {train_raw.SHAKE:g} px": (
                bench_raw.mosaiced(train_raw.shaken(full.expand(later, -1, -1, -1), move, turn)),
                move,
                turn,
            ),
        }
        bursts = {}
        for way, (scene, moved_by, turned_by) in scenes.items():
            burst = torch.cat([alone[PICKED][thing], small_print.made_up(scene, shot, read_noise, rng)])
            true, found_at = train_raw.sat(moved_by, turned_by, *clean.shape[1:]), measured(burst)
            wrong = train_raw.sat(found_at[:, :2], found_at[:, 2], *clean.shape[1:]) - true
            off[way] = max(off.get(way, 0), float(wrong.pow(2).mean().sqrt()))
            lined, told_off = lined_up(burst, found_at)
            where = fired(lined[None], [[shot, read_noise]] * FRAMES)
            taken[way] = taken.get(way, 0) + float(where.float().mean()) / len(things)
            bursts[way] = (burst[None], true[None] / 2, lined[None], told_off[None])
        for name, (clean_up, slots, told) in models.items():
            pooled, given, mean = rows.setdefault(name, {}), min(slots, FRAMES), means[len(under) - 1]
            found = {f"{iso} one frame": clean_up(alone[iso][thing], means[at]) for at, iso in enumerate(under)}
            for way, (burst, true, lined, told_off) in bursts.items() if slots > 1 else ():
                where = fired(lined[:, :given], [[shot, read_noise]] * given)
                many = clean_up(lined[:, :given], mean, told_off[:, : given - 1])
                found[f"{PICKED} {given} frames {way}"] = many
                found[f"{PICKED} {given} frames {way}, swapped"] = swapped(many, found[f"{PICKED} one frame"], where)
                if told:
                    found[f"{PICKED} {given} frames {way}, true offsets"] = clean_up(
                        burst[:, :given], mean, true[:, : given - 1]
                    )
            for way, result in found.items():
                pooled[way] = pooled.get(way, 0) + read(look(result), thing, lines[0], small)
    kept("equal", rows)
    shown(f"gr3x, clean windows under made-up noise, bursts at ISO {PICKED}'s strength", rows)
    for way, share in taken.items():
        print(
            f"{way}: the line-up up to {off[way]:.3f} px off over a window, the guard swapped {share:.1%} of the picture"
        )


def moved(name, clean_up, slots, rng):
    """One Ricoh's full-colour windows as bursts with a moved patch in each: how far the picture is from
    the truth, in 8-bit levels of the developed picture, inside the patches and then outside them,
    from the burst, from the burst behind the guard and from the picked frame alone; the share of
    the picture the guard swapped; and how far off the line-up was over a window, px."""
    data, levels = (
        np.load(bench_raw.BENCH / "colour" / f"{name}.npz"),
        np.load(small_print.READER.parent / f"{name}.npz"),
    )
    on = lambda key: torch.from_numpy(data[key].astype(np.float32)).to(small_print.DEVICE)
    truth, wb, matrix = on("truth"), on("wb"), on("matrix")
    shot, read_noise = levels["levels"][levels["isos"].tolist().index(PICKED)]
    seen, _, _, patch, true = train_raw.burst(truth, max(slots, 2), fewer=0, uneven=0, moving=1)
    frames = small_print.made_up(bench_raw.mosaiced(seen), shot, read_noise, rng)
    frames = frames.view(len(truth), -1, *truth.shape[-2:])
    found_at = [measured(burst) for burst in frames]
    lined, apart = (torch.stack(part) for part in zip(*(lined_up(burst, at) for burst, at in zip(frames, found_at))))
    wrong = torch.stack([train_raw.sat(at[:, :2], at[:, 2], *truth.shape[-2:]) for at in found_at]) - true
    mean = float(frames[:, 0].mean())
    exposure = 0.108 / bench_raw.develop(truth, wb, matrix).median()
    picture = lambda result: (bench_raw.develop(result, wb, matrix) * exposure).clamp(0, 1) ** (1 / 2.2)
    far = lambda result: 255 * (picture(result) - picture(truth)).abs().mean(1, keepdim=True)
    where = fired(lined, [[shot, read_noise]] * frames.shape[1])
    many, one = clean_up(lined, mean, apart), clean_up(frames[:, 0], mean)
    ways = [far(result) for result in (many, swapped(many, one, where), one)]
    found = [[float(way[place].mean()) for way in ways] for place in (patch, ~patch)]
    return found, float(where.float().mean()), float(wrong.pow(2).mean().sqrt())


@torch.no_grad()
def moving(weights):
    ((clean_up, slots, _),) = entrants([f"={weights}"]).values()
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for name in bench_raw.CAMERAS:
        found, share, off = moved(name, clean_up, slots, rng)
        told = [
            f"{place} burst {burst:.3f}, swapped {safe:.3f} ({safe / one:.3f} of one frame), one frame {one:.3f}"
            for place, (burst, safe, one) in zip(("inside the patches", "outside them"), found)
        ]
        print(
            f"{name}, ISO {PICKED}'s noise, {slots} frames, a moved patch in every window, 8-bit levels from the truth:"
            f" {'; '.join(told)}; the guard swapped {share:.1%} of the picture; the line-up {off:.3f} px off",
            flush=True,
        )


@torch.no_grad()
def slip():
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for name in bench_raw.CAMERAS:
        data, levels = (
            np.load(bench_raw.BENCH / "colour" / f"{name}.npz"),
            np.load(small_print.READER.parent / f"{name}.npz"),
        )
        truth = torch.from_numpy(data["truth"].astype(np.float32)).to(small_print.DEVICE)
        sides = truth.shape[-2:]
        move = (torch.rand(len(truth), 2, device=truth.device) * 2 - 1) * train_raw.SHAKE
        turn = (torch.rand(len(truth), device=truth.device) * 2 - 1) * train_raw.TURN
        scenes = bench_raw.mosaiced(truth), bench_raw.mosaiced(train_raw.shaken(truth, move, turn))
        gain = min(max(train_raw.LEVEL / float(scenes[0].mean()), 1), train_raw.MAX_GAIN)
        for iso, (shot, read_noise) in zip(levels["isos"].tolist(), levels["levels"].tolist()):
            pair = [small_print.made_up(scene, shot, read_noise, rng).cpu().numpy() for scene in scenes]
            found = torch.tensor([sits(first, second) for first, second in zip(*pair)], dtype=torch.float32)
            found = found.to(truth.device)
            wrong = train_raw.sat(found[:, :2], found[:, 2], *sides) - train_raw.sat(move, turn, *sides)
            each = wrong.pow(2).mean((1, 2, 3)).sqrt()
            noise = (2 * (train_raw.LEVEL * shot * gain + read_noise * gain**2)) ** 0.5
            print(
                f"{name} ISO {iso:>6}: two frames' noise {noise:.4f}, the line-up {each.pow(2).mean().sqrt():.3f} px off"
                f" either way over a window, the worst of {len(each)} windows {each.max():.3f},"
                f" the trainer draws {float(train_raw.slip(torch.tensor(noise))):.3f}",
                flush=True,
            )


def onto(picked, other):
    """A whole frame (H, W) moved onto the picked one by whole colour cells, a block of BLOCK px at a
    time, and how far each block then sits from the picked frame's, in colour cells (2, H / 2, W / 2),
    as train_raw.fed() takes it.

    raw_pairs.offsets says where each block sits in the other frame, to a fraction of a pixel, from a
    fit over the whole frame that covers a shift and a turn, up to REACH px: a hand turns the camera
    a little between frames, and a turn is another offset in every corner. A block is copied from
    there, rounded to whole colour cells, so nothing is resampled and a colour stays the colour its
    pixel measured. The other frame comes at the picked one's brightness.
    """
    at, out = raw_pairs.offsets(picked, other), np.empty_like(picked)
    left = np.zeros((2, picked.shape[0] // 2, picked.shape[1] // 2), np.float32)
    wide = np.pad(other, REACH, "reflect")
    starts = lambda side: sorted({*range(0, side - BLOCK, BLOCK), side - BLOCK})
    about = lambda start, side: int(np.clip(start + (BLOCK - raw_pairs.CROP) // 2, 0, side - raw_pairs.CROP)) // 2 * 2
    for y in starts(picked.shape[0]):
        for x in starts(picked.shape[1]):
            off, _ = at(about(y, picked.shape[0]), about(x, picked.shape[1]))
            dx, dy = (int(np.clip(round(part / 2) * 2, -REACH, REACH)) for part in off)
            out[y : y + BLOCK, x : x + BLOCK] = wide[y + REACH + dy :, x + REACH + dx :][:BLOCK, :BLOCK]
            left[:, y // 2 : (y + BLOCK) // 2, x // 2 : (x + BLOCK) // 2] = (
                np.reshape([off[0] - dx, off[1] - dy], (2, 1, 1)) / 2
            )
    return out, left


@torch.no_grad()
def own(weights, folder, clean):
    if small_print.DEVICE == "cpu":
        torch.set_num_threads(small_print.THREADS)
    folder = pathlib.Path(folder)
    model = small_print.trained(weights)
    paths = sorted(path for path in folder.iterdir() if path.is_file())[: max(train_raw.slots(model), 2)]
    picked, *later = (raw_pairs.mosaic(path) for path in paths)
    moved_onto, apart = (np.stack(part) for part in zip(*(onto(picked, frame) for frame in later)))
    burst = np.stack([picked, *moved_onto])
    wb, matrix = (torch.from_numpy(part) for part in bench_raw.colour(paths[0]))
    results = {
        "one": train_raw.denoise(model, picked),
        "merged": train_raw.denoise(model, burst.mean(0)),
        "burst": train_raw.denoise(model, burst, apart=apart),
    }
    develop = lambda result, way=None: bench_raw.develop(torch.from_numpy(result)[None], wb, matrix, way)
    pictures = {name: develop(result) for name, result in results.items()}
    exposures = dict.fromkeys(pictures, 0.108 / pictures["one"].median())
    pictures["clean"] = develop(onto(picked, raw_pairs.match(picked, raw_pairs.mosaic(clean))[0])[0], demosaic.menon)
    exposures["clean"] = 0.108 / pictures["clean"].median()
    (folder / "burst").mkdir(exist_ok=True)
    for name, linear in pictures.items():
        picture = small_print.as_bytes((linear * exposures[name]).clamp(0, 1) ** (1 / 2.2))
        cv2.imwrite(str(folder / "burst" / f"{name}.png"), picture[:, :, ::-1])
        print(folder / "burst" / f"{name}.png", flush=True)


if __name__ == "__main__":
    {"ladders": ladders, "equal": equal, "moving": moving, "slip": slip, "own": own}[sys.argv[1]](*sys.argv[2:])
