"""Small print read: how much of the still life's printed text is still there in a denoiser's picture.

    uv run small_print.py build                                         find the print on every camera and cut its windows
    uv run small_print.py build gr3                                     only the cameras with that in their name
    uv run small_print.py r8-p00=data/models/raw/r8-p00/joint.pth ...   score models, and rivals by rivals.py's names
    uv run small_print.py gr3x unseen r8-p00=...                        on those tests alone, of gr3, gr3x, unseen and seen

Imaging Resource's still life has five printed things in it: a proportional scale's instructions and
four bottle labels. small_print.json holds what 46 of their lines say, 734 letters and digits, typed
off the GR IIIx's ISO 100 frame, and where each line is in that frame as four corners. The scene is
the same on every camera, so build finds each printed thing in a camera's own lowest ISO frame by
matching it to the GR IIIx's (found) and carries the corners over. A thing it cannot match, or whose
lines the clean frame itself reads under HALF of, is left out for that camera.

A window is one printed thing with ROOM px around it, cut from every ISO of bench_raw.ISOS as
bench_raw.build cuts its windows: the noisy frame on whole even pixels, the clean frame resampled onto
it, and each line's corners moved by what is left between the two, so a line is read where it is.

A model's picture of a window is developed as the test develops pictures and rounded to 8 bits. Each
line is then straightened into a strip HEIGHT px tall and spelt by PP-OCRv4's recogniser, the one
RapidOCR ships, run through onnxruntime. Nothing looks for the text: the place is known, and finding
it again in every picture is one of the two things that made such a number jump. A strip is cut five
times, each a fraction of a pixel along (SHIFTS), and a line's score is the mean over the five of the
letters right: its length less the edits between what was typed and what was spelt, letters and
digits only, capitals and small letters alike. "read" is the letters right over the letters there.

The letters not read right are of two kinds, and a denoiser that draws print can trade one for the
other: "missing" are the line's letters the reader did not spell at all, and "wrong" are the rest,
where it spelt another letter in a letter's place or a letter where there is none. Of the ways to
get from the typed line to the spelt one in the fewest edits, the one with the fewest letters
missing is taken (edits). The three add up to the letters there.

One reader's habits could pass for a model's, so a second one spells the same strips: EasyOCR's
English recogniser (github.com/JaidedAI/EasyOCR, Apache 2.0), a different network trained by other
people on other data, as "second", its letters right over the letters there. Where the two readers
rank two models differently, say so; do not average them.

Beside it, "kept" and "invented": quality.py's finest band, detail 2 to 3 px across, of the picture
inside the lines' corners against the lined-up clean frame's, as quality.measure has them. kept is
the share of the clean frame's strokes the picture holds and invented what else it holds there, over
the clean frame's. The clean frame is developed by demosaic.menon, whatever made the picture: through
the bilinear demosaic it is softer than a good result, and then ranks an untouched noisy frame over
any model. It is still a mosaic's picture, moved a fraction of a pixel and with a little noise of its
own, and it leans toward a picture that went through demosaic.menon itself. So read the two between
models that make their picture the same way, at one ISO: detail on real noise, which the full-colour
tests cannot give.

Rows go to data/out/print.json as {test: {name: {ISO: [read, kept, invented, wrong, missing, second]}}},
each a share of the letters there but kept and invented, a test being "gr3x",
"gr3", "unseen" or "seen" as on the board, a group as the mean over its cameras, and each camera
under its file's name. "noisy" is the frame as shot. "clean" is the camera's clean frame as it was
cut, through demosaic.menon: one reading, the same at every ISO, of a picture with no noise in it. It
is not the lined-up clean frame, the other thing that made the number jump: a mosaic moved an odd
pixel along is each colour plane moved half a pixel, which softens print enough to lose a third of
its letters, and the GR IIIx's noisy frames sit that far off its clean one. Nothing a model is given
is resampled.

Two more kinds of row say what a camera's own noise costs beyond its strength. "noisy-menon" is the
noisy frame through demosaic.menon and no denoiser: "noisy" goes through the test's bilinear demosaic,
which alone loses print, so this is the row a model has to beat. And every name has a second row,
"<name>-made": the same cleaner on the camera's clean window under made-up noise, shot and read noise
at the levels build measured on that ISO's whole noisy frame (bench_raw.noise_levels, kept with the
windows as levels), read where the clean frame's lines are, its kept and invented against the clean
frame as cut. The noise is drawn on the CPU from one seed a camera, so every model and both machines
meet the same frame. Where "noisy-made" reads as "noisy" does, the made-up noise is as strong as the
real; a camera that smooths its raw measures low, and its made-up noise is then a little weak. A
model's made row less its real row, camera by camera, is printed under each test: next to nothing
where the raw is the scene plus noise, many points where the camera has smoothed its raw.

It runs on the GPU when there is one and on the CPU when not, there in float32 where the GPU runs
bfloat16, so compare rows made on the same machine.
"""

import functools
import io
import json
import string
import sys
import urllib.request
import zipfile

import cv2
import numpy as np
import onnxruntime
import torch

import bench_raw
import demosaic
import quality
import raw_pairs

LINES = json.loads((raw_pairs.DATA.parent / "small_print.json").read_text())
READER = bench_raw.BENCH / "print" / "reader.onnx"
WHEEL = (
    "https://files.pythonhosted.org/packages/ba/12/1e5497183bdbe782dbb91bad1d0d2297dba4d2831b2652657f7517bfc6df/"
    "rapidocr_onnxruntime-1.4.4-py3-none-any.whl"
)
RECOGNISER = "rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx"
SECOND = bench_raw.BENCH / "print" / "second.pth"
RELEASE = "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"
#: What the second reader spells with, after the blank between two letters, and how tall it takes a strip.
LETTERS, TALL = string.digits + string.punctuation + " €" + string.ascii_uppercase + string.ascii_lowercase, 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THREADS = 4
HEIGHT, NARROWEST = 48, 320
SHIFTS = ((0, 0), (0.5, 0.25), (-0.5, -0.25), (0.25, -0.5), (-0.25, 0.5))
ROOM, AROUND = 48, 150
MATCHES, HALF = 25, 0.5


def cells(cfa, wb, matrix):
    """A mosaic as linear sRGB at half its size (H/2, W/2, 3): each 2x2 cell one pixel."""
    return (np.stack([cfa[0::2, 0::2], raw_pairs.green(cfa), cfa[1::2, 1::2]], -1) * wb) @ matrix.T


def grey(linear, exposure):
    """Linear sRGB (H, W, 3) as the brightness of the picture the test would show, in 8 bits."""
    return cv2.cvtColor(((linear * exposure).clip(0, 1) ** (1 / 2.2) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def developed(cfa, wb, matrix, exposure, box):
    """A box (x0, y0, x1, y1) of a mosaic, on even pixels, as grey()."""
    x0, y0, x1, y1 = box
    crop = torch.from_numpy(np.ascontiguousarray(cfa[y0:y1, x0:x1]))[None]
    linear = bench_raw.develop(crop, torch.from_numpy(wb), torch.from_numpy(matrix))
    return grey(linear[0].permute(1, 2, 0).numpy(), exposure)


def around(points, room, shape, step=2):
    """The box (x0, y0, x1, y1) room px around points (N, 2): from an even pixel, a multiple of step px
    each way, and no nearer a frame's edge than a frame may sit off its clean one."""
    edge = 2 * raw_pairs.MAX_SHIFT
    low = np.floor((points.min(0) - room).clip(edge) / 2).astype(int) * 2
    most = (np.array(shape[::-1]) - edge - low) // step * step
    high = low + np.minimum(np.ceil((points.max(0) + room - low) / step).astype(int) * step, most)
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def laid(ours, theirs, reach):
    """The map from one grey picture's points to another's that most SIFT matches agree on within
    reach px, and how many agree."""
    sift = cv2.SIFT_create(nfeatures=20000)
    (at, described), (to, also) = (sift.detectAndCompute(image, None) for image in (ours, theirs))
    if described is None or also is None or len(at) < 2 or len(to) < 2:
        return None, 0
    close = [a for a, b in cv2.BFMatcher().knnMatch(described, also, k=2) if a.distance < 0.75 * b.distance]
    if len(close) < MATCHES:
        return None, 0
    start = np.float32([at[match.queryIdx].pt for match in close])
    end = np.float32([to[match.trainIdx].pt for match in close])
    move, agree = cv2.findHomography(start, end, cv2.RANSAC, reach)
    return move, 0 if move is None else int(agree.sum())


def sized(move, before=1.0, after=1.0, start=(0, 0), end=(0, 0)):
    """A map between two crops as the map between their frames: each crop's scale and its corner."""
    grow = lambda scale, corner: np.array([[scale, 0, corner[0]], [0, scale, corner[1]], [0, 0, 1]])
    return grow(1 / after, end) @ move @ np.linalg.inv(grow(1 / before, start))


def found(frame, reference):
    """Where each printed thing's lines are in a camera's clean frame: {thing: corners (lines, 4, 2)}.

    A frame is its mosaic, white balance, matrix and exposure. The whole pictures are matched first at
    a quarter of their size, which says roughly where a thing is, then the thing itself at full size
    with AROUND px about it: a label on a bottle is no flat sheet, and each camera stood where it
    stood, so every thing gets a map of its own.
    """
    small = lambda cfa, wb, matrix, exposure: cv2.resize(
        grey(cells(cfa, wb, matrix), exposure), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )
    rough, _ = laid(small(*reference), small(*frame), 8)
    if rough is None:
        return {}
    rough, places = sized(rough, 0.25, 0.25), {}
    for thing, lines in LINES.items():
        corners = np.float32([line for _, line in lines])
        ours = around(corners.reshape(-1, 2), AROUND, reference[0].shape)
        edge = np.float32([[ours[0], ours[1]], [ours[2], ours[1]], [ours[2], ours[3]], [ours[0], ours[3]]])
        theirs = around(cv2.perspectiveTransform(edge[None], rough)[0], AROUND, frame[0].shape)
        move, agree = laid(developed(*reference, ours), developed(*frame, theirs), 2.5)
        print(f"   {thing:12} matches {agree}", flush=True)
        if agree >= MATCHES:
            places[thing] = cv2.perspectiveTransform(corners, sized(move, start=ours[:2], end=theirs[:2]))
    return places


def lit(path):
    """A raw as found() takes a frame: its mosaic, white balance and matrix, and the exposure that
    brings its picture's median to mid-grey."""
    clean = raw_pairs.mosaic(path)
    wb, matrix = bench_raw.colour(path)
    return clean, wb, matrix, clean is not None and float(0.108 / np.median(cells(clean, wb, matrix)))


def build(name, path, base, isos, reference):
    """A camera's windows, one printed thing each: frames (1 + ISOs, H, W) with the clean one first,
    twins (ISOs, H, W), and lines (1 + ISOs, lines, 4, 2), each line's corners in every frame's window.
    With them levels (ISOs, 2), each whole noisy frame's shot and read noise as bench_raw.noise_levels has them."""
    clean, wb, matrix, exposure = lit(path(base))
    if clean is None or not isos:
        return
    print(name, flush=True)
    places = found((clean, wb, matrix, exposure), reference)
    if not places:
        return
    boxes = {thing: around(corners.reshape(-1, 2), ROOM, clean.shape, 32) for thing, corners in places.items()}
    cut = lambda mosaic, box, dx=0, dy=0: mosaic[box[1] + dy : box[3] + dy, box[0] + dx : box[2] + dx]
    kept = {thing: ([cut(clean, box)], [], [places[thing] - box[:2]]) for thing, box in boxes.items()}
    means, levels = [], []
    for iso in isos:
        noisy = raw_pairs.mosaic(path(iso))
        dx, dy = raw_pairs.shift_between(clean, noisy)
        noisy, *_ = raw_pairs.match(clean, noisy)
        at = raw_pairs.offsets(clean, noisy)
        means.append(float(noisy.mean()))
        levels.append(bench_raw.noise_levels(clean, noisy))
        for thing, (x0, y0, x1, y1) in boxes.items():
            middle = lambda low, high, side: int(
                np.clip((low + high - raw_pairs.CROP) // 4 * 2, 0, side - raw_pairs.CROP)
            )
            off, share = at(middle(y0, y1, clean.shape[0]), middle(x0, x1, clean.shape[1]))
            left = np.array(off) - (dx, dy)
            wx, wy = (np.round(left / 2) * 2).astype(int)
            frames, twins, lines = kept[thing]
            frames.append(cut(noisy, boxes[thing], dx, dy))
            twins.append(raw_pairs.resampled(clean, y0 - wy, x0 - wx, *(left - (wx, wy)), y1 - y0, x1 - x0))
            lines.append(lines[0] + left)
            print(f"   {iso:>6} {thing:12} left {left.round(2)} lined up {share:.2f}", flush=True)
    arrays = {}
    for thing, (frames, twins, lines) in kept.items():
        arrays |= {
            f"{thing}/frames": np.stack(frames).astype(np.float16),
            f"{thing}/twins": np.stack(twins).astype(np.float16),
            f"{thing}/lines": np.stack(lines).astype(np.float32),
        }
    bench_raw.save(
        f"print/{name}", means=means, levels=levels, wb=wb, matrix=matrix, isos=isos, exposure=exposure, **arrays
    )
    weed(name)


def fetch():
    """PP-OCRv4's recogniser, out of RapidOCR's wheel on PyPI, and EasyOCR's English one, out of its
    releases on GitHub, beside the windows: the PC reads them from there."""
    for file, source, packed in ((READER, WHEEL, RECOGNISER), (SECOND, RELEASE, "english_g2.pth")):
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(source) as archive:
                file.write_bytes(zipfile.ZipFile(io.BytesIO(archive.read())).read(packed))


def build_all(only=""):
    fetch()
    reference = lit(bench_raw.ricoh("gr3x")(100))
    ladders = {camera: (bench_raw.ricoh(camera), 100, bench_raw.ISOS) for camera in bench_raw.CAMERAS}
    for group, folders in bench_raw.groups().items():
        ladders |= {f"{group}--{folder.parent.name}": bench_raw.ladder(folder) for folder in folders}
    for name, ladder in ladders.items():
        if only in name:
            build(name, *ladder, reference)


@functools.cache
def reader():
    """The recogniser and the letters it knows, the first being the blank between two of them."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = THREADS
    session = onnxruntime.InferenceSession(str(READER), options, providers=["CPUExecutionProvider"])
    return session, ["", *session.get_modelmeta().custom_metadata_map["character"].splitlines(), " "]


def strips(picture, corners):
    """A line of an 8-bit picture (H, W, 3) straightened to HEIGHT px tall, once for each of SHIFTS."""
    wide = max(
        HEIGHT, round(HEIGHT * np.linalg.norm(corners[1] - corners[0]) / np.linalg.norm(corners[3] - corners[0]))
    )
    flat = np.float32([[0, 0], [wide, 0], [wide, HEIGHT], [0, HEIGHT]])
    return [
        cv2.warpPerspective(
            picture,
            cv2.getPerspectiveTransform(corners + np.float32(shift), flat),
            (wide, HEIGHT),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for shift in SHIFTS
    ]


def spelt(strip):
    """What the recogniser reads in a strip: each column's likeliest letter, repeats and blanks dropped.
    One strip at a time: several at once take it ten times as long each on a CPU."""
    session, letters = reader()
    fed = np.zeros((1, 3, HEIGHT, max(NARROWEST, strip.shape[1])), np.float32)
    fed[0, :, :, : strip.shape[1]] = strip[:, :, ::-1].transpose(2, 0, 1) / 127.5 - 1
    columns = session.run(None, {session.get_inputs()[0].name: fed})[0].argmax(2)[0]
    return "".join(letters[i] for j, i in enumerate(columns) if i and (j == 0 or i != columns[j - 1]))


class Lines(torch.nn.Module):
    """EasyOCR's second-generation recogniser, layer for layer: a strip of grey (N, 1, TALL, W) through
    seven convolutions to a column of features every 4 px, two LSTMs that run along the line both
    ways, and each column's score for the blank and for each of LETTERS."""

    def __init__(self, width=256):
        super().__init__()
        conv = lambda a, b, size=3, edge=1, **more: (torch.nn.Conv2d(a, b, size, padding=edge, **more), torch.nn.ReLU())
        normed = lambda a, b: (conv(a, b, bias=False)[0], torch.nn.BatchNorm2d(b), torch.nn.ReLU())
        pool = torch.nn.MaxPool2d
        self.strokes = torch.nn.Sequential(
            *conv(1, width // 8),
            pool(2),
            *conv(width // 8, width // 4),
            pool(2),
            *conv(width // 4, width // 2),
            *conv(width // 2, width // 2),
            pool((2, 1)),
            *normed(width // 2, width),
            *normed(width, width),
            pool((2, 1)),
            *conv(width, width, 2, 0),
        )
        along = lambda: (
            torch.nn.LSTM(width, width, bidirectional=True, batch_first=True),
            torch.nn.Linear(2 * width, width),
        )
        self.line = torch.nn.ModuleList([*along(), *along(), torch.nn.Linear(width, len(LETTERS) + 1)])

    def forward(self, strip):
        columns = self.strokes(strip).mean(2).transpose(1, 2)
        for part in self.line:
            columns = part(columns)[0] if isinstance(part, torch.nn.LSTM) else part(columns)
        return columns


@functools.cache
def second():
    """The second reader with EasyOCR's weights, which name their layers otherwise and hold them in this order."""
    model = Lines()
    model.load_state_dict(dict(zip(model.state_dict(), torch.load(SECOND, "cpu").values())))
    return model.eval()


@torch.no_grad()
def spelt_again(strip):
    """What the second reader reads in a strip, made TALL px tall and grey for it: spelt() by another network."""
    wide = round(strip.shape[1] * TALL / HEIGHT)
    grey = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY), (wide, TALL), interpolation=cv2.INTER_CUBIC)
    columns = second()(torch.from_numpy(grey)[None, None] / 127.5 - 1).argmax(2)[0].tolist()
    return "".join(LETTERS[i - 1] for j, i in enumerate(columns) if i and (j == 0 or i != columns[j - 1]))


def plain(text):
    return "".join(letter for letter in text.lower() if letter.isascii() and letter.isalnum())


def edits(typed, seen):
    """The fewest edits from a line as typed to what was spelt, and how many of them leave a letter of
    the line unspelt: the fewest that so few edits can. The rest put another letter in a letter's
    place or a letter where there is none."""
    typed, seen = plain(typed), plain(seen)
    row = [(j, 0) for j in range(len(seen) + 1)]
    for i, a in enumerate(typed, 1):
        last, row[0] = row[0], (i, i)
        for j, b in enumerate(seen, 1):
            (above, lost), (beside, gone) = row[j], row[j - 1]
            last, row[j] = row[j], min((above + 1, lost + 1), (beside + 1, gone), (last[0] + (a != b), last[1]))
    return row[-1]


def told(typed, seen):
    """A line's letters as right, wrong and missing in what was spelt: its length less the edits, never
    under none, is right; of the others, edits()'s unspelt letters are missing and the rest wrong."""
    there, (changed, unspelt) = len(plain(typed)), edits(typed, seen)
    right = max(0, there - changed)
    return right, there - right - unspelt, unspelt


def lettered(picture, thing, corners):
    """The letters of a printed thing's lines that are right, wrong and missing, and the letters the
    second reader gets right, each line's the mean over SHIFTS, and after them the letters there."""
    counts = [
        np.mean([(*told(text, spelt(strip)), told(text, spelt_again(strip))[0]) for strip in strips(picture, line)], 0)
        for (text, _), line in zip(LINES[thing], corners)
    ]
    return np.array([*map(np.sum, zip(*counts)), sum(len(plain(text)) for text, _ in LINES[thing])])


def strokes(picture, truth, corners):
    """quality.sums of a picture's finest band (1, 3, H, W) and its truth's, inside the lines' corners."""
    inside = np.zeros(picture.shape[-2:], np.uint8)
    cv2.fillPoly(inside, [np.int32(line.round()) for line in corners], 1)
    inside = torch.from_numpy(inside).bool().to(picture.device)[None, None]
    made, real = (quality.bands(quality.planes(shown))[0][:, :1] for shown in (picture, truth))
    return quality.sums(made, real, inside)


def opened(file):
    """A camera's windows: {thing: (frames, twins, lines)}, its ISOs and means, and picture(), which
    develops a result (1, H, W) or (1, 3, H, W) as the test does, through a demosaic if one is named."""
    data = np.load(file)
    on = lambda key: torch.from_numpy(data[key].astype(np.float32)).to(DEVICE)
    wb, matrix, exposure = on("wb"), on("matrix"), float(data["exposure"])
    picture = lambda result, way=None: (bench_raw.develop(result, wb, matrix, way) * exposure).clamp(0, 1) ** (1 / 2.2)
    things = {
        thing: (on(f"{thing}/frames"), on(f"{thing}/twins"), data[f"{thing}/lines"])
        for thing in LINES
        if f"{thing}/frames" in data
    }
    return things, data["isos"].tolist(), data["means"].tolist(), picture


def as_bytes(picture):
    return (picture[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def as_cut(things, picture):
    """What the readers make of each printed thing in a camera's clean frame as it was cut, through
    demosaic.menon: {thing: lettered()'s counts}."""
    return {
        thing: lettered(as_bytes(picture(frames[:1], demosaic.menon)), thing, lines[0])
        for thing, (frames, _, lines) in things.items()
    }


def weed(name):
    """Drops from a camera's windows the printed things its own clean frame reads under HALF of: print
    that is not there, or not where it was looked for. A camera left with none loses its file."""
    file = bench_raw.BENCH / "print" / f"{name}.npz"
    things, _, _, picture = opened(file)
    read = as_cut(things, picture)
    poor = [thing for thing, (hit, *_, size) in read.items() if hit < HALF * size]
    print("   clean frame reads", *(f"{thing} {hit / size:.2f}" for thing, (hit, *_, size) in read.items()), flush=True)
    if poor:
        print("   left out", *poor, flush=True)
        kept = {key: value for key, value in np.load(file).items() if key.split("/")[0] not in poor}
        file.unlink()
        if len(poor) < len(things):
            bench_raw.save(f"print/{name}", **kept)


def trained(weights):
    """train_raw.trained on whichever machine this is."""
    import train_raw

    state = torch.load(weights, map_location="cpu")
    model = train_raw.shaped(state)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def entrants(specs):
    """{name: (cleaner, demosaic)}: the noisy frame, once through the test's demosaic and once through
    demosaic.menon, each model, a mosaic-out one a second time through demosaic.menon, and rivals by
    rivals.PICTURES' names."""
    import train_raw

    untouched = lambda mosaics, mean: mosaics
    found = {"noisy": (untouched, None), "noisy-menon": (untouched, demosaic.menon)}
    for spec in specs:
        name, _, weights = spec.partition("=")
        if not weights:
            import rivals

            found[name] = (rivals.PICTURES[name](), None)
            continue
        model = trained(weights)
        found[name] = (train_raw.cleaner(model), None)
        if not train_raw.joint(model):
            found[f"{name}-menon"] = (found[name][0], demosaic.menon)
    return found


def made_up(clean, shot, read, rng):
    """A clean mosaic under shot and read noise, variance = shot * brightness + read, drawn on the CPU."""
    still = clean.cpu().clamp(min=0)
    noisy = torch.poisson(still / shot, generator=rng) * shot + torch.randn(still.shape, generator=rng) * read**0.5
    return noisy.to(clean.device)


@torch.no_grad()
def scores(cleaners, file):
    """{name: {ISO: [read, kept, invented, wrong, missing, second]}} for one camera's windows: each
    cleaner on the real noisy frame and, as "<name>-made", on the clean frame under made-up noise at
    that frame's levels; and "clean", its clean frame as cut."""
    things, isos, means, picture = opened(file)
    hit, wrong, missing, again, size = np.sum(list(as_cut(things, picture).values()), 0)
    rows = {"clean": {iso: [hit / size, 1.0, 0.0, wrong / size, missing / size, again / size] for iso in isos}}
    sharp = {thing: picture(frames[:1], demosaic.menon) for thing, (frames, _, _) in things.items()}
    rng = torch.Generator().manual_seed(0)
    for k, (iso, mean, (shot, read)) in enumerate(zip(isos, means, np.load(file)["levels"].tolist())):
        real = {
            thing: (frames[k + 1 : k + 2], picture(twins[k : k + 1], demosaic.menon), lines[k + 1])
            for thing, (frames, twins, lines) in things.items()
        }
        made = {
            thing: (made_up(frames[:1], shot, read, rng), sharp[thing], lines[0])
            for thing, (frames, _, lines) in things.items()
        }
        for name, (clean_up, way) in cleaners.items():
            for ending, given in (("", real), ("-made", made)):
                hits, together = [], []
                for thing, (frame, truth, corners) in given.items():
                    shown = picture(clean_up(frame, mean), way)
                    hits.append(lettered(as_bytes(shown), thing, corners))
                    together.append(strokes(shown, truth, corners))
                hit, wrong, missing, again, size = np.sum(hits, 0)
                _, _, _, xx, xy, yy = torch.stack(together).sum(0).tolist()
                rows.setdefault(name + ending, {})[iso] = [
                    hit / size,
                    xy / yy,
                    (max(xx - xy**2 / yy, 0) / yy) ** 0.5,
                    wrong / size,
                    missing / size,
                    again / size,
                ]
    return rows


def apart(rows, model, isos):
    """A model's made row less its real row, in points of read: a cell for each of isos, and their mean."""
    gaps = {iso: 100 * (rows[f"{model}-made"][iso][0] - row[0]) for iso, row in rows[model].items()}
    cells = [f"{gaps[iso]:.1f}" if iso in gaps else "" for iso in isos] + [f"{np.mean(list(gaps.values())):.1f}"]
    return "".join(f"{cell:>9}" for cell in cells)


def main(specs):
    if DEVICE == "cpu":
        torch.set_num_threads(THREADS)
    kept_at = raw_pairs.DATA / "out" / "print.json"
    table = json.loads(kept_at.read_text()) if kept_at.exists() else {}
    every = (*bench_raw.CAMERAS, *bench_raw.groups())
    if not any(bench_raw.cuts(f"print/{test}") for test in every):
        sys.exit(f"no windows in {READER.parent}: build them on the server, or mount that folder here")
    cleaners = entrants([spec for spec in specs if spec not in every])
    for test in [test for test in every if test in specs] or every:
        each = {name: scores(cleaners, bench_raw.BENCH / f"{name}.npz") for name in bench_raw.cuts(f"print/{test}")}
        for name, found in each.items():
            table.setdefault(name.split("/")[-1], {}).update(found)
        rows = table.setdefault(test, {}) if each else {}
        for model in next(iter(each.values()), {}):
            rows[model] = bench_raw.over_cameras([by_iso[model] for by_iso in each.values()])
        kept_at.write_text(json.dumps(table, indent=1))
        if rows:
            heads = [*rows["noisy"], "mean"]
            print(
                f"\n{test}  small print read / finest detail kept in it\n{'':24}"
                + "".join(f"{head:>14}" for head in heads)
            )
            for model, by_iso in rows.items():
                shown = [*by_iso.values(), np.mean(list(by_iso.values()), axis=0)]
                print(f"{model:24}" + "".join(f"{f'{row[0]:.3f} {row[1]:.3f}':>14}" for row in shown), flush=True)
            print(
                f"\n{test}  letters wrong / letters missing / read by the second reader\n{'':24}"
                + "".join(f"{head:>18}" for head in heads)
            )
            for model, by_iso in rows.items():
                shown = [*by_iso.values(), np.mean(list(by_iso.values()), axis=0)]
                print(f"{model:24}" + "".join(f"{' '.join(f'{part:.3f}' for part in row[3:]):>18}" for row in shown))
            print(
                f"\n{test}  made less real, points of small print read\n{'':40}"
                + "".join(f"{head:>9}" for head in heads)
            )
            for model in [model for model in next(iter(each.values())) if f"{model}-made" in rows]:
                print(f"{model:40}{apart(rows, model, heads[:-1])}", flush=True)
                for name, found in each.items() if len(each) > 1 else ():
                    print(f"   {name.split('--')[-1]:37}{apart(found, model, heads[:-1])}", flush=True)


if __name__ == "__main__":
    build_all(*sys.argv[2:]) if sys.argv[1] == "build" else main(sys.argv[1:])
