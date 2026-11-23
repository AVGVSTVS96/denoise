"""Turn a joint checkpoint into the raw model the app loads.

    uv run ship_raw.py data/models/raw/joint-v1/joint.pth      a finished run's weights
    uv run ship_raw.py data/models/raw/joint-v1/last.pth       a run still training: its averaged weights
    uv run ship_raw.py data/models/raw/r8-p00/joint.pth bright      a model trained with --bright
    uv run ship_raw.py data/models/raw/r16-crisp/joint.pth bright crisp      a second model, beside the first

The app packs a mosaic tile into four planes (R, G, G, B) and gets camera RGB at the mosaic's size
back, so the file holds train_raw's fed() and made() around the network: the fixed mix to twelve
planes, NAFNet, the pixel shuffle. Half precision, with the layer norms and pools left in fp32 as
ship.py leaves them (Hush's export), because WebGPU computes fp16 for real and those overflow it.

The app scales and tiles a photo as train_raw.denoise() does, tile_gain() and the fade between tiles
with it, and reads tile_gain()'s two numbers from the manifest this writes. A model trained with
--bright has seen lit parts of dark photos at the whole photo's gain, so for one shipped as bright the
manifest carries no numbers and every tile gets the photo's gain, as denoise() gives it with held off.
One thing differs, for light far brighter than the photo's mean:

CEILING: the network sees its input held under it, and what is held back is added to its picture as
it is. NAFNet adds its input to its output, so below the ceiling nothing changes. Above it the
network's values grow with the input: a lamp at the top gain drives them past 100,000, where fp16
ends at 65,504 and WebGPU would hand back a tile of NaN. After tile_gain() only lights smaller than
its block still pass the ceiling; without it every light a dark photo's gain lifts past it does.

Three checks before anything is written into the app:
  the largest value any fp16 step holds, on real photos' tiles at the gain the app gives them and
  on made-up ones past any photo (a frame at the highest gain that is all highlight, one colour
  blown, a hot pixel in the dark), which stops the export at half of fp16's range;
  the fp16 file against PyTorch on bench_raw.py's full-colour windows, every ISO of the GR III;
  the fp16 file against the fp32 one on the same windows, which is what half precision alone costs.

Writes public/model/<hash>/raw-joint-fp16.partN (under 20 MB each), public/model/raw.json (or the
manifest named last, which the app offers as a choice and fetches only when chosen), and
src/blacks.json, the black offsets blacks.py knows camera bodies by, which the app takes off as
raw_denoise.py does. A body is named there by a hash of its camera name and serial number, so the
app can look its own photo up and the review sites' serial numbers stay out of the repo.
"""

import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

import evaluate  # isort: split

# Hush's tools, which evaluate puts on the path.
import export

import bench_raw
import blacks
import raw_pairs
import train_raw

APP = pathlib.Path(__file__).parent.parent
PUBLIC = APP / "public" / "model"
PART = 20_000_000
TILE = 1024
CEILING = 8.0
PHOTOS = (
    raw_pairs.DATA / "his" / "R0006362.DNG",
    raw_pairs.LADDERS / "ir" / "ricoh-gr-iii" / "sli" / "102400.dng",
    raw_pairs.LADDERS / "ir" / "ricoh-gr-iii" / "sli" / "100.dng",
    raw_pairs.LADDERS / "ir" / "pentax-k-1" / "sli" / "100.dng",
)


class Picture(torch.nn.Module):
    """Packed RGGB planes (N, 4, H/2, W/2) in, camera RGB (N, 3, H, W) out."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("spread", torch.tensor(train_raw.SPREAD)[:, :, None, None])

    def forward(self, planes):
        held = planes.clamp(max=CEILING)
        spread = lambda mosaic: torch.nn.functional.conv2d(mosaic, self.spread)
        return torch.nn.functional.pixel_shuffle(self.model(spread(held)) + spread(planes - held), 2)


def loaded(path):
    """A joint model from joint.pth or a snapshot, or from a run's last.pth, whose averaged weights are the ones scored."""
    state = torch.load(path, map_location="cpu")
    step = f", step {state['step']}" if "step" in state else ""
    if "average" in state:
        state = {name.removeprefix("module."): value for name, value in state["average"].items() if "." in name}
    width, planes = state["intro.weight"].shape[:2]
    if planes != 12:
        sys.exit(f"{path} is a faithful model (mosaic out); the app has no demosaic for it")
    model = train_raw.network(width, True)
    model.load_state_dict(state)
    return Picture(model).eval(), f"NAFNet width {width}, joint denoise and demosaic on raw Bayer, {path.parent.name}{step}"


def exported(model, path):
    axes = {"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}
    torch.onnx.export(
        model,
        (torch.rand(1, 4, 128, 128),),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=axes,
        opset_version=export.OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)


def packed(mosaic):
    return train_raw.pack(torch.from_numpy(np.ascontiguousarray(mosaic))[None]).numpy()


def photo_tiles(path, held, count=3):
    """Tiles of a photo as the app feeds them: its darkest, its brightest, and the rest spread between."""
    mosaic = raw_pairs.mosaic(path)
    gain = float(np.clip(train_raw.LEVEL / max(float(mosaic.mean()), 1e-6), 1, train_raw.MAX_GAIN))
    corners = [(y, x) for y in range(0, mosaic.shape[0] - TILE, TILE) for x in range(0, mosaic.shape[1] - TILE, TILE)]
    corners.sort(key=lambda corner: mosaic[corner[0] : corner[0] + TILE, corner[1] : corner[1] + TILE].mean())
    picks = [corners[round(i * (len(corners) - 1) / (count - 1))] for i in range(count)]
    cut = lambda y, x: mosaic[y : y + TILE, x : x + TILE]
    scaled = lambda tile: tile * (train_raw.tile_gain(tile, gain) if held else gain)
    return {f"{path.parent.name}/{path.name} {y},{x}": packed(scaled(cut(y, x))) for y, x in picks}


def made_up_tiles():
    """Tiles past what a photo gives: MAX_GAIN on a frame that is all highlight, and the sharpest steps a sensor makes."""
    top = train_raw.MAX_GAIN
    rng = np.random.default_rng(0)
    side = TILE // 2
    flat = lambda *planes: np.broadcast_to(np.array(planes, np.float32).reshape(1, 4, 1, 1), (1, 4, side, side)).copy()
    edge = flat(top, top, top, top)
    edge[..., side // 2 :] = 0
    hot = flat(0.05, 0.05, 0.05, 0.05)
    hot[0, rng.integers(4, size=200), rng.integers(side, size=200), rng.integers(side, size=200)] = top
    stripes = flat(0, 0, 0, 0)
    stripes[..., ::2] = top
    return {
        "black": flat(0, 0, 0, 0),
        "below black": flat(-1, -1, -1, -1),
        "all highlight at the top gain": flat(top, top, top, top),
        "red blown at the top gain": flat(top, top / 8, top / 8, top / 64),
        "blue blown at the top gain": flat(top / 64, top / 8, top / 8, top),
        "greens apart at the top gain": flat(top / 2, top, 0, top / 2),
        "highlight against black": edge,
        "hot pixels in the dark": hot,
        "stripes of highlight": stripes,
        "noise over the whole range": rng.uniform(-1, top, (1, 4, side, side)).astype(np.float32),
    }


def peaks(fp32, fp16):
    """The fp32 graph with, for every step the fp16 file runs in fp16, its largest value as an output."""
    model = onnx.load(str(fp32))
    half = {node.name for node in onnx.load(str(fp16)).graph.node} - set(export.fp32_islands(model))
    names = []
    for node in [node for node in model.graph.node if node.name in half and node.op_type != "Constant"]:
        model.graph.node.append(onnx.helper.make_node("Abs", [node.output[0]], [f"{node.name}/size"]))
        model.graph.node.append(onnx.helper.make_node("ReduceMax", [f"{node.name}/size"], [node.name], keepdims=0))
        model.graph.output.append(onnx.helper.make_tensor_value_info(node.name, onnx.TensorProto.FLOAT, []))
        names.append(node.name)
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]), names


def audit(fp32, fp16, held):
    """Stops the export if an fp16 step comes within half of fp16's largest number."""
    session, names = peaks(fp32, fp16)
    tiles = made_up_tiles()
    for path in PHOTOS:
        tiles |= photo_tiles(path, held)
    worst = (0.0, "", "")
    for tile, planes in tiles.items():
        top, step = max(zip(map(float, session.run(names, {"input": planes})), names))
        print(f"{tile:48} largest value {top:9.1f} {step}", flush=True)
        worst = max(worst, (top, step, tile))
    if worst[0] > export.FP16_MAX / 2:
        sys.exit(f"too close to fp16's largest number ({export.FP16_MAX:.0f}): {worst}")
    return dict(zip(("largest", "step", "tile"), worst))


def parity(model, fp32, fp16, camera="colour/gr3"):
    """The fp16 file against PyTorch, and against the fp32 file, on a full-colour test's noisy windows, as denoise() scales them.

    The worst and the mean difference in the network's own units, and the PSNR between the two as
    the picture train_raw.tone makes of them, which is where a difference would show.
    """
    data = np.load(bench_raw.BENCH / f"{camera}.npz")
    sessions = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in (fp32, fp16)]
    toned = lambda picture: train_raw.tone(torch.from_numpy(picture)).clamp(0, 1)
    against = {"torch": [], "fp32": []}
    for frames, mean in zip(data["frames"].astype(np.float32), data["means"].tolist()):
        gain = min(max(train_raw.LEVEL / mean, 1), train_raw.MAX_GAIN)
        for mosaic in frames:
            planes = packed(mosaic * gain)
            with torch.no_grad():
                expected = model(torch.from_numpy(planes)).numpy()
            exact, half = (session.run(None, {"input": planes})[0] for session in sessions)
            for name, other in (("torch", expected), ("fp32", exact)):
                error = float(((toned(half) - toned(other)) ** 2).mean())
                against[name].append((float(np.abs(half - other).max()), float(np.abs(half - other).mean()), error))
    summed = lambda rows: {
        "windows": len(rows),
        "largest difference": max(row[0] for row in rows),
        "mean difference": float(np.mean([row[1] for row in rows])),
        "psnr": float(-10 * np.log10(np.mean([row[2] for row in rows]))),
        "lowest psnr": float(-10 * np.log10(max(row[2] for row in rows))),
    }
    return {f"fp16 against {name}": summed(rows) for name, rows in against.items()}


def shipped(fp16, name, held, manifest):
    """The file cut into the app's parts, under its hash, and the parts this manifest last named gone."""
    manifest = PUBLIC / f"{manifest}.json"
    if manifest.exists():
        shutil.rmtree(PUBLIC / json.loads(manifest.read_text())["parts"][0].split("/")[0], ignore_errors=True)
    blob, digest = fp16.read_bytes(), export.sha256_file(fp16)
    folder = PUBLIC / digest[:8]
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, start in enumerate(range(0, len(blob), PART)):
        (folder / f"raw-joint-fp16.part{i}").write_bytes(blob[start : start + PART])
        parts.append(f"{digest[:8]}/raw-joint-fp16.part{i}")
    brightest = {"block": train_raw.BLOCK, "level": train_raw.BRIGHTEST} if held else None
    found = {"name": name, "bytes": len(blob), "sha256": digest, "parts": parts, "brightest": brightest}
    manifest.write_text(json.dumps(found, indent=2) + "\n")
    named = lambda body: hashlib.sha256(body.encode()).hexdigest()[:16]
    rounded = {named(body): {iso: [round(v, 6) for v in row] for iso, row in rows.items()} for body, rows in blacks.table().items()}
    (APP / "src" / "blacks.json").write_text(json.dumps(rounded, separators=(",", ":")) + "\n")
    return manifest


def main(checkpoint, bright="", manifest="raw"):
    held = bright != "bright"
    model, name = loaded(pathlib.Path(checkpoint))
    fp32, fp16 = (evaluate.MODELS / f"{manifest}-joint-{precision}.onnx" for precision in ("fp32", "fp16"))
    exported(model, fp32)
    export.convert_fp16(fp32, fp16)
    report = {"model": name, "fp16 range": audit(fp32, fp16, held), "parity": parity(model, fp32, fp16)}
    print(json.dumps(report, indent=1))
    print(shipped(fp16, f"{name}, fp16", held, manifest), fp16.stat().st_size, export.sha256_file(fp16))


if __name__ == "__main__":
    main(*sys.argv[1:])
