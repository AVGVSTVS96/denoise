"""Raw denoisers on his DNG and the GR IIIx night pairs, developed exactly like the reference renders.

    uv run raw_denoise.py rawnind MODEL.onnx    darktable 5.6's RawNIND Bayer model (darktable-ai rawdenoise-nind)
    uv run raw_denoise.py rawforge DIR          RawForge outputs, DIR/<raw name>.dng, made by
                                                rawforge TreeNetDenoiseHeavy IN.DNG OUT.dng --cfa --onnx

Each model ends in a denoised CFA mosaic in the camera's own raw units, the way darktable and RawForge
both hand it back. That mosaic replaces the DNG's sensor data and rawpy develops it with the reference
settings: his crops from the camera-WB render his R0006362.png came from (its auto brightness held
fixed), the pairs through night_pairs.py at the noisy frame's own gain. Writes
data/out/frontier/R0006362-<crop>_<method>.png and data/out/pairs/gr3x-<iso>-labels_<method>.png.
"""

import pathlib
import sys

import cv2
import numpy as np
import rawpy

import night_pairs

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
HIS = DATA / "his" / "R0006362.DNG"
#: Top-left (y, x) of each of his 1024 crops in R0006362.png.
HIS_CROPS = {"signs": (1688, 2488), "truck": (1728, 488), "facade": (800, 4100)}
PAIRS = DATA / "night-raw" / "imaging-resource"
PAIR_ISOS = ("025600", "102400")
PAIR_CENTRE = (0.62, 0.68)
#: darktable's tiling for this model: 512 px packed tiles, cores 32 px in from each edge, ramps across seams.
TILE = 512
OVERLAP = 32


def pair_raw(iso):
    return PAIRS / f"ricoh-gr-iiix--GR3xhSLI{iso}NR0.DNG"


def raws():
    return [HIS] + [pair_raw(iso) for iso in PAIR_ISOS]


def rawnind(path, model):
    """darktable's dt_restore_raw_bayer: black-subtract and range-normalise, no WB, a gain per tile, re-mosaic."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    with rawpy.imread(str(path)) as raw:
        cfa = raw.raw_image_visible.astype(np.float32)
        pattern = raw.raw_pattern
        black = np.asarray(raw.black_level_per_channel, np.float32)[pattern]
        white = float(raw.white_level)
        site_colour = np.asarray([raw.color_desc.decode()[c] for c in pattern.ravel()]).reshape(2, 2)
    # Packed R, G1, G2, B as the model expects; this camera's mosaic already starts on red.
    assert (site_colour == [["R", "G"], ["G", "B"]]).all(), site_colour
    level = np.tile(black, (cfa.shape[0] // 2, cfa.shape[1] // 2))
    span = white - level
    normalised = (cfa - level) / span
    packed = np.stack([normalised[i::2, j::2] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))])
    hh, wh = packed.shape[1:]
    step = TILE - 2 * OVERLAP
    rows, cols = -(-hh // step), -(-wh // step)
    padded = np.pad(
        packed, ((0, 0), (OVERLAP, rows * step - hh + OVERLAP), (OVERLAP, cols * step - wh + OVERLAP)), "reflect"
    )
    seam = 2 * OVERLAP  # sensor px either side of a core edge
    ramp = (np.arange(2 * seam) + 0.5) / (2 * seam)

    def weights(first, last):
        """Along one axis of a tile's output: ramps across shared seams, nothing for the mirrored border."""
        w = np.ones(2 * TILE, np.float32)
        if first:
            w[:seam] = 0
        else:
            w[: 2 * seam] = ramp
        if last:
            w[-seam:] = 0
        else:
            w[-2 * seam :] = 1 - ramp
        return w

    # Accumulated in sensor space shifted by one seam, so the first tile's mirrored border has room.
    total = np.zeros((2 * rows * step + 2 * seam, 2 * cols * step + 2 * seam), np.float32)
    weight = np.zeros_like(total)
    site = np.indices((2 * TILE, 2 * TILE)) % 2
    channel = np.array([[0, 1], [1, 2]])[site[0], site[1]]
    for ty in range(rows):
        for tx in range(cols):
            tile = np.ascontiguousarray(padded[None, :, ty * step : ty * step + TILE, tx * step : tx * step + TILE])
            out = session.run(None, {session.get_inputs()[0].name: tile})[0][0]
            mosaic = np.take_along_axis(out, channel[None], 0)[0]
            # darktable matches the 3-channel output's mean to the mosaic's, which counts green twice and
            # renders ~10% bright here; the re-mosaicked output against the mosaic keeps the brightness.
            mosaic *= tile.mean() / mosaic.mean()

            w = np.outer(weights(ty == 0, ty == rows - 1), weights(tx == 0, tx == cols - 1))
            y, x = 2 * ty * step, 2 * tx * step
            total[y : y + 2 * TILE, x : x + 2 * TILE] += w * mosaic
            weight[y : y + 2 * TILE, x : x + 2 * TILE] += w
    h, w = cfa.shape
    denoised = (total / np.maximum(weight, 1e-6))[seam : seam + h, seam : seam + w] * span + level
    return np.clip(denoised + 0.5, 0, white).astype(np.uint16)


def rawforge(path, folder):
    """RawForge's CFA DNG holds (raw - black) / range in camera RGB, scaled to 16 bits, on the same grid."""
    import tifffile

    out = tifffile.imread(pathlib.Path(folder) / f"{path.stem}.dng").astype(np.float32) / 65535
    with rawpy.imread(str(path)) as raw:
        assert out.shape == raw.raw_image_visible.shape, out.shape
        level = np.tile(
            np.asarray(raw.black_level_per_channel, np.float32)[raw.raw_pattern], (out.shape[0] // 2, out.shape[1] // 2)
        )
        white = float(raw.white_level)
    return np.clip(out * (white - level) + level + 0.5, 0, white).astype(np.uint16)


def his_brightness(path):
    """The bright= that holds libraw's auto brightness for this raw, so a denoised mosaic can't shift it."""
    with rawpy.imread(str(path)) as raw:
        linear = raw.postprocess(use_camera_wb=True, gamma=(1, 1), no_auto_bright=True, output_bps=16)
    clip = linear.shape[0] * linear.shape[1] * 0.01
    top = 0
    for c in range(3):
        counts = np.bincount((linear[..., c] >> 3).ravel(), minlength=0x2000)[:0x2000]
        above = np.cumsum(counts[::-1])
        top = max(top, 0x1FFF - int(np.argmax(above > clip)))
    return 0x1FFF / top


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def develop(method, mosaics):
    with rawpy.imread(str(HIS)) as raw:
        raw.raw_image_visible[:] = mosaics[HIS]
        his = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=his_brightness(HIS))
    for crop, (y, x) in HIS_CROPS.items():
        write(DATA / "out" / "frontier" / f"R0006362-{crop}_{method}.png", his[y : y + 1024, x : x + 1024])
    clean = night_pairs.linear(pair_raw("000100"))
    for iso in PAIR_ISOS:
        gain, match = night_pairs.gains(night_pairs.linear(pair_raw(iso)), clean)
        denoised = night_pairs.window(night_pairs.linear(pair_raw(iso), mosaics[pair_raw(iso)]), *PAIR_CENTRE)
        night_pairs.srgb(denoised, gain * match).save(DATA / "out" / "pairs" / f"gr3x-{iso}-labels_{method}.png")


if __name__ == "__main__":
    method, source = sys.argv[1:]
    run = {"rawnind": rawnind, "rawforge": rawforge}[method]
    (DATA / "out" / "pairs").mkdir(parents=True, exist_ok=True)
    develop(method, {path: run(path, source) for path in raws()})
