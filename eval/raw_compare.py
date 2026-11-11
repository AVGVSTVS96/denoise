"""Denoise the camera raw, or the developed picture? Scores both on RawNIND's held-out scenes.

    uv run raw_compare.py run        fetch raws, develop, run every method, cache crops
    uv run raw_compare.py report     table and before/after strips

Test frames are the Bayer scenes RawNIND's authors held out of training (rawnind_jddc
config/test_reserve.yaml), at ISO 6,400 and up, one frame per scene and ISO. The clean base-ISO
frame and the noisy frame are developed the same way as fetch_rawnind.py: rawpy AHD, the clean
frame's white balance, linear sRGB, one exposure per scene, the noisy frame's brightness matched
to the clean one over bright areas, and its whole-pixel shift found and undone.

The developed-picture models get that noisy frame as 8-bit sRGB. The raw model (RawNIND UtNet2
Bayer, from darktable-ai's rawdenoise-nind package) gets the noisy frame's sensor data and
returns camera RGB at an arbitrary gain; that goes through the same steps rawpy takes (white
balance, clip, the camera's colour matrix to linear sRGB), is scaled to the noisy frame's own
developed brightness, then gets the noisy frame's gain and exposure. Every method is scored on the
same 512 px crops against the clean frame.
"""

import collections
import json
import pathlib
import re
import sys
import urllib.request

import cv2
import numpy as np
import rawpy

ROOT = pathlib.Path(__file__).parent
WORK = ROOT / "data" / "raw-test"
RAWS = WORK / "raw"
OUT = WORK / "out"
RAW_MODEL = WORK / "weights" / "rawdenoise-nind" / "model_bayer.onnx"
HOST = "https://dataverse.uclouvain.be/api"
DOI = "doi:10.14428/DVN/DEQCIM"
MIN_ISO = 6400
CROP = 512
CROPS = 3
#: Context each developed-picture model sees around a crop, so its tile edges aren't scored.
MARGIN = 64
#: The raw model reads a fixed 512 x 512 packed tile, 1,024 px of sensor.
RAW_TILE = 1024
MAX_SHIFT = 16
BANDS = ((10000, "6400-10000"), (20000, "12800-20000"), (40000, "25600-40000"), (10**9, "51200+"))
XYZ_TO_SRGB = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
METHODS = ("noisy", "app", "scunet-psnr", "restormer", "raw-model")


def frames():
    """{scene: {"clean": (id, name), "noisy": {iso: (id, name)}}} for the held-out Bayer scenes."""
    listing = json.load(urllib.request.urlopen(f"{HOST}/datasets/:persistentId/?persistentId={DOI}"))
    found = {}
    for entry in listing["data"]["latestVersion"]["files"]:
        file = entry["dataFile"]
        m = re.match(r"Bayer_(?:UNK_)?TEST_(.+?)_(GT_)?ISO(\d+)_sha1=\w+\.(\w+)$", file["filename"])
        if not m or m.group(4) == "xmp":
            continue
        scene = found.setdefault(m.group(1), {"clean": None, "noisy": {}})
        source = (file["id"], file["filename"])
        if m.group(2):
            scene["clean"] = scene["clean"] or source
        elif int(m.group(3)) >= MIN_ISO:
            scene["noisy"].setdefault(int(m.group(3)), source)
    return {name: s for name, s in sorted(found.items()) if s["clean"] and s["noisy"]}


def fetch(source):
    file_id, name = source
    path = RAWS / f"{file_id}{pathlib.Path(name).suffix}"
    if not path.exists():
        RAWS.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".part").write_bytes(urllib.request.urlopen(f"{HOST}/access/datafile/{file_id}").read())
        path.with_suffix(".part").rename(path)
    return path


def develop(path, white_balance=None):
    """Linear sRGB in [0, 1] in sensor orientation, and the white balance used."""
    with rawpy.imread(str(path)) as raw:
        white_balance = white_balance or list(raw.camera_whitebalance)
        image = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, user_wb=white_balance, user_flip=0)
    return image.astype(np.float32) / 65535, white_balance


def srgb(linear):
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return (encoded * 255 + 0.5).astype(np.uint8)


def shift_between(clean, noisy):
    small = [cv2.resize(cv2.GaussianBlur(i.mean(axis=2), (0, 0), 6), None, fx=0.25, fy=0.25) for i in (clean, noisy)]
    (dx, dy), _ = cv2.phaseCorrelate(small[0], small[1])
    return round(dx * 4), round(dy * 4)


def corners(clean, rng):
    """The two most detailed 512 px crops and one at random, far enough in for every method's context."""
    h, w = clean.shape[:2]
    edge = MARGIN + MAX_SHIFT
    grid = [(y, x) for y in range(edge, h - CROP - edge, CROP) for x in range(edge, w - CROP - edge, CROP)]
    gray = clean.mean(axis=2)
    detail = [float(np.abs(cv2.Laplacian(gray[y : y + CROP, x : x + CROP], cv2.CV_32F)).mean()) for y, x in grid]
    ranked = [grid[i] for i in np.argsort(detail)[::-1]]
    return ranked[: CROPS - 1] + [ranked[CROPS - 1 :][rng.integers(len(ranked) - CROPS + 1)]]


def bright_mask(clean):
    luminance = clean.mean(axis=2)
    return (luminance > np.percentile(luminance, 60)) & (luminance < np.percentile(luminance, 99))


class RawModel:
    """RawNIND UtNet2: packed RGGB in, demosaiced camera RGB out at twice the size and a learned gain."""

    def __init__(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(RAW_MODEL), providers=["CPUExecutionProvider"])

    def __call__(self, path, y, x, white_balance):
        """Linear sRGB for the CROP x CROP region at (y, x), rawpy's colour pipeline after the model."""
        with rawpy.imread(str(path)) as raw:
            mosaic = raw.raw_image_visible.astype(np.float32)
            pattern = raw.raw_pattern
            colors = raw.color_desc.decode()
            black = np.asarray(raw.black_level_per_channel, np.float32)
            white = float(raw.white_level)
            # dcraw's camera-to-sRGB matrix, which libraw only fills in after its own postprocess.
            cam_to_srgb = np.asarray(raw.rgb_xyz_matrix)[:3] @ XYZ_TO_SRGB
        cam_to_srgb = np.linalg.inv(cam_to_srgb / cam_to_srgb.sum(axis=1, keepdims=True)).astype(np.float32)
        h, w = mosaic.shape
        # The model assumes RGGB, so the tile starts on a red photosite.
        ry, rx = next((i, j) for i in range(2) for j in range(2) if colors[pattern[i, j]] == "R")
        top = min(max(y + CROP // 2 - RAW_TILE // 2, 0), h - RAW_TILE - 1)
        left = min(max(x + CROP // 2 - RAW_TILE // 2, 0), w - RAW_TILE - 1)
        top += (ry - top) % 2
        left += (rx - left) % 2
        tile = mosaic[top : top + RAW_TILE, left : left + RAW_TILE]
        for i in range(2):
            for j in range(2):
                level = black[pattern[(top + i) % 2, (left + j) % 2]]
                tile[i::2, j::2] = np.clip((tile[i::2, j::2] - level) / (white - level), 0, 1)
        packed = np.stack([tile[0::2, 0::2], tile[0::2, 1::2], tile[1::2, 0::2], tile[1::2, 1::2]])[None]
        cam = self.session.run(None, {self.session.get_inputs()[0].name: packed})[0][0].transpose(1, 2, 0)
        cam = cam[y - top : y - top + CROP, x - left : x - left + CROP] * (packed.mean() / cam.mean())
        # rawpy's own order: white balance scaled so the weakest channel is 1, clip at white, then colour.
        multipliers = np.asarray(white_balance[:3], np.float32)
        cam = np.clip(cam * multipliers / multipliers.min(), 0, 1)
        return cam @ cam_to_srgb.T


def developed_methods():
    import evaluate

    return {
        "app": evaluate.METHODS["app"](),
        "scunet-psnr": evaluate.torch_method(evaluate.scunet("psnr")),
        "restormer": evaluate.torch_method(evaluate.restormer()),
    }


def save(method, key, image):
    (OUT / method).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / method / f"{key}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def run():
    raw_model = RawModel()
    models = developed_methods()
    rng = np.random.default_rng(0)
    log = []
    for scene, source in frames().items():
        clean_path = fetch(source["clean"])
        clean, white_balance = develop(clean_path)
        exposure = 0.9 / max(float(np.percentile(clean, 99.5)), 1e-3)
        bright = bright_mask(clean)
        picks = corners(clean, rng)
        for n, (y, x) in enumerate(picks):
            save("clean", f"{scene}_{n}", srgb(clean[y : y + CROP, x : x + CROP] * exposure))
            # Calibration: the raw model on the clean raw shows what the pipeline itself costs.
            calibration = raw_model(clean_path, y, x, white_balance)
            reference = clean[y : y + CROP, x : x + CROP]
            mask = bright_mask(reference)
            save(
                "raw-model-on-clean",
                f"{scene}_{n}",
                srgb(calibration * reference[mask].mean() / calibration[mask].mean() * exposure),
            )
        for iso, noisy_source in sorted(source["noisy"].items()):
            path = fetch(noisy_source)
            noisy, _ = develop(path, white_balance)
            if noisy.shape != clean.shape:
                log.append((scene, iso, "size differs, dropped"))
                continue
            dx, dy = shift_between(clean, noisy)
            if max(abs(dx), abs(dy)) > MAX_SHIFT:
                log.append((scene, iso, f"shift {dx},{dy}, dropped"))
                continue
            gain = float(clean[bright].mean() / noisy[bright].mean())
            for n, (y, x) in enumerate(picks):
                key = f"{scene}_iso{iso}_{n}"
                if (OUT / "raw-model" / f"{key}.png").exists():
                    continue
                ny, nx = y + dy, x + dx
                context = srgb(
                    noisy[ny - MARGIN : ny + CROP + MARGIN, nx - MARGIN : nx + CROP + MARGIN] * gain * exposure
                )
                save("noisy", key, context[MARGIN:-MARGIN, MARGIN:-MARGIN])
                for name, model in models.items():
                    save(name, key, model(context)[MARGIN:-MARGIN, MARGIN:-MARGIN])
                developed = noisy[ny : ny + CROP, nx : nx + CROP]
                mask = bright_mask(clean[y : y + CROP, x : x + CROP])
                denoised = raw_model(path, ny, nx, white_balance)
                denoised *= developed[mask].mean() / denoised[mask].mean()
                save("raw-model", key, srgb(denoised * gain * exposure))
                print(key, f"shift {dx},{dy} gain {gain:.2f}", flush=True)
            log.append((scene, iso, f"shift {dx},{dy} gain {gain:.2f}"))
    (WORK / "run.json").write_text(json.dumps(log, indent=1))


def read(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def report():
    import lpips
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    perceptual = lpips.LPIPS(net="alex", verbose=False)

    def tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]

    def scores(image, clean):
        with torch.no_grad():
            distance = perceptual(tensor(image), tensor(clean)).item()
        return peak_signal_noise_ratio(clean, image), structural_similarity(clean, image, channel_axis=-1), distance

    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted((OUT / "raw-model").glob("*.png")):
        scene, iso, n = re.match(r"(.+)_iso(\d+)_(\d+)", path.stem).groups()
        clean = read(OUT / "clean" / f"{scene}_{n}.png")
        band = next(label for limit, label in BANDS if int(iso) <= limit)
        # The other side of the reference question: the raw model against the clean raw it developed itself.
        own_clean = read(OUT / "raw-model-on-clean" / f"{scene}_{n}.png")
        results = {m: scores(read(OUT / m / path.name), clean) for m in METHODS}
        results["raw-model vs own"] = scores(read(path), own_clean)
        for method, result in results.items():
            for group in (band, "12800+" if int(iso) >= 12800 else None, "all"):
                if group:
                    rows[group][method].append(result)
    for path in sorted((OUT / "raw-model-on-clean").glob("*.png")):
        rows["calibration"]["raw-model-on-clean"].append(scores(read(path), read(OUT / "clean" / path.name)))
    table = {
        group: {m: [len(s), *np.mean(s, axis=0).round(4).tolist()] for m, s in by_method.items()}
        for group, by_method in rows.items()
    }
    (WORK / "report.json").write_text(json.dumps(table, indent=1))
    order = [label for _, label in BANDS] + ["12800+", "all", "calibration"]
    for group in [g for g in order if g in table]:
        print(f"\nISO {group}\n{'method':20}{'n':>4}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
        for m, (n, psnr, ssim, dist) in table[group].items():
            print(f"{m:20}{n:4}{psnr:8.2f}{ssim:8.3f}{dist:8.3f}")
    strips()


def strips(count=3, size=320):
    """noisy | app | Restormer | raw model | clean, from the highest-ISO frame of different scenes."""
    shown = ("noisy", "app", "restormer", "raw-model", "clean")
    best = {}
    for path in (OUT / "raw-model").glob("*.png"):
        scene, iso, n = re.match(r"(.+)_iso(\d+)_(\d+)", path.stem).groups()
        if n == "0" and int(iso) > best.get(scene, (0,))[0]:
            best[scene] = (int(iso), path.stem)
    picks = sorted(best.items(), key=lambda kv: -kv[1][0])[:count]
    (WORK / "strips").mkdir(exist_ok=True)
    for scene, (iso, key) in picks:
        o = (CROP - size) // 2
        panels = []
        for method in shown:
            name = f"{scene}_0.png" if method == "clean" else f"{key}.png"
            panel = cv2.imread(str(OUT / method / name))[o : o + size, o : o + size].copy()
            label = f"{method} ISO {iso}" if method == "noisy" else method
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        out = WORK / "strips" / f"{key}.jpg"
        cv2.imwrite(str(out), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(out)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
