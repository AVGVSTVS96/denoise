"""Score denoisers on real noisy photos that have a clean frame of the same scene.

    uv run evaluate.py nind app nafnet-w32 scunet-psnr      run methods on a set, cache outputs
    uv run evaluate.py nind --report                         table of every cached method

Sets: `nind` (tripod series, ISO 12,800 and up, fetch_nind.py), `polyu` (camera JPEGs,
ISO 1,600-6,400) and `sidd` (phone crops). Every image is one tile of at most 512 px, the size
the app feeds its model.
PSNR and SSIM are against the clean frame on 8-bit output; LPIPS (AlexNet) is the perceptual
distance, lower is better.
"""

import argparse
import collections
import json
import pathlib
import re
import sys
import time

import cv2
import numpy as np
import torch

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
MODELS = DATA / "models"
TILE = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

sys.path.insert(0, str(DATA / "src"))
sys.path.insert(0, str(DATA / "src" / "Hush" / "tools" / "models"))


def center(image):
    y, x = max(0, image.shape[0] - TILE) // 2, max(0, image.shape[1] - TILE) // 2
    return image[y : y + TILE, x : x + TILE]


def read(path):
    return center(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))


def nind():
    for noisy in sorted((DATA / "nind").glob("*_iso*.png")):
        name, iso = re.match(r"(.+)_iso(\d+)", noisy.stem).groups()
        yield noisy.stem, int(iso), read(noisy), read(noisy.with_name(f"{name}_clean.png"))


def polyu():
    for noisy in sorted((DATA / "src" / "polyu" / "CroppedImages").glob("*_real.JPG")):
        iso = int(noisy.stem.split("_")[3])
        yield noisy.stem, iso, read(noisy), read(noisy.with_name(noisy.name.replace("_real", "_mean")))


def sidd():
    """Every eighth SIDD validation crop: phone noise, what the published weights were trained on."""
    import sidd as source

    for key, noisy, clean in source.pairs(every=8):
        yield key, 0, noisy, clean


SETS = {"nind": nind, "polyu": polyu, "sidd": sidd}


def to_tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)[None]


def to_image(tensor):
    return (tensor[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()


def torch_method(model):
    model.eval().to(DEVICE)

    def run(image):
        with torch.no_grad():
            return to_image(model(to_tensor(image).to(DEVICE)).cpu())

    return run


def onnx_method(path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return lambda image: to_image(
        torch.from_numpy(session.run(None, {session.get_inputs()[0].name: to_tensor(image).numpy()})[0])
    )


def nafnet(width):
    from nafnet import load_nafnet

    return load_nafnet(f"nafnet-sidd-w{width}", str(MODELS / f"NAFNet-SIDD-width{width}.pth"))


def scunet(kind):
    from network_scunet import SCUNet

    model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(MODELS / f"scunet_color_real_{kind}.pth", map_location="cpu"))
    return model


def restormer():
    from restormer_arch import Restormer

    model = Restormer(LayerNorm_type="BiasFree")
    model.load_state_dict(torch.load(MODELS / "restormer_real_denoising.pth")["params"])
    return model


def bm3d(image, clean):
    """BM3D told the true noise level, measured against the clean frame: its best case."""
    import bm3d as lib

    sigma = float(np.std(image.astype(np.float32) - clean)) / 255
    return (np.clip(lib.bm3d_rgb(image / 255, sigma), 0, 1) * 255 + 0.5).astype(np.uint8)


def drunet(image, clean):
    """DRUNet told the true noise level, capped at the 50 it was trained up to: its best case."""
    global _drunet
    if _drunet is None:
        from network_unet import UNetRes

        _drunet = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R", bias=False)
        _drunet.load_state_dict(torch.load(MODELS / "drunet_color.pth"))
        _drunet.eval().to(DEVICE)
    sigma = min(50.0, float(np.std(image.astype(np.float32) - clean))) / 255
    x = to_tensor(image).to(DEVICE)
    with torch.no_grad():
        return to_image(_drunet(torch.cat([x, torch.full_like(x[:, :1], sigma)], 1)).cpu())


_drunet = None
ORACLE = {"bm3d": bm3d, "drunet": drunet}


METHODS = {
    "noisy": lambda: lambda image: image,
    "app": lambda: onnx_method(MODELS / "nafnet-w32-fp16.onnx"),
    "nafnet-w32": lambda: torch_method(nafnet(32)),
    "nafnet-w64": lambda: torch_method(nafnet(64)),
    "scunet-psnr": lambda: torch_method(scunet("psnr")),
    "scunet-gan": lambda: torch_method(scunet("gan")),
    "restormer": lambda: torch_method(restormer()),
    "bm3d": None,
    "drunet": None,
    "tuned": lambda: onnx_method(MODELS / "nafnet-w32-hiso-fp16.onnx"),
}


def run(set_name, method_name, min_iso, every):
    out = DATA / "out" / set_name / method_name
    out.mkdir(parents=True, exist_ok=True)
    method = None
    for i, (key, iso, noisy, clean) in enumerate(SETS[set_name]()):
        path = out / f"{key}.png"
        if path.exists() or iso < min_iso or i % every:
            continue
        start = time.time()
        if method_name in ORACLE:
            result = ORACLE[method_name](noisy, clean)
        else:
            method = method or METHODS[method_name]()
            result = method(noisy)
        cv2.imwrite(str(path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"{method_name} {key} {time.time() - start:.1f}s", flush=True)


def bucket(iso):
    return next(
        label
        for limit, label in ((6400, "≤6400"), (12800, "12800"), (25600, "25600"), (10**9, "51200+"))
        if iso <= limit
    )


def report(set_name):
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    perceptual = lpips.LPIPS(net="alex", verbose=False)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    methods = sorted(p.name for p in (DATA / "out" / set_name).iterdir())
    for key, iso, _, clean in SETS[set_name]():
        outputs = {m: DATA / "out" / set_name / m / f"{key}.png" for m in methods}
        for m, path in outputs.items():
            if not path.exists():
                continue
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                distance = perceptual(to_tensor(image) * 2 - 1, to_tensor(clean) * 2 - 1).item()
            scores = (
                peak_signal_noise_ratio(clean, image),
                structural_similarity(clean, image, channel_axis=-1),
                distance,
            )
            for group in (bucket(iso), "all"):
                rows[group][m].append(scores)
    table = {
        group: {m: [len(s), *np.mean(s, axis=0).round(4).tolist()] for m, s in by_method.items()}
        for group, by_method in rows.items()
    }
    (DATA / "out" / f"{set_name}.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    for group in sorted(table):
        print(f"\n{set_name} ISO {group}\n{'method':14}{'n':>4}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
        for m, (n, psnr, ssim, dist) in sorted(table[group].items(), key=lambda kv: kv[1][3]):
            print(f"{m:14}{n:4}{psnr:8.2f}{ssim:8.3f}{dist:8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("set", choices=SETS)
    parser.add_argument("methods", nargs="*")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--min-iso", type=int, default=0)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for name in args.methods:
        run(args.set, name, args.min_iso, args.every)
    if args.report:
        report(args.set)
