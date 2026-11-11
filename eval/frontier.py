"""Run hosted image models on the NIND test shots through OpenRouter and score them against ours.

    OPENROUTER_API_KEY=... uv run frontier.py run <model> <key> ...   cache outputs in data/out/nind/<slug>/
    uv run frontier.py report <key> ...                              table on exactly those shots
    uv run frontier.py photo <model> <png> [1K|2K|4K]                 a real photo in one call
    uv run frontier.py tiled <model> <png>                           a real photo at full size, in 1024 px tiles

Each output is resized to the 512 px input and warped onto it (ECC affine, fitted to the noisy input,
never to the clean frame), so a model that shifts or crops the picture isn't scored as noise.
"""

import base64
import json
import os
import pathlib
import sys
import time
import urllib.request

import cv2
import numpy as np

import evaluate

PROMPT = (
    "This is a real photo shot at very high ISO. Remove the sensor noise (grain and color speckles) only. "
    "Keep it the same photo: same framing, geometry, colors, exposure and every real edge and texture. "
    "Do not add, remove, sharpen into, or redraw any detail that is not visibly there: where text or texture is "
    "too noisy to read, leave it soft and unreadable rather than guessing it. Return the cleaned photo."
)
COSTS = evaluate.DATA / "out" / "frontier" / "costs.json"


def slug(model):
    return model.split("/")[-1]


def call(model, image, extra=None):
    big = image.shape[0] * image.shape[1] > 4e6
    _, png = cv2.imencode(
        ".jpg" if big else ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    body = {
        "model": model,
        "modalities": ["image", "text"],
        "usage": {"include": True},
        **(extra or {}),
        **({"reasoning": {"effort": "minimal"}} if model.startswith("openai/gpt-5.4") else {}),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{'jpeg' if big else 'png'};base64," + base64.b64encode(png).decode()
                        },
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        json.dumps(body).encode(),
        {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"},
    )
    reply = json.load(urllib.request.urlopen(request, timeout=300))
    images = reply.get("choices", [{}])[0].get("message", {}).get("images")
    if not images:
        raise RuntimeError(json.dumps(reply)[:800])
    url = images[0]["image_url"]["url"]
    data = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), reply.get("usage", {}).get("cost")


def align(output, noisy):
    """Resize to the input and warp onto it; returns the image and the raw output size."""
    size = output.shape[1], output.shape[0]
    output = cv2.resize(output, (noisy.shape[1], noisy.shape[0]), interpolation=cv2.INTER_AREA)
    gray = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (0, 0), 2).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        _, warp = cv2.findTransformECC(gray(noisy), gray(output), warp, cv2.MOTION_AFFINE, criteria, None, 5)
        output = cv2.warpAffine(
            output,
            warp,
            output.shape[1::-1],
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error:
        pass
    return output, size, warp.round(3).tolist()


def run(model, keys):
    out = evaluate.DATA / "out" / "nind" / slug(model)
    out.mkdir(parents=True, exist_ok=True)
    costs = json.loads(COSTS.read_text()) if COSTS.exists() else {}
    shots = {k: (iso, n) for k, iso, n, _ in evaluate.nind() if k in keys}
    for key in keys:
        path = out / f"{key}.png"
        if path.exists():
            continue
        start = time.time()
        raw, cost = call(model, shots[key][1])
        cv2.imwrite(
            str(out.parent.parent / "frontier" / f"{slug(model)}_{key}_raw.png"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        )
        image, size, warp = align(raw, shots[key][1])
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        costs.setdefault(slug(model), {})[key] = {
            "cost": cost,
            "size": size,
            "warp": warp,
            "seconds": round(time.time() - start),
        }
        COSTS.write_text(json.dumps(costs, indent=1))
        print(slug(model), key, size, f"${cost}", f"{time.time() - start:.0f}s", flush=True)


def report(keys):
    import lpips
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    perceptual = lpips.LPIPS(net="alex", verbose=False)
    root = evaluate.DATA / "out" / "nind"
    rows = {}
    for key, iso, _, clean in evaluate.nind():
        if key not in keys:
            continue
        for method in sorted(p.name for p in root.iterdir()):
            path = root / method / f"{key}.png"
            if not path.exists():
                continue
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                distance = perceptual(evaluate.to_tensor(image) * 2 - 1, evaluate.to_tensor(clean) * 2 - 1).item()
            rows.setdefault(method, []).append(
                (peak_signal_noise_ratio(clean, image), structural_similarity(clean, image, channel_axis=-1), distance)
            )
    print(f"{'method':24}{'n':>4}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}")
    for method, scores in sorted(rows.items(), key=lambda kv: -np.mean(kv[1], axis=0)[0]):
        if len(scores) == len(keys):
            psnr, ssim, dist = np.mean(scores, axis=0)
            print(f"{method:24}{len(scores):4}{psnr:8.2f}{ssim:8.3f}{dist:8.3f}")


def photo(model, path, size="2K"):
    """One real photo with no clean frame, kept at up to 2K, judged by eye."""
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    ratio = min(
        ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"], key=lambda r: abs(eval(r.replace(":", "/")) - w / h)
    )
    start = time.time()
    raw, cost = call(model, image, {"image_config": {"aspect_ratio": ratio, "image_size": size}})
    out = evaluate.DATA / "out" / "frontier" / f"{pathlib.Path(path).stem}_{slug(model)}.png"
    cv2.imwrite(str(out), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
    print(slug(model), raw.shape[1::-1], f"${cost}", f"{time.time() - start:.0f}s", out, flush=True)


def match_tone(result, image, sigma=24):
    """Keep the model's fine detail but the photo's own tone, so tiles the model lit differently don't show."""
    blur = lambda im: cv2.GaussianBlur(im.astype(np.float32), (0, 0), sigma)
    return (result + blur(image) - blur(result) + 0.5).clip(0, 255).astype(np.uint8)


def tiled(model, path, tile=1024, overlap=128):
    """A full-size photo, one call per overlapping tile, each warped onto its input and feathered together."""
    from concurrent.futures import ThreadPoolExecutor

    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    step = tile - overlap
    starts = lambda n: sorted({min(i, n - tile) for i in range(0, n - overlap, step)})
    boxes = [(y, x) for y in starts(h) for x in starts(w)]
    ramp = np.minimum(1, (np.arange(tile) + 0.5) / overlap)
    weight = np.minimum(ramp, ramp[::-1])
    weight = np.outer(weight, weight)[..., None].astype(np.float32)

    def one(box):
        y, x = box
        crop = image[y : y + tile, x : x + tile]
        for attempt in range(3):
            try:
                raw, cost = call(model, crop, {"image_config": {"aspect_ratio": "1:1", "image_size": "1K"}})
                return box, align(raw, crop)[0], cost or 0
            except (OSError, RuntimeError, KeyError) as error:
                print("retry", box, str(error)[:120], flush=True)
        return box, crop, 0

    total = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w, 1), np.float32)
    spent = 0
    with ThreadPoolExecutor(6) as pool:
        for (y, x), out, cost in pool.map(one, boxes):
            total[y : y + tile, x : x + tile] += out * weight
            weights[y : y + tile, x : x + tile] += weight
            spent += cost
    result = match_tone(total / weights, image)
    out = evaluate.DATA / "out" / "frontier" / f"{pathlib.Path(path).stem}_{slug(model)}-tiled.png"
    cv2.imwrite(str(out), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(slug(model), len(boxes), "tiles", f"${spent:.2f}", out, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "tone":
        image, result = (cv2.imread(p).astype(np.float32) for p in sys.argv[2:4])
        cv2.imwrite(sys.argv[4], match_tone(result, image))
    elif sys.argv[1] == "tiled":
        tiled(*sys.argv[2:])
    elif sys.argv[1] == "photo":
        photo(*sys.argv[2:])
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3:])
    else:
        report(sys.argv[2:])
