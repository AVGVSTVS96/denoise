"""Fine-tune NAFNet (SIDD, width 32) on extreme-ISO pairs from fetch_rawnind.py.

    uv run train.py --steps 6000

Starts from the published SIDD weights, so what the model already does on phone noise is the
starting point. A third of the noisy patches are JPEG-compressed first, since photos reach the
app as JPEG. Every `--check` steps it scores the held-out NIND scenes and saves
data/models/finetune/step<N>.pth.
"""

import argparse
import collections
import random
import re

import cv2
import numpy as np
import torch

import evaluate

TRAIN = evaluate.DATA / "train"
JPEG_SHARE = 1 / 3
OUT = evaluate.MODELS / "finetune"
PATCH = 256


def pairs():
    by_crop = collections.defaultdict(list)
    for noisy in sorted(TRAIN.glob("*_iso*.png")):
        by_crop[re.sub(r"_iso\d+$", "", noisy.stem)].append(noisy)
    return [(TRAIN / f"{crop}_clean.png", frames) for crop, frames in by_crop.items()]


def batch(loaded, size, rng):
    """Random 256 px patches, flipped and turned, one ISO picked per crop."""
    noisy, clean = [], []
    for _ in range(size):
        target, frames = rng.choice(loaded)
        source = rng.choice(frames)
        y, x = rng.randrange(evaluate.TILE - PATCH + 1), rng.randrange(evaluate.TILE - PATCH + 1)
        pair = np.stack([source[y : y + PATCH, x : x + PATCH], target[y : y + PATCH, x : x + PATCH]])
        if rng.random() < 0.5:
            pair = pair[:, :, ::-1]
        pair = np.rot90(pair, rng.randrange(4), axes=(1, 2))
        if rng.random() < JPEG_SHARE:
            _, encoded = cv2.imencode(
                ".jpg", np.ascontiguousarray(pair[0]), [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(70, 96)]
            )
            pair = np.stack([cv2.imdecode(encoded, cv2.IMREAD_COLOR), pair[1]])
        noisy.append(pair[0])
        clean.append(pair[1])
    to = lambda images: (
        torch.from_numpy(np.ascontiguousarray(np.stack(images)))
        .permute(0, 3, 1, 2)
        .float()
        .div(255)
        .to(evaluate.DEVICE)
    )
    return to(noisy), to(clean)


def psnr(model, held_out):
    model.eval()
    scores = []
    with torch.no_grad():
        for noisy, clean in held_out:
            out = evaluate.to_image(model(evaluate.to_tensor(noisy).to(evaluate.DEVICE)).cpu())
            scores.append(cv2.PSNR(out, clean))
    model.train()
    return float(np.mean(scores))


def main(args):
    rng = random.Random(0)
    torch.manual_seed(0)
    read = lambda path: cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    loaded = [(read(target), [read(f) for f in frames]) for target, frames in pairs()]
    held_out = [(n, c) for _, _, n, c in evaluate.nind()]
    model = evaluate.nafnet(32).to(evaluate.DEVICE).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr / 100)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(loaded)} crops; step 0 nind {psnr(model, held_out):.2f}", flush=True)

    running = 0.0
    for step in range(1, args.steps + 1):
        noisy, clean = batch(loaded, args.batch, rng)
        loss = torch.nn.functional.l1_loss(model(noisy), clean)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        schedule.step()
        running += loss.item()
        if step % args.check == 0:
            print(f"step {step} loss {running / args.check:.4f} nind {psnr(model, held_out):.2f}", flush=True)
            running = 0.0
            torch.save(model.state_dict(), OUT / f"step{step}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--check", type=int, default=500)
    main(parser.parse_args())
