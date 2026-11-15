"""Train the faithful model: NAFNet (width 32) on raw Bayer mosaics, noisy in, clean mosaic out.

    uv run train_raw.py                                   from scratch, or resumes data/models/raw/last.pth
    uv run train_raw.py --out gr3x --init data/models/raw/faithful.pth --scenes 'gr3x-*' \
        --profile data/noise/gr3x.json --steps 20000 --lr 5e-5      his camera's fine-tune, once it has pairs

The network sees the mosaic packed into four half-size planes (R, G, G, B), so a phone runs it at a
quarter of the pixels. Its body starts from the SIDD sRGB weights; the first and last layers are new.
Batches come from raw_pairs.py's crops: real noisy frames, and clean frames with made-up sensor noise
(shot noise, read noise and row banding). Without --profile that noise is drawn from a wide range of
cameras; with one, from a camera's measured levels (calibrate_raw.py). Each crop is scaled to a set
brightness first, the same way denoise() scales a photo, so dark night frames look like any other.
"""

import argparse
import fnmatch
import json
import math
import pathlib
import random
import time

import numpy as np
import torch

import evaluate

PAIRS = evaluate.DATA / "rawpairs"
MODELS = evaluate.MODELS / "raw"
#: Mean a crop or photo is scaled to before the network sees it.
LEVEL = 0.1
MAX_GAIN = 64.0
SYNTHETIC_SHARE = 0.4
#: Imaging Resource's Ricoh GR II and GR III lab scenes, the nearest thing to his camera's own noise.
RICOH_WEIGHT = 4


def network():
    from nafnet import CONFIGS, NAFNet

    return NAFNet(CONFIGS["nafnet-sidd-w32"], img_channel=4)


def pack(mosaic):
    return torch.stack(
        [mosaic[..., 0::2, 0::2], mosaic[..., 0::2, 1::2], mosaic[..., 1::2, 0::2], mosaic[..., 1::2, 1::2]], -3
    )


def unpack(planes):
    n, _, h, w = planes.shape
    out = planes.new_empty(n, 2 * h, 2 * w)
    out[:, 0::2, 0::2], out[:, 0::2, 1::2], out[:, 1::2, 0::2], out[:, 1::2, 1::2] = planes.unbind(1)
    return out


def tone(x):
    """Roughly how a developed picture spaces brightness, so shadows count as much as highlights."""
    return (x.clamp(min=0) + 1e-3) ** (1 / 2.2)


@torch.no_grad()
def denoise(model, mosaic, tile=1024, overlap=64):
    """A whole normalised RGGB mosaic (H, W float32 numpy) through the model, in overlapping tiles.

    Every tile keeps only its middle; the last row and column of tiles sit flush with the far edge.
    """
    gain = float(np.clip(LEVEL / max(float(mosaic.mean()), 1e-6), 1, MAX_GAIN))
    padded = np.pad(mosaic * gain, overlap, "reflect")
    out = np.zeros_like(padded)
    starts = lambda side: sorted({*range(0, side - tile, tile - 2 * overlap), side - tile})
    for y in starts(padded.shape[0]):
        for x in starts(padded.shape[1]):
            piece = torch.from_numpy(np.ascontiguousarray(padded[y : y + tile, x : x + tile])).cuda()[None]
            with torch.autocast("cuda", torch.bfloat16):
                result = unpack(model(pack(piece)).float())[0].cpu().numpy()
            core = np.s_[overlap:-overlap, overlap:-overlap]
            out[y : y + tile, x : x + tile][core] = result[core]
    return out[overlap:-overlap, overlap:-overlap] / gain


class Crops(torch.utils.data.IterableDataset):
    """Endless (noisy-or-None, clean) mosaics, flipped and turned with the colours kept on RGGB."""

    def __init__(self, scenes, size, seed):
        self.scenes, self.size, self.seed = scenes, size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0))
        arrays = {}
        weights = [RICOH_WEIGHT if s.startswith("ir-") else 1 for s in self.scenes]
        while True:
            scene = rng.choices(self.scenes, weights)[0]
            stack = arrays.get(scene)
            if stack is None:
                stack = arrays[scene] = np.load(PAIRS / f"{scene}.npy", mmap_mode="r")
            frames, crops, side, _ = stack.shape
            crop = rng.randrange(crops)
            synthetic = frames == 1 or rng.random() < SYNTHETIC_SHARE
            frame = 0 if synthetic else rng.randrange(1, frames)
            flip_y, flip_x, turn = rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5
            # A flipped mosaic starts on red again one pixel in, so the window's corner takes that parity.
            y = rng.randrange((side - self.size) // 2) * 2 + flip_y
            x = rng.randrange((side - self.size) // 2) * 2 + flip_x
            pair = np.stack([stack[0, crop], stack[frame, crop]]).astype(np.float32)
            if flip_y:
                pair = pair[:, ::-1]
            if flip_x:
                pair = pair[:, :, ::-1]
            pair = pair[:, y : y + self.size, x : x + self.size]
            if turn:
                pair = pair.transpose(0, 2, 1)
            yield torch.from_numpy(np.ascontiguousarray(pair)), synthetic


def made_up_noise(clean, profile):
    """Shot, read and row noise on clean mosaics (N, H, W) already at LEVEL brightness.

    A profile holds a camera's measured levels per ISO in normalised raw units; the photo is taken as
    shot 1-16x darker than LEVEL and brought up, as a dim night frame is.
    """
    n, device = clean.shape[0], clean.device
    uniform = lambda low, high: torch.exp(torch.empty(n, device=device).uniform_(math.log(low), math.log(high)))
    if profile:
        levels = [random.choice(list(profile["iso"].values())) for _ in range(n)]
        dim = uniform(1, 16)
        shot = torch.tensor([lv["shot"] for lv in levels], device=device) * dim
        read = torch.tensor([lv["read"] for lv in levels], device=device) * dim
        row = torch.tensor([lv.get("row", 0.0) for lv in levels], device=device) * dim
    else:
        # From 3,000 down to 10 electrons at LEVEL, with half to four electrons of read noise.
        shot = uniform(LEVEL / 3000, LEVEL / 10)
        read = shot * uniform(0.5, 4)
        row = read * torch.empty(n, device=device).uniform_(0, 0.3)
    view = lambda t: t.view(n, 1, 1)
    photons = torch.poisson(clean.clamp(min=0) / view(shot)) * view(shot)
    return photons + torch.randn_like(clean) * view(read) + torch.randn(n, clean.shape[1], 1, device=device) * view(row)


def gains(clean):
    """Scale each crop to LEVEL, give or take a factor of two, as denoise() scales a photo to its mean."""
    mean = clean.mean(dim=(1, 2)).clamp(min=1e-6)
    spread = torch.exp(torch.empty_like(mean).uniform_(-math.log(2), math.log(2)))
    return (LEVEL / mean * spread).clamp(1 / 4, MAX_GAIN).view(-1, 1, 1)


def scenes(split, pattern):
    found = []
    for meta in sorted(PAIRS.glob("*.json")):
        info = json.loads(meta.read_text())
        if info["split"] == split and fnmatch.fnmatch(meta.stem, pattern) and meta.with_suffix(".npy").exists():
            found.append(meta.stem)
    return found


@torch.no_grad()
def validate(model, val):
    """Mean PSNR on the toned mosaic, over every held-out real noisy frame's first four crops."""
    model.eval()
    scores = []
    for scene in val:
        stack = np.load(PAIRS / f"{scene}.npy", mmap_mode="r")
        for frame in range(1, stack.shape[0]):
            clean = torch.from_numpy(stack[0, :4, 64:448, 64:448].astype(np.float32)).cuda()
            noisy = torch.from_numpy(stack[frame, :4, 64:448, 64:448].astype(np.float32)).cuda()
            g = (LEVEL / clean.mean(dim=(1, 2)).clamp(min=1e-6)).clamp(1 / 4, MAX_GAIN).view(-1, 1, 1)
            with torch.autocast("cuda", torch.bfloat16):
                out = unpack(model(pack(noisy * g)).float())
            error = ((tone(out).clamp(0, 1) - tone(clean * g).clamp(0, 1)) ** 2).mean(dim=(1, 2))
            scores += (-10 * torch.log10(error)).tolist()
    model.train()
    return float(np.mean(scores))


def main(args):
    torch.manual_seed(args.seed)
    out = MODELS / args.out
    out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(pathlib.Path(args.profile).read_text()) if args.profile else None
    model = network().cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    else:
        sidd = torch.load(evaluate.MODELS / "NAFNet-SIDD-width32.pth", map_location="cpu")["params"]
        model.load_state_dict({k: v for k, v in sidd.items() if not k.startswith(("intro", "ending"))}, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0)
    warmup = 1000
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            min(1, (s + 1) / warmup) * (0.01 + 0.99 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) / 2)
        ),
    )
    step = 0
    if (out / "last.pth").exists():
        state = torch.load(out / "last.pth", map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step = state["step"]
    val = scenes("val", args.scenes)
    log = (out / "log.jsonl").open("a")
    running = 0.0
    while step < args.steps:
        # The scene list is read again every round, so crops raw_pairs.py finishes meanwhile join in.
        train = scenes("train", args.scenes)
        loader = torch.utils.data.DataLoader(
            Crops(train, args.size, args.seed + step), batch_size=args.batch, num_workers=args.workers, pin_memory=True
        )
        for pair, synthetic in loader:
            pair = pair.cuda(non_blocking=True)
            clean, noisy = pair[:, 0], pair[:, 1]
            synthetic = synthetic.cuda()
            g = gains(clean)
            clean, noisy = clean * g, noisy * g
            if synthetic.any():
                noisy = torch.where(synthetic.view(-1, 1, 1), made_up_noise(clean, profile), noisy)
            with torch.autocast("cuda", torch.bfloat16):
                result = unpack(model(pack(noisy)).float())
            loss = torch.nn.functional.l1_loss(tone(result), tone(clean))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            schedule.step()
            step += 1
            running += loss.item()
            if step % args.check == 0 or step == args.steps:
                score = validate(model, val) if val else float("nan")
                line = {
                    "step": step,
                    "loss": round(running / args.check, 5),
                    "val_psnr": round(score, 3),
                    "lr": schedule.get_last_lr()[0],
                    "scenes": len(train),
                    "time": round(time.time()),
                }
                print(json.dumps(line), flush=True)
                log.write(json.dumps(line) + "\n")
                log.flush()
                running = 0.0
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "step": step,
                }
                torch.save(state, out / "last.tmp")
                (out / "last.tmp").rename(out / "last.pth")
                if step % args.keep == 0:
                    torch.save(model.state_dict(), out / f"step{step}.pth")
            if step >= args.steps or step % args.rescan == 0:
                break
    torch.save(model.state_dict(), out / "faithful.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="faithful")
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=384, help="sensor pixels per side")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--check", type=int, default=2000)
    parser.add_argument("--keep", type=int, default=20000, help="steps between saved snapshots")
    parser.add_argument("--rescan", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init")
    parser.add_argument("--scenes", default="*")
    parser.add_argument("--profile")
    main(parser.parse_args())
