"""Train the faithful model: NAFNet (width 32) on raw Bayer mosaics, noisy in, clean mosaic out.

    uv run train_raw.py                                   from scratch, or resumes data/models/raw/last.pth
    uv run train_raw.py --out gr3 --init data/models/raw/faithful/faithful.pth \
        --profile data/noise/gr3.json --steps 20000 --lr 1e-4       one camera's fine-tune

The network sees the mosaic packed into four half-size planes (R, G, G, B), so a phone runs it at a
quarter of the pixels. Its body starts from the SIDD sRGB weights; the first and last layers are new.
--width 64 trains the wider NAFNet, four times the work for a picture, for when quality matters more.
Batches come from raw_pairs.py's crops: real noisy frames, each against the clean twin resampled onto
it, and clean frames with noise laid over them.
A third of the draws go to the scenes that have real noisy frames, however many clean ones there are.
Without --profile that noise is either made up (shot noise, read noise and row banding, from a wide
range of cameras) or a window of the Ricoh GR III's real noise (noise_bank.py). A profile narrows the
run to one camera: every clean frame gets that camera's noise, its real noise fields if the profile
has them beside it and else noise made up from its measured levels (calibrate_raw.py), and only the
camera's own scenes (named <profile>-..., as ir-gr3-inbi is for gr3.json) keep their real noisy
frames. Each crop is scaled to a set brightness first, the same way denoise() scales a photo, so dark
night frames look like any other.

The weights that get scored and saved are a running average of the trained ones, taken every fourth
step, which is steadier than any single step. Every check logs the held-out RawNIND score and
bench_raw.py's scores for picking between runs: the GR III's, its mean and each ISO from 3,200 to
102,400, and the mean over the cameras whose other scenes train. The "aligned" ones are against clean
windows resampled onto the noisy ones, the ones to pick by; the others are what older runs logged.
run.json beside the log keeps the run's settings.
"""

import argparse
import fnmatch
import json
import math
import pathlib
import random
import threading
import time

import numpy as np
import torch

import bench_raw
import evaluate
import raw_pairs

PAIRS = evaluate.DATA / "rawpairs"
NOISE = evaluate.DATA / "noise"
MODELS = evaluate.MODELS / "raw"
#: Mean a crop or photo is scaled to before the network sees it.
LEVEL = 0.1
MAX_GAIN = 64.0
SYNTHETIC_SHARE = 0.4
#: How much of the laid-over noise is the GR III's real noise: one camera's, in a model for any camera.
BANK_SHARE = 0.2
#: The fewest electrons a pixel at LEVEL gets under laid-over noise. His night photo averages 17 and
#: half of it sits under 6; the GR IIIx's ISO 102,400 frame averages 9.
FLOOR = 2
#: How often a scene is drawn, by the start of its name. The review sites' ladders are hundreds of
#: cameras on a handful of scenes, so together they get about as many draws as the other real pairs,
#: which are far fewer cameras on far more scenes.
WEIGHTS = {"ladder-": 0.5}
#: The share of draws that go to scenes with real noisy frames, however many clean-only scenes join.
PAIR_SHARE = 1 / 3
#: Crops kept in memory, and how many of them a second are swapped for fresh ones from disk. The
#: scenes outgrew the disk cache, and a read per sample kept a shared server's disk busy.
POOL, READS = 1024, 4
#: His camera's colour, for the part of the loss that looks at the picture: white balance ranges that
#: cover daylight, tungsten and street light, and the GR III's matrix from camera colour to sRGB.
RED, BLUE = (1.5, 2.8), (1.4, 3.2)
TO_SRGB = ((1.703, -0.719, 0.016), (-0.188, 1.563, -0.375), (0.004, -0.406, 1.402))
EMA = 0.996


def network(width=32):
    from nafnet import CONFIGS, NAFNet

    return NAFNet(CONFIGS[f"nafnet-sidd-w{width}"], img_channel=4)


def trained(weights):
    """A saved model, at whatever width it was trained."""
    state = torch.load(weights, map_location="cpu")
    model = network(state["intro.weight"].shape[0])
    model.load_state_dict(state)
    return model.cuda().eval()


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
    """Roughly how a developed picture spaces brightness, so shadows count as much as highlights.

    Mirrored below zero: noise leaves a dark pixel negative as often as not, and a curve that is flat
    there would give the network nothing to learn from.
    """
    return x.sign() * ((x.abs() + 1e-3) ** (1 / 2.2) - 1e-3 ** (1 / 2.2))


def loss_of(result, clean):
    """How far the cleaned mosaic is from the clean one, as the picture they develop into.

    Half on the mosaic under a white balance, where detail lives, and half on its 2x2 cells as sRGB:
    the colour matrix is what blows left-over colour noise up into blotches.
    """
    n = clean.shape[0]
    draw = lambda low, high: torch.empty(n, device=clean.device).uniform_(low, high)
    red, green, blue = draw(*RED), torch.ones(n, device=clean.device), draw(*BLUE)
    balance = torch.stack([red, green, green, blue], 1).view(n, 4, 1, 1)
    a, b = pack(result) * balance, pack(clean) * balance
    cells = lambda p: torch.einsum(
        "ij,njhw->nihw", p.new_tensor(TO_SRGB), torch.stack([p[:, 0], (p[:, 1] + p[:, 2]) / 2, p[:, 3]], 1)
    )
    l1 = torch.nn.functional.l1_loss
    return (l1(tone(a), tone(b)) + l1(tone(cells(a)), tone(cells(b)))) / 2


def cleaner(model):
    """bench_raw.score's view of a model: mosaics (N, H, W) in, cleaned mosaics out, scaled as denoise() does."""

    def clean_up(mosaics, mean):
        gain = min(max(LEVEL / mean, 1), MAX_GAIN)
        with torch.autocast("cuda", torch.bfloat16):
            parts = [unpack(model(pack(part * gain)).float()) for part in mosaics.split(8)]
        return torch.cat(parts) / gain

    return clean_up


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


class Crops:
    """Endless batches of (clean, noisy) mosaics, flipped and turned with the colours kept on RGGB.

    One thread reads: it fills the pool, then swaps READS crops a second. A crop is drawn many times
    before it leaves, each time under another window, flip and noise.
    """

    def __init__(self, scenes, size, seed, real):
        self.size, self.real, self.rng, self.pool = size, real, random.Random(seed), []
        self.use(scenes)
        threading.Thread(target=self.read, daemon=True).start()

    def use(self, scenes):
        weights = [next((w for start, w in WEIGHTS.items() if s.startswith(start)), 1) for s in scenes]
        paired = list(scenes.values())
        total = {kind: sum(w for w, p in zip(weights, paired) if p == kind) for kind in (True, False)}
        weights = [w * (PAIR_SHARE if p else 1 - PAIR_SHARE) / total[p] for w, p in zip(weights, paired)]
        self.draw = list(scenes), weights

    def read(self):
        rng = random.Random(self.rng.random())
        while True:
            scene = rng.choices(*self.draw)[0]
            path = PAIRS / f"{scene}.npy"
            stack = np.load(path, mmap_mode="r")
            frames, crops, side, _ = stack.shape
            crop = rng.randrange(crops)
            noisy = frames // 2
            synthetic = not noisy or not fnmatch.fnmatch(scene, self.real) or rng.random() < SYNTHETIC_SHARE
            frame = 0 if synthetic else rng.randrange(1, noisy + 1)
            picks = [0] if synthetic else [frame + noisy, frame]
            # Read by offset: through the memory map the kernel fetches about three times what a crop holds.
            starts = [stack.offset + (i * crops + crop) * side * side * stack.itemsize for i in picks]
            planes = [np.fromfile(path, stack.dtype, side * side, offset=start) for start in starts]
            fresh = np.stack(planes).reshape(-1, side, side), synthetic
            if len(self.pool) < POOL:
                self.pool.append(fresh)
            else:
                self.pool[rng.randrange(POOL)] = fresh
            time.sleep(1 / READS if len(self.pool) == POOL else 0.05)

    def batch(self, n):
        while len(self.pool) < POOL // 4:
            time.sleep(1)
        pairs, kinds = [], []
        for _ in range(n):
            pair, synthetic = self.rng.choice(self.pool)
            side = pair.shape[-1]
            flip_y, flip_x, turn = (self.rng.random() < 0.5 for _ in range(3))
            # A flipped mosaic starts on red again one pixel in, so the window's corner takes that parity.
            y = self.rng.randrange((side - self.size) // 2) * 2 + flip_y
            x = self.rng.randrange((side - self.size) // 2) * 2 + flip_x
            if flip_y:
                pair = pair[:, ::-1]
            if flip_x:
                pair = pair[:, :, ::-1]
            pair = pair[:, y : y + self.size, x : x + self.size]
            if turn:
                pair = pair.transpose(0, 2, 1)
            pairs.append(np.ascontiguousarray(pair[[0, -1]]))
            kinds.append(synthetic)
        return torch.from_numpy(np.stack(pairs)).cuda().float(), torch.tensor(kinds, device="cuda")


def spread(n, device, low, high):
    """n draws between low and high, even on a log scale."""
    return torch.exp(torch.empty(n, device=device).uniform_(math.log(low), math.log(high)))


def bank_noise(clean, bank):
    """The GR III's real noise on clean mosaics (N, H, W) already at LEVEL brightness.

    Each crop gets a window of one ISO's unit-strength field, scaled by that ISO's shot and read noise
    at the brightness under it. The photo is taken as shot 1-16x darker than LEVEL and brought up, as
    a dim night frame is, but never past FLOOR.
    """
    fields, shot, read = bank
    n, size = clean.shape[0], clean.shape[-1]
    iso = torch.randint(len(fields), (n,), device=clean.device)
    corner = lambda side: (torch.randint((side - size) // 2, (n,)) * 2).tolist()
    unit = torch.stack(
        [
            fields[i, y : y + size, x : x + size]
            for i, y, x in zip(iso.tolist(), corner(fields.shape[1]), corner(fields.shape[2]))
        ]
    ).float()
    darkest = LEVEL / (FLOOR * shot[iso].mean(1))
    dim = torch.minimum(spread(n, clean.device, 1, 16), darkest).clamp(min=1).view(n, 1, 1, 1)
    variance = pack(clean).clamp(min=0) * shot[iso].view(n, 4, 1, 1) * dim + (read[iso].view(n, 4, 1, 1) * dim) ** 2
    return clean + unit * unpack(variance.sqrt())


def noise_bank(profile):
    """A camera's noise fields on the GPU, with each ISO's shot and read noise per colour plane."""
    levels = json.loads(profile.read_text())["iso"].values()
    fields = torch.from_numpy(np.load(profile.with_suffix(".npy"))).cuda()
    return fields, *(torch.tensor([level[kind] for level in levels], device="cuda") for kind in ("shot", "read"))


def made_up_noise(clean, profile):
    """Shot, read and row noise on clean mosaics (N, H, W) already at LEVEL brightness.

    A profile holds a camera's measured levels per ISO in normalised raw units; the photo is taken as
    shot 1-16x darker than LEVEL and brought up, as a dim night frame is.
    """
    n, device = clean.shape[0], clean.device
    uniform = lambda low, high: spread(n, device, low, high)
    if profile:
        levels = [random.choice(list(profile["iso"].values())) for _ in range(n)]
        dim = uniform(1, 16)
        shot = torch.tensor([lv["shot"] for lv in levels], device=device) * dim
        read = torch.tensor([lv["read"] for lv in levels], device=device) * dim
        row = torch.tensor([lv.get("row", 0.0) for lv in levels], device=device) * dim
    else:
        # From 3,000 electrons at LEVEL down to FLOOR, with half to eight electrons of read noise; the
        # GR IIIx has about 3.
        shot = uniform(LEVEL / 3000, LEVEL / FLOOR)
        read = shot * uniform(0.5, 8)
        row = read * torch.empty(n, device=device).uniform_(0, 0.3)
    view = lambda t: t.view(n, 1, 1)
    photons = torch.poisson(clean.clamp(min=0) / view(shot)) * view(shot)
    return photons + torch.randn_like(clean) * view(read) + torch.randn(n, clean.shape[1], 1, device=device) * view(row)


def gains(clean):
    """Scale each crop to LEVEL, from six times darker to twice brighter.

    denoise() scales a whole photo to LEVEL, and a night photo's tiles are mostly darker than its mean.
    """
    mean = clean.mean(dim=(1, 2)).clamp(min=1e-6)
    return (LEVEL / mean * spread(len(mean), mean.device, 1 / 6, 2)).clamp(1 / 4, MAX_GAIN).view(-1, 1, 1)


def scenes(split, pattern):
    """Each scene's name, and whether it has real noisy frames.

    A scene whose noisy frames have no clean twins yet, built by an older raw_pairs.py, stays out.
    """
    found, stale = {}, 0
    for meta in sorted(PAIRS.glob("*.json")):
        info = json.loads(meta.read_text())
        if info["split"] == split and fnmatch.fnmatch(meta.stem, pattern) and meta.with_suffix(".npy").exists():
            if raw_pairs.twinned(info):
                found[meta.stem] = bool(info["frames"])
            else:
                stale += 1
    if stale:
        print(f"{stale} {split} scenes left out for want of clean twins: rerun raw_pairs.py", flush=True)
    return found


@torch.no_grad()
def validate(model, val):
    """Mean PSNR on the toned mosaic, over every held-out real noisy frame's first four crops."""
    model.eval()
    scores = []
    for scene in val:
        stack = np.load(PAIRS / f"{scene}.npy", mmap_mode="r")
        twins = stack.shape[0] // 2
        for frame in range(1, twins + 1):
            clean = torch.from_numpy(stack[frame + twins, :4, 64:448, 64:448].astype(np.float32)).cuda()
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
    torch.backends.cudnn.benchmark = True
    out = MODELS / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(vars(args)))
    camera = pathlib.Path(args.profile or NOISE / "gr3.json")
    bank = noise_bank(camera) if camera.with_suffix(".npy").exists() else None
    profile = json.loads(camera.read_text()) if args.profile and not bank else None
    share = 1 if args.profile else BANK_SHARE
    model = network(args.width).cuda().to(memory_format=torch.channels_last)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    else:
        sidd = torch.load(evaluate.MODELS / f"NAFNet-SIDD-width{args.width}.pth", map_location="cpu")["params"]
        model.load_state_dict({k: v for k, v in sidd.items() if not k.startswith(("intro", "ending"))}, strict=False)
    average = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0, fused=True)
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
        average.load_state_dict(state["average"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step = state["step"]
    compiled = torch.compile(model)
    bench_raw.kept("gr3")
    val = scenes("val", args.scenes)
    log = (out / "log.jsonl").open("a")
    running = torch.zeros((), device="cuda")
    train = scenes("train", args.scenes)
    crops = Crops(train, args.size, args.seed + step, f"*{camera.stem}-*" if args.profile else "*")
    while step < args.steps:
        pair, synthetic = crops.batch(args.batch)
        clean, noisy = pair[:, 0], pair[:, 1]
        synthetic = synthetic.view(-1, 1, 1)
        g = gains(clean)
        clean, noisy = clean * g, noisy * g
        if synthetic.any():
            laid = made_up_noise(clean, profile)
            if bank:
                laid = torch.where(torch.rand_like(g) < share, bank_noise(clean, bank), laid)
            noisy = torch.where(synthetic, laid, noisy)
        with torch.autocast("cuda", torch.bfloat16):
            result = unpack(compiled(pack(noisy).contiguous(memory_format=torch.channels_last)).float())
        loss = loss_of(result, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        schedule.step()
        step += 1
        if step % 4 == 0:
            average.update_parameters(model)
        running += loss.detach()
        if step % args.check == 0 or step == args.steps:
            bench, aligned = bench_raw.scores(cleaner(average.module), "gr3")
            seen, seen_aligned = bench_raw.scores(cleaner(average.module), "seen")
            mean_psnr = lambda by_iso: round(float(np.mean([psnr for psnr, _ in by_iso.values()])), 3)
            line = {
                "step": step,
                "loss": round(running.item() / args.check, 5),
                "val_psnr": round(validate(average.module, val), 3) if val else float("nan"),
                "gr3_psnr": mean_psnr(bench),
                "gr3_by_iso": [round(psnr, 3) for psnr, _ in bench.values()],
                "seen_psnr": mean_psnr(seen),
                "gr3_aligned": mean_psnr(aligned),
                "seen_aligned": mean_psnr(seen_aligned),
                "lr": schedule.get_last_lr()[0],
                "scenes": len(train),
                "time": round(time.time()),
            }
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + "\n")
            log.flush()
            running.zero_()
            state = {
                "model": model.state_dict(),
                "average": average.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "step": step,
            }
            torch.save(state, out / "last.tmp")
            (out / "last.tmp").rename(out / "last.pth")
            if step % args.keep == 0:
                torch.save(average.module.state_dict(), out / f"step{step}.pth")
        if step % args.rescan == 0:
            # Crops raw_pairs.py finished meanwhile join in.
            train = scenes("train", args.scenes)
            crops.use(train)
    torch.save(average.module.state_dict(), out / "faithful.pth")


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init")
    parser.add_argument(
        "--width", type=int, default=32, help="the network's width, 32 or 64: four times the work at 64"
    )
    parser.add_argument("--scenes", default="*")
    parser.add_argument("--profile")
    main(parser.parse_args())
