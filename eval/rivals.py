"""Other people's raw denoisers on bench_raw.py's test, and a free trick for ours.

    uv run rivals.py cameras                                  where the raws are: what RawForge needs of each camera
    uv run rivals.py fetch                                    the rivals' weights, into data/rivals
    uv run rivals.py darktable rawforge-heavy led-nafnet ...  rivals, by the names below
    uv run rivals.py v4=data/models/raw/faithful-v4/faithful.pth ...

cameras and fetch write data/rivals, which is all a machine that scores needs beside data/bench: copy
it there. Every rival becomes a row in data/out/bench.json beside ours, scored by bench_raw.scores as
ours are, and is fed the way its own app feeds it:

darktable    darktable 5.6's rawdenoise-nind, a U-Net trained on RawNIND. Packed RGGB in 512 px tiles,
             mirrored past the edge. It demosaics too: camera RGB comes out at a gain of its own.
             darktable sets that back to the input's mean over all four planes, which leaves a
             picture too bright wherever green is not the mean of red and blue (3 dB here). Each
             colour is set back to its own planes' mean instead, and the input is not clipped at
             black as darktable clips it, which costs it up to 1 dB at ISO 102,400.
rawforge-*   RawForge's TreeNet denoisers, NAFNet-like and told the ISO. They take a picture, not a
             mosaic: the mosaic's colours turned to linear Rec.2020 without a white balance, then
             Malvar's demosaic, clipped to 0..1. Rec.2020 comes out and is turned back to camera RGB.
rawforge-jdd RawForge's joint demosaic and denoise: the mosaic as three sparse colour planes and their
             three masks with the black level left in, camera RGB out. It counts on a black level
             of 1/32 of white or more: the Ricohs' 1/256 loses it 2 dB, so that is the least it gets.
led-*        LED's pretrained networks, the starting points it fine-tunes per camera: packed R, G, B, G
             under the mean of its five virtual cameras' alignments, the shot's exposure as it is.
             LED's own tests fit a result's brightness to the clean frame before scoring; here each
             colour plane is fitted to the noisy window, worth 1 to 4 dB to it.
eld-*        ELD's U-Nets as LED's model zoo ships them, each trained on ELD's noise model of one
             camera, the Sony A7S II or the Nikon D850, for gains of 1 to 200. Fed and fitted as LED's.

One that puts out a picture gets two rows. "<name>" develops that picture with the test's white
balance and matrix, so its own demosaic is scored against the clean frame's bilinear one.
"<name>-cfa" takes the picture's pixels back to a mosaic first, which is what RawForge's --cfa saves,
so it goes through the same bilinear demosaic as the clean frame and as ours: the like-for-like row.

On bench_raw.py's full-colour tests, "<camera>-colour" and "shift", a rival gets one row: its own
picture against truth no demosaic made, or its mosaic through the bilinear demosaic.

A model of ours gets "<name>-x8", the mean over the mosaic's eight flips and turns.
"""

import collections
import functools
import importlib.util
import json
import subprocess
import sys
import types
import urllib.request
import zipfile

import numpy as np
import torch

import bench_raw
import raw_pairs
import train_raw

RIVALS = raw_pairs.DATA / "rivals"
BOARD = raw_pairs.DATA / "out" / "bench.json"
RELEASES = {
    "darktable/rawdenoise-nind.dtmodel": "darktable-org/darktable-ai/releases/download/release-5.6.0",
    "rawforge/ShadowWeightedL1.pt": "rymuelle/RawRefinery/releases/download/v1.2.1-alpha",
    "rawforge/ShadowWeightedL1_24_deep_500.pt": "rymuelle/RawRefinery/releases/download/v1.2.1-alpha",
    "rawforge/ShadowWeightedL1_light.pt": "rymuelle/RawRefinery/releases/download/v1.2.1-alpha",
    "rawforge/DemoRestormerDiT_to_demo_cont_cont_768_RawNIND.pt": "rymuelle/RawForge/releases/download/JDD_v1.0.0",
}
DRIVE = {
    "led/nafnet.pth": "1FmqGv_YICLX4Gc-aWzvcTl8aWgeJ5cEB",
    "led/unet-1-200.pth": "1IzOkJuHWQVXmkzFJzQ9-gkPXBlrutO2p",
    "led/eld-sony.pth": "1o0kr446Se2j5iemXiLjHxXy5UjLqgc6c",
    "led/eld-nikon.pth": "1seTloGkpM2XbeeSwKSJHvq9khICnbfLv",
}
UTNET = "https://raw.githubusercontent.com/trougnouf/rawnind_jddc/master/src/rawnind/models/raw_denoiser.py"
REC2020_FROM_XYZ = (
    (1.71666343, -0.35567332, -0.25336809),
    (-0.66667384, 1.61645574, 0.0157683),
    (0.01764248, -0.04277698, 0.94224328),
)
Shot = collections.namedtuple("Shot", "iso camera_from_xyz black_over_white")


def cameras():
    """Each test camera's XYZ matrix and black level, from its clean raw, into data/rivals/cameras.json."""
    import rawpy

    paths = {
        name: raw_pairs.NIGHT / "imaging-resource" / pattern.format(100) for name, pattern in bench_raw.CAMERAS.items()
    }
    for folder in (*(folder for folders in bench_raw.groups().values() for folder in folders), bench_raw.SHIFT_LADDER):
        steps = {int(file.stem): file for file in folder.iterdir() if file.stem.isdigit()}
        paths[folder.parent.name] = steps[min(steps)]
    table = {}
    for name, path in paths.items():
        with rawpy.imread(str(path)) as raw:
            black = float(np.mean(raw.black_level_per_channel)) / raw.white_level
            table[name] = {"camera_from_xyz": raw.rgb_xyz_matrix[:3].tolist(), "black_over_white": black}
    RIVALS.mkdir(parents=True, exist_ok=True)
    (RIVALS / "cameras.json").write_text(json.dumps(table, indent=1))


@functools.cache
def shots():
    """Every test frame's Shot by its mean, which is all bench_raw hands a cleaner beside the pixels.

    A Shot is what a rival's own app knows of a frame: its ISO, the camera's matrix from XYZ to its
    own colour, and its black level over its white level.
    """
    known = json.loads((RIVALS / "cameras.json").read_text())
    table = {}
    for file in bench_raw.BENCH.rglob("*.npz"):
        data = np.load(file)
        if "isos" not in data:
            continue
        camera = file.stem.split("--")[-1]
        camera = camera if camera in known else camera.rsplit("-", 1)[0]
        for iso, mean in zip(data["isos"].tolist(), data["means"].tolist()):
            table[mean] = Shot(iso, **known[camera])
    return table


def source(path, **stubs):
    """One file of a rival's code as a module, with stand-ins for the parts of its package it imports."""
    for name, stub in stubs.items():
        *packages, last = name.split(".")
        for depth in range(1, len(packages) + 1):
            sys.modules.setdefault(".".join(packages[:depth]), types.ModuleType(packages[depth - 1]))
        setattr(sys.modules[".".join(packages)], last, sys.modules.setdefault(name, stub))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch():
    """Every rival's weights. darktable's come as ONNX and are saved again as TorchScript: RawNIND's
    UtNet2 under the same weights, so a machine that scores needs torch alone."""
    import gdown
    import onnx
    from onnx import numpy_helper

    for file in (*RELEASES, *DRIVE):
        (RIVALS / file).parent.mkdir(parents=True, exist_ok=True)
    for file, release in RELEASES.items():
        if not (RIVALS / file).exists():
            torch.hub.download_url_to_file(f"https://github.com/{release}/{file.split('/')[1]}", RIVALS / file)
    for file, drive_id in DRIVE.items():
        if not (RIVALS / file).exists():
            gdown.download(id=drive_id, output=str(RIVALS / file))
    if not (RIVALS / "led" / "LED").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Srameo/LED.git"], cwd=RIVALS / "led")
    urllib.request.urlretrieve(UTNET, RIVALS / "darktable" / "raw_denoiser.py")
    with zipfile.ZipFile(RIVALS / "darktable" / "rawdenoise-nind.dtmodel") as package:
        graph = onnx.load_model_from_string(package.read("rawdenoise-nind/model_bayer.onnx")).graph
    unet = source(RIVALS / "darktable" / "raw_denoiser.py", **{"rawnind.libs.rawproc": types.ModuleType("rawproc")})
    model = unet.UtNet2(in_channels=4, funit=32, activation="LeakyReLU").eval()
    model.load_state_dict({w.name: torch.from_numpy(numpy_helper.to_array(w).copy()) for w in graph.initializer})
    torch.jit.trace(model, torch.zeros(1, 4, 512, 512)).save(str(RIVALS / "darktable" / "model_bayer.pt"))


def darktable():
    model = torch.jit.load(str(RIVALS / "darktable" / "model_bayer.pt"), map_location="cuda").eval()
    level = lambda planes: planes.mean((2, 3), keepdim=True)

    def clean_up(mosaics, mean):
        planes = train_raw.pack(mosaics)
        tiles = torch.nn.functional.pad(planes, (128, 128, 128, 128), mode="reflect")
        rgb = torch.cat([model(part) for part in tiles.split(4)])[..., 256:-256, 256:-256]
        red, green, also_green, blue = level(planes).unbind(1)
        return rgb * torch.stack([red, (green + also_green) / 2, blue], 1) / level(rgb)

    return clean_up


def malvar(mosaics):
    """RGGB mosaics (N, H, W) as pictures (N, 3, H, W) by Malvar, He and Cutler's demosaic, RawForge's choice."""
    cross = [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]
    beside = [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]]
    corner = [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]
    cross, beside, corner = mosaics.new_tensor([cross, beside, corner]) / 8
    padded = torch.nn.functional.pad(mosaics[:, None], (2, 2, 2, 2), mode="reflect")
    cross, beside, above, corner = torch.nn.functional.conv2d(
        padded, torch.stack([cross, beside, beside.T, corner])[:, None]
    ).unbind(1)
    red, green, blue = mosaics.clone(), mosaics.clone(), mosaics.clone()
    green[:, 0::2, 0::2], green[:, 1::2, 1::2] = cross[:, 0::2, 0::2], cross[:, 1::2, 1::2]
    red[:, 0::2, 1::2], red[:, 1::2, 0::2], red[:, 1::2, 1::2] = (
        beside[:, 0::2, 1::2],
        above[:, 1::2, 0::2],
        corner[:, 1::2, 1::2],
    )
    blue[:, 1::2, 0::2], blue[:, 0::2, 1::2], blue[:, 0::2, 0::2] = (
        beside[:, 1::2, 0::2],
        above[:, 0::2, 1::2],
        corner[:, 0::2, 0::2],
    )
    return torch.stack([red, green, blue], 1)


def rec2020(mosaics, camera_from_xyz):
    """RawForge's input: the mosaics' colours in linear Rec.2020, mixed within each 2x2 cell, then demosaiced.

    Returns the pictures and the matrix that took camera RGB there.
    """
    matrix = mosaics.new_tensor(REC2020_FROM_XYZ) @ torch.linalg.inv(mosaics.new_tensor(camera_from_xyz))
    red, green, also_green, blue = train_raw.pack(mosaics).unbind(1)
    mixed = lambda row, green: matrix[row, 0] * red + matrix[row, 1] * green + matrix[row, 2] * blue
    both = (green + also_green) / 2
    turned = torch.stack([mixed(0, both), mixed(1, green), mixed(1, also_green), mixed(2, both)], 1)
    return malvar(train_raw.unpack(turned)).clamp(0, 1), matrix


def rawforge(file):
    model = torch.jit.load(str(RIVALS / "rawforge" / file), map_location="cuda").eval()

    def clean_up(mosaics, mean):
        shot = shots()[mean]
        pictures, matrix = rec2020(mosaics, shot.camera_from_xyz)
        told = mosaics.new_tensor([[min(shot.iso, 65535) / 6400]]).half()
        with torch.autocast("cuda", torch.float16):
            cleaned = torch.cat([model(part, told.expand(len(part), 1)) for part in pictures.half().split(4)])
        return torch.einsum("ij,njhw->nihw", torch.linalg.inv(matrix), cleaned.float())

    return clean_up


def rawforge_jdd():
    file = "DemoRestormerDiT_to_demo_cont_cont_768_RawNIND.pt"
    model = torch.jit.load(str(RIVALS / "rawforge" / file), map_location="cuda").eval()

    def clean_up(mosaics, mean):
        black = max(shots()[mean].black_over_white, 1 / 32)
        sites = mosaics.new_zeros(3, *mosaics.shape[1:])
        sites[0, 0::2, 0::2] = sites[1, 0::2, 1::2] = sites[1, 1::2, 0::2] = sites[2, 1::2, 1::2] = 1
        with_black = (mosaics * (1 - black) + black).clamp(0, 1)
        fed = torch.cat([with_black[:, None] * sites, sites.expand(len(mosaics), -1, -1, -1)], 1)
        with torch.autocast("cuda", torch.float16):
            cleaned = torch.cat([model(part) for part in fed.split(1)])
        return (cleaned.float() - black) / (1 - black)

    return clean_up


def led_arch(name):
    """One of LED's network files, without the rest of its package: only its registry is in the way."""
    registry = types.SimpleNamespace(ARCH_REGISTRY=types.SimpleNamespace(register=lambda: lambda arch: arch))
    return source(RIVALS / "led" / "LED" / "led" / "archs" / f"{name}.py", **{"led.utils.registry": registry})


def led(weights):
    if "nafnet" in weights:
        blocks = dict(enc_blk_nums=[2, 2, 2, 2], middle_blk_num=2, dec_blk_nums=[2, 2, 2, 2])
        base = led_arch("nafnet_arch").NAFNet(img_channel=4, width=32, **blocks)
        plain = ["ending", "dwconv2", "downconv", "upconv"]
    else:
        base, plain = led_arch("unet_arch").UNetArch(inchannels=4, outchannels=4, channels=32), ["conv10_1"]
    model = led_arch("repnr_utils").build_repnr_arch_from_base(
        base, dont_convert_module=plain, branch_num=5, align_opts={"init_weight": 1.0, "init_bias": 0.0}
    )
    model.load_state_dict(torch.load(RIVALS / "led" / weights, map_location="cpu")["params"])
    model.generalize()
    return low_light(model)


def eld(weights):
    model = led_arch("unet_arch").UNetArch(inchannels=4, outchannels=4, channels=32)
    model.load_state_dict(torch.load(RIVALS / "led" / weights, map_location="cpu")["params"])
    return low_light(model)


def low_light(model):
    """A cleaner from a network of the SID family: packed R, G, B, G in, clipped at white, and the same out."""
    model.cuda().eval()

    def clean_up(mosaics, mean):
        red, green, also_green, blue = train_raw.pack(mosaics).unbind(1)
        fed = torch.stack([red, green, blue, also_green], 1).clamp(max=1)
        red, green, blue, also_green = torch.cat([model(part) for part in fed.split(4)]).clamp(0, 1).unbind(1)
        return train_raw.unpack(torch.stack([red, green, also_green, blue], 1))

    return fitted(clean_up)


def fitted(clean_up):
    """clean_up with each colour plane of its result put on its noisy window's scale, by the line that fits best."""

    def run(mosaics, mean):
        noisy, result = train_raw.pack(mosaics), train_raw.pack(clean_up(mosaics, mean))
        level = lambda planes: planes.mean((2, 3), keepdim=True)
        spread = result - level(result)
        slope = level(spread * (noisy - level(noisy))) / level(spread**2)
        return train_raw.unpack(spread * slope + level(noisy))

    return run


PICTURES = {
    "darktable": darktable,
    "rawforge": lambda: rawforge("ShadowWeightedL1.pt"),
    "rawforge-heavy": lambda: rawforge("ShadowWeightedL1_24_deep_500.pt"),
    "rawforge-light": lambda: rawforge("ShadowWeightedL1_light.pt"),
    "rawforge-jdd": rawforge_jdd,
}
MOSAICS = {
    "led-nafnet": lambda: led("nafnet.pth"),
    "led-unet": lambda: led("unet-1-200.pth"),
    "eld-sony": lambda: eld("eld-sony.pth"),
    "eld-nikon": lambda: eld("eld-nikon.pth"),
}


def ensemble(clean_up):
    """clean_up as the mean of its results over a mosaic's eight flips and turns.

    A flipped mosaic starts on red again one pixel in, so a flip drops a pixel off one side and 31 off
    the other to stay a size the network takes; the strip it leaves is the mean of the passes that reach it.
    """

    def run(mosaics, mean):
        total, count = torch.zeros_like(mosaics), torch.zeros_like(mosaics)
        for down in (False, True):
            for across in (False, True):
                rows, columns = (slice(31, -1) if flipped else slice(None) for flipped in (down, across))
                dims = [dim for dim, flipped in ((1, down), (2, across)) if flipped]
                for turned in (False, True):
                    view = mosaics[:, rows, columns].flip(dims)
                    result = clean_up(view.transpose(1, 2) if turned else view, mean)
                    total[:, rows, columns] += (result.transpose(1, 2) if turned else result).flip(dims)
                    count[:, rows, columns] += 1
        return total / count

    return run


def replayed(clean_up):
    """clean_up, and a second cleaner that hands the first one's pictures back in order, as mosaics."""
    pictures = []

    def first(mosaics, mean):
        pictures.append(clean_up(mosaics, mean))
        return pictures[-1]

    return first, lambda mosaics, mean: bench_raw.mosaiced(pictures.pop(0))


def rows(spec):
    """What a name on the command line adds: {row name: cleaner} for the tests with real noise, in the
    order they must run, and the cleaner a full-colour test scores, which ours get from bench_raw.py."""
    if "=" in spec:
        name, weights = spec.split("=")
        return {f"{name}-x8": ensemble(train_raw.cleaner(train_raw.trained(weights)))}, None
    if spec in MOSAICS:
        clean_up = MOSAICS[spec]()
        return {spec: clean_up}, clean_up
    clean_up = PICTURES[spec]()
    first, second = replayed(clean_up)
    return {spec: first, f"{spec}-cfa": second}, clean_up


def main(specs):
    def add(test, name, by_iso):
        table = json.loads(BOARD.read_text()) if BOARD.exists() else {}
        table.setdefault(test, {})[name] = by_iso
        BOARD.with_suffix(".tmp").write_text(json.dumps(table, indent=1))
        BOARD.with_suffix(".tmp").rename(BOARD)
        cells = [*by_iso.values(), np.mean(list(by_iso.values()), axis=0)]
        print(f"{test:16}{name:20}" + "".join(f"{p:8.2f} {d:.3f}" for p, d in cells), flush=True)

    for spec in specs:
        cleaners, own = rows(spec)
        for camera in (*bench_raw.CAMERAS, *bench_raw.groups(), "exact"):
            tests = (camera,) if camera == "exact" else (camera, f"{camera}-aligned")
            for name, clean_up in cleaners.items():
                for test, by_iso in zip(tests, bench_raw.scores(clean_up, camera, with_lpips=True)):
                    add(test, name, by_iso)
        for camera in (*bench_raw.CAMERAS, *bench_raw.groups()) if own else ():
            add(f"{camera}-colour", spec, bench_raw.scores(own, f"colour/{camera}", with_lpips=True)[0])
        if own:
            add("shift", spec, bench_raw.scores(own, "shift", with_lpips=True)[0])


if __name__ == "__main__":
    if sys.argv[1:] in (["cameras"], ["fetch"]):
        {"cameras": cameras, "fetch": fetch}[sys.argv[1]]()
    else:
        main(sys.argv[1:])
