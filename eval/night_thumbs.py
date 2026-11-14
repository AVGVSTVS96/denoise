"""Render a small preview of each raw file given, next to it under thumbs/, and print its EXIF basics."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image


def exif(path: Path) -> dict:
    out = subprocess.run(["exiv2", "-q", "-Pkv", str(path)], capture_output=True, text=True).stdout
    keys = {
        "Exif.Image.Model": "camera",
        "Exif.Photo.ISOSpeedRatings": "iso",
        "Exif.Photo.PhotographicSensitivity": "iso",
        "Exif.Photo.RecommendedExposureIndex": "iso",
        "Exif.Photo.ExposureTime": "exposure",
        "Exif.Photo.DateTimeOriginal": "time",
    }
    info = {}
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key in keys and keys[key] not in info:
            info[keys[key]] = value.strip()
    return info


def thumb(path: Path) -> Path:
    dest = path.parent / "thumbs" / (path.stem + ".jpg")
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=False, output_bps=8)
        img = Image.fromarray(np.asarray(rgb))
        img.thumbnail((800, 800))
        img.save(dest, quality=85)
    return dest


for arg in sys.argv[1:]:
    p = Path(arg)
    try:
        t = thumb(p)
        print(p.name, exif(p), t, sep="\t")
    except Exception as e:  # noqa: BLE001
        print(p.name, "FAILED", e, sep="\t")
