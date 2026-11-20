"""A photo's black offset per colour plane, looked up by the camera body that took it: uv run blacks.py

A noisy frame sits on an offset of its own that the file's black level misses (raw_pairs.match), up to
a tenth of the frame's mean at a camera's highest ISO. The photo itself does not say how big it is:
its darkest corner is offset and scene together, its noise gives the offset only to someone who knows
the read noise better than anyone does, and the masked pixels some raws keep are on another scale
than the picture or missing. What does say is the same body's other frames at that ISO. So this
writes data/noise/blacks.json from the rows fit_noise.py measured on the review sites' ladders: for
each body, each ISO's offsets, the mean over that body's scenes, in the frame's own normalised units.

A body is its EXIF camera name and, where the raw carries one, its serial number. Offsets do not
carry over from one body of a camera to the next, so a photo whose body is not in the table, or
whose ISO is outside what was measured, gets no offsets at all. A raw without a serial number
(Ricoh, Sony, Samsung) cannot tell its body from another of the same camera, and takes that camera's row.
A pre-production body calls itself MODEL-NAME whatever camera it is, so those stay out.
"""

import functools
import json
import subprocess

import numpy as np

import raw_pairs

TABLE = raw_pairs.DATA / "noise" / "blacks.json"
ROWS = raw_pairs.DATA / "noise" / "cameras.json"
ISO_TAGS = ("RecommendedExposureIndex", "StandardOutputSensitivity", "ISOSpeedRatings")
SERIAL_TAGS = ("BodySerialNumber", "SerialNumber")
UNKNOWN_ISO = (0, 65535)
UNNAMED = ("", "MODEL-NAME")


def body(path):
    """The body that took a raw, as its camera name and serial number, and the ISO it was taken at (0 when unnamed).

    EXIF's ISO field stops at 65,535, so a higher ISO is read from the fields that took over from it.
    """
    tags = ["-gExif.Image.Model", "-gSerialNumber", *(f"-gExif.Photo.{tag}" for tag in ISO_TAGS)]
    listed = subprocess.run(["exiv2", "-Pkv", *tags, str(path)], check=False, capture_output=True, text=True)
    found = {}
    for line in listed.stdout.splitlines():
        key, _, value = line.partition(" ")
        found.setdefault(key.split(".")[-1], value.strip())
    serial = next((found[tag] for tag in SERIAL_TAGS if found.get(tag)), "")
    number = lambda tag: int(found[tag].split()[0]) if found.get(tag, "")[:1].isdigit() else 0
    iso = next((number(tag) for tag in ISO_TAGS if number(tag) not in UNKNOWN_ISO), 0)
    return f"{found.get('Model', '')} {serial}".strip(), iso


@functools.cache
def table():
    return json.loads(TABLE.read_text()) if TABLE.exists() else {}


def of(path):
    """A raw's four black offsets (R, G, G, B) in its own normalised units, between the ISOs its body was measured at."""
    name, iso = body(path)
    rows = table().get(name)
    if not rows or not iso:
        return np.zeros(4, np.float32)
    isos = sorted(rows, key=int)
    steps, offsets = np.log([int(step) for step in isos]), np.array([rows[step] for step in isos])
    return np.array([np.interp(np.log(iso), steps, plane, left=0, right=0) for plane in offsets.T], np.float32)


def main():
    scenes = {}
    for row in json.loads(ROWS.read_text()):
        scenes.setdefault((row["site"], row["camera"], row["scene"]), {})[row["iso"]] = row["offset"]
    bodies = {}
    for ladder, by_iso in sorted(scenes.items()):
        name, _ = body(next(file for file in raw_pairs.LADDERS.joinpath(*ladder).iterdir() if file.stem.isdigit()))
        for iso, offsets in by_iso.items() if name not in UNNAMED else ():
            bodies.setdefault(name, {}).setdefault(iso, []).append(offsets)
    found = {
        name: {str(iso): np.mean(offsets, axis=0).tolist() for iso, offsets in sorted(by_iso.items())}
        for name, by_iso in sorted(bodies.items())
    }
    TABLE.write_text(json.dumps(found))
    print(len(found), "bodies from", len(scenes), "ladders")


if __name__ == "__main__":
    main()
