"""Pages of rendered print, for train_raw.py --text to lay into the joint crops' truth: uv run text_bank.py

Small print is the finest detail a photo holds that anyone reads, and the scenes that train hold
little of it. A page here is SIDE px square: one face's lines in one language, as coverage from 0,
bare, to 255, ink. Every page is set at the size that makes its strokes STROKE px wide, the widest
train_raw.py asks for, so a page is only ever shrunk. A face too thin to give two lines at that size
stays out.

The typefaces are the ones installed here, asked of fontconfig language by language. Families that
live in the same files count as one typeface, as Noto Sans CJK's five regions do, and its weights,
widths and italics are each a face: a page draws a typeface, then a face of it. The words are the
translations of the programs installed here (LOCALES), with English from the same files, and
"numbers" is digits, signs and capitals made up. A word a face has no letter for is left out, and
one page in four is set in capitals. LANGUAGES says how often each is drawn: about half the pages
are in Latin letters.

Writes data/noise/text.npy, uint8 (PAGES, SIDE, SIDE).
"""

import bisect
import collections
import functools
import pathlib
import random
import re
import string
import struct
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import raw_pairs

OUT = raw_pairs.DATA / "noise" / "text.npy"
LOCALES = pathlib.Path("/usr/share/locale")
PAGES, SIDE, STROKE = 512, 512, 4
TRIAL, SAMPLE, WORDS = 100, 24, 400
CAPITALS = 0.25
LANGUAGES = {
    "en": 8,
    "numbers": 3,
    "de": 1,
    "fr": 1,
    "es": 1,
    "pl": 1,
    "tr": 1,
    "vi": 1,
    "ru": 2,
    "uk": 1,
    "el": 1,
    "ar": 2,
    "fa": 1,
    "he": 1,
    "hi": 1,
    "bn": 1,
    "ta": 1,
    "th": 1,
    "ja": 2,
    "ko": 1,
    "zh_CN": 2,
    "zh_TW": 1,
    "ka": 1,
    "hy": 1,
}
ENGLISH_FROM = "de"
SKIPPED = ("Nerd Font", "D050000L", "Symbols")
FORMATTING = re.compile(r"%\S+|\{[^}]*\}|<[^>]*>|&\w+;|\\\w")
ACCELERATORS = str.maketrans("", "", "_&|")
MARKS = string.digits * 3 + string.ascii_uppercase + ".,:;/-+%$€£#&()*=@°'\"!?"


@functools.cache
def typefaces(language):
    """A language's typefaces, each the list of its faces: family, file, fontconfig's index, the characters it has."""
    wanted = "en" if language == "numbers" else language.lower().replace("_", "-")
    listed = subprocess.run(
        ["fc-list", f":lang={wanted}", "--format", "%{family[0]}\t%{file}\t%{index}\t%{variable}\t%{charset}\n"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    families = collections.defaultdict(list)
    for line in sorted(set(listed.splitlines())):
        family, file, index, variable, charset = line.split("\t")
        if variable == "False" and not any(name in family for name in SKIPPED):
            families[family].append((family, file, int(index), charset))
    found = collections.defaultdict(list)
    for faces in families.values():
        found[frozenset(file for _, file, _, _ in faces)] += faces
    return list(found.values())


def messages(path):
    """A compiled catalog's messages (.mo), English to translation, and the charset its header names.

    The file is two tables of lengths and offsets, English and translation, and the header is the
    translation of nothing. A message's context ends at byte 4 and its plurals start at byte 0.
    """
    data = path.read_bytes()
    count, english, translated = struct.unpack_from("<3I", data, 8)
    column = lambda at: [
        data[start : start + length].split(b"\0")[0].split(b"\x04")[-1]
        for length, start in struct.iter_unpack("<2I", data[at : at + 8 * count])
    ]
    found = dict(zip(column(english), column(translated)))
    charset = re.search(rb"charset=([\w-]+)", found.pop(b"", b""))
    return found, charset[1].decode() if charset else "utf-8"


@functools.cache
def words(language):
    """A language's print, word by word and in the files' order."""
    if language == "numbers":
        rng = random.Random(0)
        return ["".join(rng.choices(MARKS, k=rng.randint(1, 8))) for _ in range(100 * WORDS)]
    found = []
    for path in sorted((LOCALES / (ENGLISH_FROM if language == "en" else language) / "LC_MESSAGES").glob("*.mo")):
        said, charset = messages(path)
        for message in said if language == "en" else said.values():
            found += FORMATTING.sub(" ", message.decode(charset)).translate(ACCELERATORS).split()
    return found


def face(file, index, size):
    """A face at a size in px. fontconfig counts a variable font's named weights from 1 in its index's upper half."""
    font = ImageFont.truetype(file, size, index=index & 0xFFFF)
    if index >> 16:
        font.set_variation_by_name(font.get_variation_names()[(index >> 16) - 1])
    return font


def stroke(font, sample):
    """How wide a face's strokes are, in px. Across a stroke w px wide the distance to its edge averages (w + 2) / 4."""
    image = Image.new("L", ((SAMPLE + 2) * TRIAL, 3 * TRIAL))
    ImageDraw.Draw(image).text((TRIAL, TRIAL), sample, 255, font)
    ink = (np.asarray(image) > 127).astype(np.uint8)
    return 4 * cv2.distanceTransform(ink, cv2.DIST_L2, 5)[ink > 0].mean() - 2


def page(faces, language, rng):
    """One page of a typeface: its family, and the page (SIDE, SIDE) uint8, or None from a face too thin for two lines."""
    family, file, index, charset = rng.choice(faces)
    spans = [[int(end, 16) for end in span.split("-")] for span in charset.split()]
    has = {chr(code) for span in spans for code in range(span[0], span[-1] + 1)}
    start = rng.randrange(len(words(language)) - WORDS)
    capitals = rng.random() < CAPITALS
    drawn = (word.upper() if capitals else word for word in words(language)[start : start + WORDS])
    text = " ".join(word for word in drawn if has.issuperset(word))
    font = face(file, index, TRIAL * STROKE / stroke(face(file, index, TRIAL), text[:SAMPLE]))
    tall = sum(font.getmetrics())
    if tall > SIDE / 2:
        return family, None
    image = Image.new("L", (SIDE, SIDE))
    for y in range(0, SIDE - tall + 1, tall):
        fits = bisect.bisect_right(range(min(len(text), SIDE)), SIDE, key=lambda end: font.getlength(text[:end])) - 1
        space = text.rfind(" ", fits // 2, fits + 1)
        end = space if space > 0 and fits < len(text) else fits
        ImageDraw.Draw(image).text((0, y), text[:end], 255, font)
        text = text[end:].lstrip()
    return family, np.asarray(image)


def main():
    rng = random.Random(0)
    pages = []
    while len(pages) < PAGES:
        language = rng.choices(list(LANGUAGES), list(LANGUAGES.values()))[0]
        family, made = page(rng.choice(typefaces(language)), language, rng)
        if made is not None:
            pages.append(made)
            print(f"{len(pages)} {language}: {family}, {made.mean() / 255:.0%} ink", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, np.stack(pages))


if __name__ == "__main__":
    main()
