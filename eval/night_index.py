"""Build night-raw/index.jsonl and night-raw/INDEX.md from the files on disk and the source notes.

usage: python -I night_index.py <night-raw dir>
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAW = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
root = Path(sys.argv[1])
manifests = root.parent / "manifests"


def exif(path: Path) -> dict:
    out = subprocess.run(["exiv2", "-q", "-Pkv", str(path)], capture_output=True, text=True).stdout
    wanted = {
        "Exif.Image.Make": "make",
        "Exif.Image.Model": "camera",
        "Exif.Photo.ISOSpeedRatings": "iso",
        "Exif.Photo.PhotographicSensitivity": "iso",
        "Exif.Photo.ExposureTime": "exposure",
    }
    info = {}
    for line in out.splitlines():
        key, _, value = line.partition(" ")
        if key in wanted and wanted[key] not in info and value.strip():
            info[wanted[key]] = value.strip()
    if "iso" in info:
        info["iso"] = int(info["iso"].split()[0])
    return info


notes = {}
for line in (root / "sources.jsonl").read_text().splitlines():
    r = json.loads(line)
    notes[r["file"]] = r

reviews = manifests / "ir-download.jsonl"
if reviews.exists():
    for line in reviews.read_text().splitlines():
        r = json.loads(line)
        if r.get("file"):
            notes[r["file"]] = {**r, "source": r.get("source") or r["page"].split("/")[2]}

pairs = {}
for line in (manifests / "datasets.jsonl").read_text().splitlines():
    r = json.loads(line)
    name = Path(r.get("member") or r["filename"]).name
    pairs[f"pairs/{r['pair_id']}/{r['role']}_{name}"] = {
        "source": r["dataset"],
        "url": r["url"],
        "license": r["license"],
        "scene": r["scene"],
        "pair_id": r["pair_id"],
        "role": r["role"],
    }

rows = []
for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in RAW and "thumbs" not in p.parts):
    rel = str(path.relative_to(root))
    note = notes.get(rel) or pairs.get(rel)
    if note is None:
        print("no source note:", rel, file=sys.stderr)
        continue
    rows.append({"file": rel, "bytes": path.stat().st_size, **exif(path), **{k: v for k, v in note.items() if v not in (None, "")}})

(root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

singles = [r for r in rows if "pair_id" not in r]
noisy = [r for r in rows if r.get("role") == "noisy"]
clean = {r["pair_id"] for r in rows if r.get("role") == "clean"}
complete = [r for r in noisy if r["pair_id"] in clean]


def band(iso):
    if not iso:
        return "unknown"
    return next(f"{lo:,}+" for lo in (51200, 25600, 12800, 6400, 3200, 0) if iso >= lo)


lines = [
    "# Night raw collection",
    "",
    f"{len(singles)} single high-ISO night raws, and {len(complete)} noisy frames with a clean tripod reference "
    f"across {len({r['pair_id'] for r in complete})} scenes "
    f"({len(rows)} files, {sum(r['bytes'] for r in rows) / 1e9:.1f} GB). "
    "Every file's source URL and license is in index.jsonl. Rebuild with `python -I eval/night_index.py eval/data/night-raw`.",
    "",
    "## Single shots by ISO",
    "",
    *(f"- {b}: {n}" for b, n in sorted(Counter(band(r.get("iso")) for r in singles).items(), key=lambda kv: -int(kv[0].rstrip("+").replace(",", "") or 0) if kv[0] != "unknown" else 1)),
    "",
    "## Pairs by dataset",
    "",
    *(f"- {d}: {n}" for d, n in Counter(r["source"] for r in complete).most_common()),
    "",
    "## Single shots",
    "",
    "| file | camera | ISO | scene | source | license |",
    "|---|---|---|---|---|---|",
    *(
        f"| {r['file']} | {r.get('camera', '?')} | {r.get('iso', '?')} | {r.get('scene', '')} | {r['source']} | {r.get('license', '')} |"
        for r in sorted(singles, key=lambda r: -(r.get("iso") or 0))
    ),
    "",
    "## Pairs (noisy frame shown; its clean reference sits in the same folder)",
    "",
    "| pair | camera | noisy ISO | scene | dataset | license |",
    "|---|---|---|---|---|---|",
    *(
        f"| {r['pair_id']} | {r.get('camera', '?')} | {r.get('iso', '?')} | {r['scene']} | {r['source']} | {r['license']} |"
        for r in sorted(complete, key=lambda r: -(r.get("iso") or 0))
    ),
    "",
]
(root / "INDEX.md").write_text("\n".join(lines))
print(f"{len(singles)} singles, {len(complete)} pairs, {len(rows)} files")
