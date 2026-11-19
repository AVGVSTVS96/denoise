"""ISO ladders from camera review sites: one tripod scene per camera, raw, at base ISO and at every high ISO.

    python fetch_ladders.py list ir          write data/ladders/ir.jsonl, what the site holds
    python fetch_ladders.py get ir           download it, one file at a time; run again to carry on

Sources: ir is Imaging Resource's indoor portrait scene (inbi) and its still life (sli), epz is
ePHOTOzine's studio still life, pb is Photography Blog's, a scene of its own for every review.
dpreview is DPReview's studio scene in daylight and in low light: its listing reads, but the site
answers a script's file requests with a bot challenge, so nothing of it is fetched. Where every
camera shot the same scene, a ladder's worth is its noise, real and from hundreds of sensors, more
than its picture.

A ladder keeps its lowest ISO as the clean frame and every ISO at least 16 times that and 800 or more:
below that the clean frame's own noise is too large a share of the difference. Files land in
data/ladders/<source>/<camera>/<scene>/<iso>.<ext>, where raw_pairs.py picks them up. These are
review samples, all rights reserved: for research runs, not for a model that is sold.
"""

import gzip
import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).parent / "data" / "ladders"
AGENT = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
}
RAW = "dng|nef|nrw|arw|cr2|cr3|raf|orf|rw2|pef|srw|rwl|3fr|iiq|x3f"
IR = "https://old.imaging-resource.com"
DPREVIEW = (
    "https://web.archive.org/web/20260715020822id_/"
    "https://www.dpreview.com/wp-json/wayfinder-image-compare/v1/widgets/541206/frontend"
)
RATIO, FLOOR = 16, 800
TRIES = 8


def get(url, post=None):
    asked = urllib.request.Request(
        url, post and json.dumps(post).encode(), {**AGENT, "Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            body = urllib.request.urlopen(asked, timeout=120).read()
            return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
        time.sleep(10)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def ir():
    cameras = re.findall(
        r'viewcam=(\d+)&action=images&frame=left&toggle=1">([^<]+)',
        get(f"{IR}/IMCOMP/in_display.php?toggle=1&action=selectcam&frame=left").decode("latin1"),
    )
    for number, name in cameras:
        shown = get(f"{IR}/IMCOMP/in_display.php?viewcam={number}&action=images&frame=left&toggle=1").decode("latin1")
        folder = re.search(r"PRODS/([A-Za-z0-9_-]+)/", shown)
        time.sleep(0.2)
        if not folder or not re.search(r"INBI|SLI", shown):
            continue
        files = get(f"{IR}/PRODS/{folder[1]}/{folder[1]}THMB.HTM").decode("latin1")
        time.sleep(0.2)
        found = set(re.findall(rf"\b(\w+?(INBI|hSLI)(\d{{5,6}})(NR0)?\.({RAW}))\b", files, re.IGNORECASE))
        for file, scene, iso, plain, _ in sorted(found):
            # The still life comes with and without the camera's own noise reduction; the indoor scene only without.
            if (scene == "hSLI") == bool(plain):
                yield {
                    "camera": slug(name.strip()),
                    "scene": "sli" if plain else "inbi",
                    "iso": int(iso),
                    "url": f"{IR}/PRODS/{folder[1]}/FULLRES/{file}",
                }


def dpreview():
    listing = json.loads(get(DPREVIEW))
    names = {int(key): product["name"] for key, product in listing["products"].items()}
    for image in listing["images"]:
        said = image["attribute_values"]
        whole = said.get("multi-shotmode") in (None, "", "1", "normal") and said.get("rawsize") in (
            None,
            "",
            "1",
            "off",
        )
        if image["raw_file_url"] and whole and str(said.get("iso", "")).isdigit():
            yield {
                "camera": slug(names[image["product_id"]]),
                "scene": slug(said["lighting"]),
                "iso": int(said["iso"]),
                "url": image["raw_file_url"],
            }


def runs(shots):
    """The ISO series among a review's raws, in page order: six or more in a row, each ISO at least 1.5 times the last."""
    series = []
    for iso, url in shots:
        if series and iso >= 1.5 * series[-1][0]:
            series.append((iso, url))
            continue
        if len(series) >= 6:
            yield series
        series = [(iso, url)]
    if len(series) >= 6:
        yield series


def pb():
    for page in (0, 1):
        asked = {"requests": [{"indexName": "cameras", "params": f"query=&hitsPerPage=1000&page={page}"}]}
        for hit in json.loads(get("https://www.photographyblog.com/api/search?indexName=cameras", asked))["results"][0][
            "hits"
        ]:
            try:
                address = urllib.parse.urljoin("https://www.photographyblog.com", hit["url"])
                review = get(address).decode("utf8", "replace").split("Sample RAW Images")[-1]
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                continue
            time.sleep(0.3)
            shots = []
            for block in review.split('<p class="exif">')[1:]:
                iso = re.search(r"ISO\s*(\d+)", html.unescape(block[:400]))
                url = re.search(
                    rf'href="(https?://img\.photographyblog\.com/[^"]+\.({RAW}))"', block[:1500], re.IGNORECASE
                )
                if iso and url:
                    shots.append((int(iso[1]), url[1]))
            for number, series in enumerate(runs(shots)):
                for iso, url in series:
                    yield {
                        "camera": slug(hit["title"].removesuffix(" Review")),
                        "scene": f"review{number}",
                        "iso": iso,
                        "url": url,
                    }


def epz():
    seen = set()
    for page in range(1, 157):
        try:
            listed = get(f"https://www.ephotozine.com/reviews/digital-cameras-29/p-{page}").decode("latin1")
        except urllib.error.URLError:
            continue
        for review in sorted(
            set(re.findall(r'href="(https://www\.ephotozine\.com/article/[a-z0-9-]+-\d+)"', listed)) - seen
        ):
            seen.add(review)
            try:
                body = get(review + "/performance").decode("latin1")
            except urllib.error.URLError:
                continue
            time.sleep(0.3)
            for section in body.split("<h2>")[1:]:
                title = html.unescape(re.sub(r"<[^>]+>", "", section.split("</h2>")[0])).strip()
                if not title.endswith("ISO test images"):
                    continue
                for said, url in re.findall(r'data-description1="([^"]*)"[^>]*?data-description3="([^"]+)"', section):
                    iso = re.search(r"ISO\s*(\d+)", said)
                    if iso and re.search(rf"\.({RAW})$", url, re.IGNORECASE):
                        yield {
                            "camera": slug(title.removesuffix(" ISO test images")),
                            "scene": "studio",
                            "iso": int(iso[1]),
                            "url": url,
                        }


SOURCES = {"ir": ir, "dpreview": dpreview, "pb": pb, "epz": epz}


def ladders(rows):
    """The rows worth fetching: each ladder's clean frame and its noisy ones."""
    found = {}
    for row in rows:
        found.setdefault((row["camera"], row["scene"]), {}).setdefault(row["iso"], row)
    for steps in found.values():
        base = min(steps)
        noisy = [iso for iso in steps if iso >= max(base * RATIO, FLOOR)]
        if noisy:
            yield from (steps[iso] for iso in [base, *sorted(noisy)])


def fetch(row, source):
    path = (
        ROOT
        / source
        / row["camera"]
        / row["scene"]
        / f"{row['iso']}{pathlib.Path(urllib.parse.urlparse(row['url']).path).suffix.lower()}"
    )
    if path.exists():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    for attempt in range(TRIES):
        try:
            part.write_bytes(get(urllib.parse.quote(row["url"], safe=":/?=&")))
            # A challenge page or an error page is small; a raw file never is.
            if part.stat().st_size > 1 << 20:
                part.rename(path)
                return path.stat().st_size
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last = error
        else:
            last = "too small"
        time.sleep(3 * (attempt + 1))
    part.unlink(missing_ok=True)
    print("gave up", row["url"], last, flush=True)
    return 0


def main(what, source):
    ROOT.mkdir(parents=True, exist_ok=True)
    listing = ROOT / f"{source}.jsonl"
    if what == "list":
        rows = []
        with listing.open("w") as file:
            for row in SOURCES[source]():
                rows.append(row)
                file.write(json.dumps(row) + "\n")
                file.flush()
        kept = list(ladders(rows))
        print(
            len(rows), "raws,", len({(r["camera"], r["scene"]) for r in kept}), "ladders,", len(kept), "files to fetch"
        )
        return
    rows = list(ladders(json.loads(line) for line in listing.read_text().splitlines()))
    for done, row in enumerate(rows):
        size = fetch(row, source)
        if size:
            print(f"{done + 1}/{len(rows)} {row['camera']} {row['scene']} {row['iso']} {size >> 20} MB", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main(*sys.argv[1:])
