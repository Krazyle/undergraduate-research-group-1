# /// script
# requires-python = ">=3.12"
# ///
"""Build the data for the prototype slide: illustrative posts in Marikina and the map tiles under them.

Run with `uv run scripts/marikina_posts.py` from the defense-v1 folder. The tiles are saved in the
deck, so the slide's map works without an internet connection.
"""

import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "marikina-posts.json"
MAP = ROOT / "public" / "map"

# OpenFreeMap's Positron style, a light map from OpenStreetMap data. The slide shows the whole city,
# which needs tiles only up to zoom 13
STYLE_URL = "https://tiles.openfreemap.org/styles/positron"
ZOOMS = range(11, 14)
GLYPH_RANGES = ("0-255", "256-511", "8192-8447")
WEST, SOUTH, EAST, NORTH = 121.05, 14.595, 121.15, 14.695

# The centre of each barangay, from OpenStreetMap's Nominatim search, with a weight for its share of
# the posts. The barangays along the Marikina River, which flood first, post the most
BARANGAYS = {
    "Barangka": (14.63279, 121.08221, 1.5),
    "Calumpang": (14.62413, 121.09325, 1.5),
    "Concepcion Uno": (14.65105, 121.10412, 1.5),
    "Concepcion Dos": (14.64059, 121.12082, 0.7),
    "Fortune": (14.66329, 121.12736, 0.7),
    "Industrial Valley": (14.62710, 121.08000, 1.5),
    "Jesus de la Peña": (14.63609, 121.09211, 2),
    "Malanday": (14.65016, 121.09375, 3),
    "Marikina Heights": (14.65011, 121.11714, 0.7),
    "Nangka": (14.67291, 121.10921, 2.5),
    "Parang": (14.65886, 121.11173, 1),
    "San Roque": (14.62573, 121.10288, 1),
    "Santa Elena": (14.63278, 121.09749, 1),
    "Santo Niño": (14.64006, 121.09684, 2),
    "Tañong": (14.63455, 121.08583, 1.5),
    "Tumana": (14.66041, 121.10078, 3),
}

# The categories the side panel lists, each with its share of the posts and the share of its posts
# that name a place. Sympathy posts seldom say where they are
CATEGORIES = [
    ("Requests or urgent needs", 0.25, 0.9),
    ("Rescue, volunteering, or donation effort", 0.17, 0.85),
    ("Displaced people and evacuations", 0.14, 0.9),
    ("Infrastructure and utility damage", 0.12, 0.9),
    ("Sympathy and support", 0.18, 0.3),
    ("Other categories", 0.14, 0.6),
]
POSTS = 640

# A post is placed about 250 m around its barangay's centre
SPREAD = 0.0022


def tile(lon, lat, zoom):
    n = 2**zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


generator = random.Random(7)
names = list(BARANGAYS)
weights = [BARANGAYS[name][2] for name in names]
located, unlocated = [], [0] * len(CATEGORIES)
for _ in range(POSTS):
    category = generator.choices(range(len(CATEGORIES)), [share for _, share, _ in CATEGORIES])[0]
    if generator.random() > CATEGORIES[category][2]:
        unlocated[category] += 1
        continue
    lat, lon, _ = BARANGAYS[generator.choices(names, weights)[0]]
    located.append([round(generator.gauss(lat, SPREAD), 5), round(generator.gauss(lon, SPREAD), 5), category])

OUTPUT.parent.mkdir(exist_ok=True)
OUTPUT.write_text(json.dumps({
    "bounds": [[SOUTH, WEST], [NORTH, EAST]],
    "categories": [name for name, _, _ in CATEGORIES],
    "posts": located,
    "unlocated": unlocated,
}, ensure_ascii=False))
print(f"Wrote {OUTPUT} with {len(located)} posts on the map and {sum(unlocated)} without a location")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "thesis-defense-slides"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def save(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(url))
        time.sleep(0.05)


# The style points at the saved copies, with {base} filled in by the slide, since the map needs full URLs
style = json.loads(fetch(STYLE_URL))
tiles = json.loads(fetch(style["sources"]["openmaptiles"]["url"]))["tiles"][0]
fonts = {font for layer in style["layers"] for font in layer.get("layout", {}).get("text-font", [])}
sprite = style["sprite"]
style["sources"] = {"openmaptiles": {"type": "vector", "tiles": ["{base}map/tiles/{z}/{x}/{y}.pbf"], "minzoom": 0, "maxzoom": max(ZOOMS), "attribution": '<a href="https://openfreemap.org">OpenFreeMap</a> © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}}
style["layers"] = [layer for layer in style["layers"] if layer.get("source") in (None, "openmaptiles")]
style["glyphs"] = "{base}map/fonts/{fontstack}/{range}.pbf"
style["sprite"] = "{base}map/sprites/ofm"
MAP.mkdir(parents=True, exist_ok=True)
(MAP / "style.json").write_text(json.dumps(style))

for zoom in ZOOMS:
    left, top = tile(WEST, NORTH, zoom)
    right, bottom = tile(EAST, SOUTH, zoom)
    for x in range(left - 1, right + 2):
        for y in range(top - 1, bottom + 2):
            save(tiles.format(z=zoom, x=x, y=y), MAP / "tiles" / str(zoom) / str(x) / f"{y}.pbf")
for font in fonts:
    for glyphs in GLYPH_RANGES:
        save(f"https://tiles.openfreemap.org/fonts/{urllib.parse.quote(font)}/{glyphs}.pbf", MAP / "fonts" / font / f"{glyphs}.pbf")
for suffix in ("", "@2x"):
    for kind in ("json", "png"):
        save(f"{sprite}{suffix}.{kind}", MAP / "sprites" / f"ofm{suffix}.{kind}")
print(f"Saved the map in {MAP}")
