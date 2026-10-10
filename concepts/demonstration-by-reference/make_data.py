"""Generate every session and render its frames.

  uv run python make_data.py --items 0-11 --sizes 64 128 256 500
  uv run python make_data.py --items 100-101 --sizes 64 128 256 500   # development items, never scored

Writes data/sessions/<name>.json (frames, what was said on each, ground truth)
data/frames/<name>/<frame id>.webp (lossless WebP, not committed), the same at
half width and height in data/frames_half/ (for inline_low), and
data/frames_manifest.json with the sha256 of every frame, so a re-render can be
checked byte for byte (--verify). The frames every run used were rendered on
macOS arm64; a render on Linux x86_64 with the same Pillow, FreeType and libwebp
versions differs in about 1.3% of pixels (glyph edges), so the runs on the cloud
machine used frames copied from the laptop, checked against the manifest. "sanity-<item>" is the sanity condition: three tasks,
each on its own page with the words on the same frame, nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from demoref.arms import half_frame
from demoref.render import encode, render
from demoref.session import build_session, load, save, world_for

HERE = Path(__file__).parent
DATA = HERE / "data"


def name_of(item: int, size: int | None) -> str:
    return f"sanity-{item:02d}" if size is None else f"item{item:02d}-n{size}"


def render_session(name: str) -> dict[str, str]:
    """Render a session's frames, and each at half size for inline_low."""
    s = load(DATA / "sessions" / f"{name}.json")
    world = world_for(s)
    out = DATA / "frames" / name
    half = DATA / "frames_half" / name
    out.mkdir(parents=True, exist_ok=True)
    half.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in s.frames:
        data = encode(render(world, f.spec), "webp_lossless")
        small = half_frame(data)
        (out / f"{f.id}.webp").write_bytes(data)
        (half / f"{f.id}.webp").write_bytes(small)
        hashes[f.id] = hashlib.sha256(data).hexdigest()
        hashes[f"{f.id}-half"] = hashlib.sha256(small).hexdigest()
    return hashes


def verify() -> None:
    """Check every frame on disk against the manifest (the frames every run used)."""
    manifest = json.loads((DATA / "frames_manifest.json").read_text())
    def path(s: str, f: str) -> Path:
        return DATA / "frames_half" / s / f"{f[:-5]}.webp" if f.endswith("-half") else DATA / "frames" / s / f"{f}.webp"

    bad = [(s, f) for s, frames in manifest.items() for f, h in frames.items()
           if hashlib.sha256(path(s, f).read_bytes()).hexdigest() != h]
    n = sum(len(v) for v in manifest.values())
    print(f"{n} frames, {len(bad)} differ from the manifest")
    if bad:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="0-11")
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 500])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="only check the frames on disk against the manifest")
    a = ap.parse_args()
    if a.verify:
        verify()
        return
    names = []
    lo, hi = (int(x) for x in a.items.split("-"))
    for item in range(lo, hi + 1):
        s = build_session(item, None, simple=True)
        save(s, DATA / "sessions" / f"{name_of(item, None)}.json")
        names.append(name_of(item, None))
        for size in a.sizes:
            s = build_session(item, size)
            save(s, DATA / "sessions" / f"{name_of(item, size)}.json")
            names.append(name_of(item, size))
    with ProcessPoolExecutor(a.workers) as ex:
        manifest = dict(zip(names, ex.map(render_session, names)))
    path = DATA / "frames_manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(manifest)
    path.write_text(json.dumps(old, indent=0, sort_keys=True))
    print(f"{len(names)} sessions, {sum(len(v) for v in manifest.values())} frames")


if __name__ == "__main__":
    main()
