"""Make the small films and posters the askeditor.com/skills pages play, from the example films.

    .venv/bin/python tools/web_films.py                 # every effect that has an example film
    .venv/bin/python tools/web_films.py clean-cut chart # just these
    .venv/bin/python tools/web_films.py --lab DIR       # the folder holding catalogue/films/ (default FX_LAB, or tests/lab)

For each `effects/<id>/` with a film at `<lab>/catalogue/films/<id>.mp4` it writes, into
`<lab>/catalogue/films/web/`:

- `<id>.mp4`: 540 px wide, no sound, about 8 s, H.264 with faststart, under 1 MB. A film longer than
  that keeps the 8 s centred on the effect (the seconds `catalogue/films/TRIMS.md` gives), or, when
  the effect runs longer than 8 s, the 8 s from just before it starts. The quality is lowered a step
  at a time until the file fits;
- `<id>.jpg`: the poster, 540 px wide, from the frame of the film that `effects/<id>/poster.jpg` was
  taken from (found by comparing every frame with it), or that poster scaled when no frame matches;
- for `clean-cut`, also `clean-cut-before.mp4` (run E's raw take) and `clean-cut-after.mp4` (run E's
  render), each whole, so the page shows what was cut;
- `index.json`: one entry per effect (id, film, poster, bytes, seconds). An effect with no film
  (it writes words, or reports) still gets its poster, with `film` null.

The films stay out of the repository: they are media, kept in the lab. One ffmpeg job runs at a time.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
LAB = Path(os.environ.get("FX_LAB") or PACK / "tests" / "lab")  # the example films are not in this repository

WIDTH = 540
SECONDS = 8.0
LIMIT = 1_000_000  # bytes: under 1 MB
MAX_FPS = 30
CRF_FIRST, CRF_STEP, CRF_LAST = 24, 2, 40
MATCH = 14.0  # mean difference (0-255) above which no frame is taken to be the poster's
# Clean Cut's before and after: run E, a close-up with long pauses and a retake (31 s -> 11 s)
CLEAN_CUT_BEFORE = "test-clips/E-closeup-long-pauses-1080x1920.mp4"
CLEAN_CUT_AFTER = "clean-cut/E.mp4"


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": num / den, "seconds": float(j["format"]["duration"])}


def _secs(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def trims(path: Path) -> dict[str, tuple[float, float] | None]:
    """{id: (start, end) of the effect in the example film's own seconds}, or None for the whole film."""
    out: dict[str, tuple[float, float] | None] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*`([a-z0-9-]+)`\s*\|([^|]*)\|([^|]*)\|", line)
        if not m:
            continue
        name, kept, shown = m.group(1), m.group(2).strip(), m.group(3).strip()
        kept_from = 0.0 if kept.startswith("whole") else _secs(kept)[0]
        nums = _secs(shown.split("(")[0])
        if "whole" in shown or len(nums) < 2:
            out[name] = None
        else:
            out[name] = (nums[0] - kept_from, nums[1] - kept_from)
    return out


def window(seconds: float, effect: tuple[float, float] | None) -> tuple[float, float]:
    """(start, length) of the part kept: the whole film when it is about 8 s already."""
    if seconds <= SECONDS + 1.0:
        return 0.0, seconds
    if effect is None:
        start = 0.0
    else:
        a, b = max(0.0, effect[0]), min(seconds, effect[1])
        start = (a + b) / 2 - SECONDS / 2 if b - a <= SECONDS else a - 1.0
    start = min(max(0.0, start), seconds - SECONDS)
    return round(start, 3), SECONDS


def encode(src: Path, dst: Path, start: float, length: float) -> tuple[int, int]:
    """Writes dst under LIMIT, lowering the quality until it fits; returns (bytes, crf)."""
    info = probe(src)
    vf = f"scale={WIDTH}:-2:flags=lanczos" + (f",fps={MAX_FPS}" if info["fps"] > MAX_FPS + 0.5 else "")
    tmp = dst.with_suffix(".part.mp4")
    for crf in range(CRF_FIRST, CRF_LAST + 1, CRF_STEP):
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{length}", "-i", str(src), "-an",
               "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-profile:v", "high", "-movflags", "+faststart", "-threads", "4", str(tmp)]
        subprocess.run(cmd, check=True)
        size = tmp.stat().st_size
        if size < LIMIT:
            tmp.replace(dst)
            return size, crf
    raise SystemExit(f"{dst.name} is still {tmp.stat().st_size} bytes at crf {CRF_LAST}")


def frames(src: Path, w: int, h: int, fps: float = 10.0):
    """Yields (seconds, grey frame of w x h) through the film, fps a second."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"fps={fps},scale={w}:{h},format=gray",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = 0
    assert p.stdout is not None
    while True:
        buf = p.stdout.read(w * h)
        if len(buf) < w * h:
            break
        yield n / fps, np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.float32)
        n += 1
    p.wait()


def poster_frame(src: Path, poster: Path) -> tuple[float | None, float]:
    """The second of the film that looks most like the poster, and how far off it is (0-255)."""
    import cv2

    img = cv2.imread(str(poster), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 255.0
    w, h = 90, round(90 * img.shape[0] / img.shape[1])
    want = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    best, best_t = 255.0, None
    for t, f in frames(src, w, h):
        d = float(np.abs(f - want).mean())
        if d < best:
            best, best_t = d, t
    return best_t, best


def write_poster(name: str, src: Path | None, dst: Path) -> str:
    poster = PACK / "effects" / name / "poster.jpg"
    t, off = poster_frame(src, poster) if src is not None and poster.exists() else (None, 255.0)
    if t is not None and off <= MATCH:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t}", "-i", str(src), "-frames:v", "1",
                        "-vf", f"scale={WIDTH}:-2:flags=lanczos", "-q:v", "3", str(dst)], check=True)
        return f"frame at {t:.1f} s (off by {off:.1f})"
    if not poster.exists():
        return "no poster"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(poster), "-vf", f"scale={WIDTH}:-2:flags=lanczos",
                    "-q:v", "3", str(dst)], check=True)
    return "the repo's poster, scaled" + (f" (nearest frame off by {off:.1f})" if off < 255 else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/web_films.py")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--lab", type=Path, default=LAB)
    args = ap.parse_args(argv)
    films = args.lab / "catalogue" / "films"
    web = films / "web"
    web.mkdir(parents=True, exist_ok=True)
    cuts = trims(films / "TRIMS.md")
    names = sorted(p.parent.name for p in (PACK / "effects").glob("*/effect.json"))
    chosen = args.names or names
    unknown = set(chosen) - set(names)
    if unknown:
        raise SystemExit(f"no such effect: {', '.join(sorted(unknown))}")
    index_path = web / "index.json"
    index = {e["id"]: e for e in (json.loads(index_path.read_text()) if index_path.exists() else [])}
    for name in chosen:
        src = films / f"{name}.mp4"
        entry: dict = {"id": name, "film": None, "poster": None, "bytes": None, "seconds": None}
        if src.exists():
            info = probe(src)
            start, length = window(info["seconds"], cuts.get(name))
            size, crf = encode(src, web / f"{name}.mp4", start, length)
            entry.update(film=f"{name}.mp4", bytes=size, seconds=round(probe(web / f"{name}.mp4")["seconds"], 2))
            print(f"{name}: {start:.1f}-{start + length:.1f} s of {info['seconds']:.1f} s, crf {crf}, {size} bytes")
        how = write_poster(name, src if src.exists() else None, web / f"{name}.jpg")
        if (web / f"{name}.jpg").exists():
            entry["poster"] = f"{name}.jpg"
        print(f"{name}: poster: {how}")
        if name == "clean-cut":
            for part, rel in (("before", CLEAN_CUT_BEFORE), ("after", CLEAN_CUT_AFTER)):
                s = args.lab / rel
                if not s.exists():
                    print(f"clean-cut: no {part} film at {s}", file=sys.stderr)
                    continue
                out = web / f"clean-cut-{part}.mp4"
                size, crf = encode(s, out, 0.0, probe(s)["seconds"])
                entry[part] = {"film": out.name, "bytes": size, "seconds": round(probe(out)["seconds"], 2)}
                print(f"clean-cut {part}: whole {probe(s)['seconds']:.1f} s, crf {crf}, {size} bytes")
        index[name] = entry
    index_path.write_text(json.dumps([index[k] for k in sorted(index)], indent=2) + "\n")
    print(f"{index_path}: {len(index)} effects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
