"""The core, checked by eye on the test clips: the face track, the person mask, and a word laid
behind the person through the API the behind-you effects use.

    .venv/bin/python tests/core.py [--clips A,B,C] [--model person|person-parts] [--out DIR]

For each clip it writes `<clip>-core.jpg` to a new temp folder (printed), or to `--out DIR` when
given, so the lab's `core/` sheets change only when `--out` names that folder. Five frames across
the clip, with the head's box (what text keeps clear of) drawn on, the mask beside each, and the same frame
with a giant word behind the person. It prints the mask's stored range (the inside must be exactly
1, or whatever is behind shows through the speaker) and how long each part took.
"""

from __future__ import annotations

import os
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import compose, person, text, video  # noqa: E402
from fx.core.clip import Clip  # noqa: E402

# The test clips are not in this repository. Point FX_LAB at a folder holding them (test-clips/,
# raw-takes/), or put them, or a link to them, at tests/lab (ignored by git).
LAB = Path(os.environ.get("FX_LAB") or Path(__file__).resolve().parent / "lab")
if __name__ == "__main__" and not LAB.exists():
    print("skipped: these checks need the test clips, which are not in this repository. "
          "Set FX_LAB to their folder, or put them at tests/lab.")
    raise SystemExit(0)
CLIPS = {
    "A": LAB / "test-clips/A-demo-day-talk-clean-8s-720x1280.mp4",
    "B": LAB / "test-clips/B-airport-headroom-1080x1920.mp4",
    "C": LAB / "test-clips/C-lake-path-hard-1080x1920.mp4",
    "33": LAB / "raw-takes/33-demo-day-pt2-talking-only.mp4",
}


def _box(img: np.ndarray, r, colour, width: int = 3) -> None:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in (r.x0, r.y0, r.x1, r.y1))
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    img[y0:y0 + width, x0:x1] = colour
    img[max(0, y1 - width):y1, x0:x1] = colour
    img[y0:y1, x0:x0 + width] = colour
    img[y0:y1, max(0, x1 - width):x1] = colour


def check(name: str, path: Path, model: str, out: Path) -> dict:
    import cv2
    from PIL import Image

    clip = Clip(path)
    info = clip.info
    t0 = time.time()
    track = clip.face
    t_face = time.time() - t0
    t0 = time.time()
    masks = clip.person(model)
    t_mask = time.time() - t0
    rng = person.check_range(masks)

    picks = np.linspace(0.08, 0.92, 5) * (info.frames - 1)
    word = text.draw_word("BEHIND", "anton", int(info.height * 0.16), fill=(255, 212, 0))
    rows = []
    for i in picks.astype(int):
        frame = video.read_frame(info, int(i))
        m = masks.at(int(i))
        alpha = masks.full(m, (info.width, info.height), guide=frame)
        layer = compose.blank_layer(info.width, info.height)
        head = track.head(int(i))
        y = (head.y0 if head else info.height * 0.3) - word.ascent * 0.35
        compose.onto_layer(layer, word.sprite, info.width / 2 - word.sprite.w / 2, y - word.oy + word.ascent * 0.5)
        behind = compose.behind_person(frame, layer[0], layer[1], alpha)
        boxed = frame.copy()
        if head:
            _box(boxed, head, (255, 64, 64), max(2, info.width // 240))
        crown = masks.crown(m, head.x0, head.x1) if head else None
        if crown is not None:
            boxed[int(crown):int(crown) + max(2, info.width // 240), :] = (64, 200, 255)
        mask_rgb = np.dstack([alpha * 255] * 3).astype(np.uint8)
        tile = np.hstack([boxed, mask_rgb, behind])
        scale = 360 / info.width
        rows.append(cv2.resize(tile, (int(tile.shape[1] * scale), int(tile.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    sheet = np.hstack(rows) if len(rows) <= 2 else np.vstack([np.hstack(rows[:3]), np.hstack(rows[3:] + [np.zeros_like(rows[0])] * (6 - len(rows)))])
    target = out / f"{name}-core{'' if model == person.MODEL else '-' + model}.jpg"
    Image.fromarray(sheet).save(target, quality=86)
    return {"clip": name, "face_found": round(track.found_share, 3), "face_s": round(t_face, 1),
            "mask_s": round(t_mask, 1), "mask_range": rng, "sheet": str(target)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="A,B,C")
    ap.add_argument("--model", default=person.MODEL, choices=sorted(person.STRETCH))
    ap.add_argument("--out", default=None, help="where the sheets go (default: a new temp folder)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="fx-core-"))
    print("sheets in", out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in args.clips.split(","):
        r = check(name, CLIPS[name], args.model, out)
        good = r["mask_range"]["max"] >= 1.0 and r["mask_range"]["min"] <= 0.0
        ok &= good
        print(("ok  " if good else "FAIL"), r)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
