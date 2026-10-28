"""post-copy's counting and checking, on fixed text: no video, no model.

    .venv/bin/python tests/post_copy.py

The counts are the platforms' own ways of counting (X's weights, TikTok's UTF-16 units, YouTube's
bytes), and the check is run on a post.md written here, once filled right and once with faults it
must name.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.effects import post_copy as pc  # noqa: E402

failures = 0


def check(what: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("ok   " if ok else "FAIL ") + what + (f"  ({detail})" if detail and not ok else ""))
    failures += 0 if ok else 1


# -- counting
for text, want in [("hello world", 11), ("\U0001F468‍\U0001F3A4", 2), ("\U0001F64B\U0001F3FD", 2),
                   ("\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466", 2), ("日本", 4),
                   ("see askeditor.com now", 31), ("https://example.com/a/very/long/path", 23),
                   ("\U0001F1EC\U0001F1E7", 2), ("1️⃣", 2), ("“quoted” — dash", 15)]:
    check(f"X counts {text!r} as {want}", pc.x_length(text) == want, str(pc.x_length(text)))
check("TikTok counts an emoji as two UTF-16 units", pc.utf16("\U0001F600a") == 3)
check("hashtags are found and their trailing punctuation dropped",
      pc.hashtags("Go #train, #HuaHin. and #chicken_patty! not#this") == ["#train", "#HuaHin", "#chicken_patty"])


# -- the check, on a file written here
class FakeInfo:
    width, height, rate, frames, duration = 1080, 1920, 30.0, 3000, 100.0


class FakeClip:
    info = FakeInfo()
    path = "/videos/talk.mp4"
    unsure: list = []


def post(blocks: dict, chapters: str = "") -> str:
    out = ["# Post copy: talk.mp4", ""]
    if chapters:
        out += ["<!-- youtube.chapters -->", chapters, "<!-- /youtube.chapters -->", ""]
    for key, _, _ in pc.BLOCKS:
        out += [f"<!-- {key} -->", blocks.get(key, f"{pc.FILL} write it"), f"<!-- /{key} -->", ""]
    return "\n".join(out)


CHAPTERS = "00:00 Why\n00:31 How\n01:05 What it cost"
GOOD = {
    "reels.title": "Three ways to edit faster",
    "reels.caption": "Three ways to edit faster\nCut the pauses, caption it, lay music under it.\n#editing #reels",
    "tiktok.title": "Edit faster",
    "tiktok.caption": "Edit faster\nThree things I do on every video. #editing",
    "shorts.title": "Three ways to edit your videos faster",
    "shorts.description": "Cut the pauses, add captions, lay music under your voice. #editing",
    "youtube.title": "Three ways to edit your videos faster, with no expensive app",
    "youtube.description": "What I do on every video.\n\n" + CHAPTERS + "\n\n#editing #video",
    "linkedin.title": "Three things I do on every video.",
    "linkedin.post": "Three things I do on every video.\n\nCut the pauses. Caption it. Lay music under it.",
    "hashtags": "#editing #reels #video",
    "short.post": "Three ways to edit faster: cut the pauses, caption it, lay music under your voice.",
}
with tempfile.TemporaryDirectory() as tmp:
    f = Path(tmp) / "talk.post.md"
    opts = {"titles": "auto", "min_part": "auto", "platforms": "all", "check": True}
    f.write_text(post(GOOD, CHAPTERS))
    got = pc.PostCopy(FakeClip(), opts).check(f)
    check("a right file passes", not got["failed"], got["results"])
    bad = dict(GOOD)
    bad["reels.caption"] = "Something else\n#a #b #c #d #e #f"
    bad["youtube.title"] = "x" * 101
    bad["youtube.description"] = "No chapters <here> #fine #not$fine"
    bad["short.post"] = "y" * 281
    del bad["linkedin.post"]
    f.write_text(post(bad, CHAPTERS))
    got = pc.PostCopy(FakeClip(), opts).check(f)
    res = got["results"]
    check("a faulty file fails", got["failed"])
    for want in ("6 of 5 hashtags", "101 of 100", "< or > is not allowed", "#not$fine", "chapter lines must be in it",
                 "281 of 280", "linkedin.post: not written yet", "should start with the title"):
        check(f"the check names: {want}", want in res, res)
    f.write_text(post(GOOD, "00:00 Why\n00:05 Too soon\n01:05 What it cost"))
    got = pc.PostCopy(FakeClip(), opts).check(f)
    check("a chapter under 10 s is caught", got["failed"] and "lasts 5 s" in got["results"], got["results"])

print(f"\n{failures} failed" if failures else "\nall passed")
sys.exit(1 if failures else 0)
