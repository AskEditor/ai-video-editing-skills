"""The B-roll cutaways' rules, checked on a transcript written for the test (no clip needed).

    .venv/bin/python tests/broll_cutaways.py

A short travel talk, timed word by word, with real pauses after each sentence, is matched against a
folder of clips: some named for what they show, one listed in a tags file, some with only a camera's
number for a name. Every cutaway must start on the first moment of a word, come back between two
words, run one to three seconds (or straight into the next), leave you on screen in the opening
second and between cutaways, and cover no more than the share allowed. A clip whose name is never
said must be left out with its reason; camera-named clips must take the strongest things named, in
folder order; describing words, counting words and the speaker's own name must never carry one.
It also checks the map, the file-name words, the tags file and the plurals. Exits 1 on any failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core.words import Word  # noqa: E402
from fx.effects import broll_cutaways as BC  # noqa: E402

TALK = ("Hi, it's Sam. Last week I took the train to the coast. The station was packed, but the sea was "
        "calm. We walked along the harbour, then ate at a tiny cafe. It was a good trip, and the food was "
        "better than I expected. Next time I want to see the mountains.")


def timed(text: str) -> tuple[list[Word], list[tuple[float, float]], float]:
    """Words a third of a second each, a breath between words, a real pause after each sentence."""
    words, quiet, t = [], [], 0.6
    quiet.append((0.0, 0.6))
    for raw in text.split():
        length = 0.18 + 0.035 * len(raw.strip(".,"))
        words.append(Word(raw, round(t, 3), round(t + length, 3)))
        t += length
        gap = 0.45 if raw.endswith(".") else 0.0
        if gap:
            quiet.append((round(t, 3), round(t + gap, 3)))
        t += gap
    quiet.append((round(t, 3), round(t + 0.8, 3)))
    return words, quiet, round(t + 0.8, 3)


def clip(name: str, seconds: float = 12.0, tags: tuple = ()) -> BC.Broll:
    return BC.Broll(f"/broll/{name}", "video", 1080, 1920, seconds, names=BC.name_words(name), tags=tags)


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    # what a file name says, plurals, the tags file, the map
    check(BC.name_words("IMG_5922.MOV") == (), "a camera's number says nothing")
    check(BC.name_words("rice-fields_mountains.mov") == ("rice", "fields", "mountains"), "words from a file name")
    check(BC.name_words("LakeAtDusk.mp4") == ("lake", "dusk"), "words from a joined-up name")
    check(BC.name_words("final-v2-broll-clip-copy.mp4") == (), "editing words in a name are not what it shows")
    check([BC.stem(w) for w in ("cities", "trains", "boxes", "glass", "bus", "mountains")]
          == ["city", "train", "box", "glass", "bus", "mountain"], "plurals meet their singulars")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "tags.txt").write_text("# notes\nIMG_1.MOV: sea, harbour boats\nIMG_2: cafe\n")
        tags = BC.read_tags(tmp)
        check(tags.get("img_1.mov") == (("sea",), ("harbour", "boat")) and tags.get("img_2") == (("cafe",),),
              "tags.txt: one clip a line, phrases separated by commas")
    check(BC.parse_map('{"lake": "lake.mp4", "the city": "roofs.mov"}') == [("lake", "lake.mp4"),
                                                                          ("the city", "roofs.mov")],
          "map as JSON")
    check(BC.parse_map("lake=lake.mp4; city=roofs.mov, sea=sea.mp4") == [("lake", "lake.mp4"), ("city", "roofs.mov"),
                                                                         ("sea", "sea.mp4")], "map as word=clip pairs")
    try:
        BC.parse_map("lake lake.mp4")
        check(False, "a map pair without = is refused")
    except ValueError:
        check(True, "a map pair without = is refused")

    # the things named
    words, quiet, seconds = timed(TALK)
    named = {w.word.lower() for w in BC.keywords(words, quiet)}
    for thing in ("train", "station", "harbour", "cafe", "mountains"):
        check(thing in named, f"“{thing}” is a thing named")
    for not_thing in ("good", "better", "time", "sam", "walked", "expected"):
        check(not_thing not in named, f"“{not_thing}” never carries a cutaway")

    # the cutaways
    pool = [
        clip("train-station.mp4"),  # named: lands on "train"
        clip("IMG_0001.MOV", tags=(("sea",),)),  # tagged: lands on "sea"
        clip("snowy-forest.mp4"),  # never said: left out
        clip("DJI_0002.MP4"),  # camera-named: the strongest things left, in folder order
        clip("DJI_0003.MP4"),
        clip("DJI_0004.MP4", seconds=1.2),  # too short to carry one
    ]
    cuts, left, notes = BC.decide(words, quiet, None, pool, {}, "keywords", "auto", 0.5, seconds)
    starts = {round(w.start, 3) for w in words}
    by_clip = {c.clip: c for c in cuts}
    check(0 in by_clip and by_clip[0].word.lower() == "train", "the clip named for the train lands on “train”")
    check(1 in by_clip and by_clip[1].word.lower() == "sea", "the clip tagged sea lands on “sea”")
    check(2 not in by_clip and "never said" in left.get(2, ""), "a clip whose name is never said is left out, saying why")
    check(5 not in by_clip and "long" in left.get(5, ""), "a clip too short is left out, saying why")
    numbered = sorted((c for c in cuts if c.clip in (3, 4)), key=lambda c: c.start)
    check([c.clip for c in numbered] == sorted(c.clip for c in numbered) and len(numbered) >= 1,
          "camera-named clips go in folder order")
    check(all(c.why == "keyword" for c in numbered), "camera-named clips go on the things named")
    check(all(round(c.start, 3) in starts for c in cuts), "every cutaway starts on the first moment of a word")
    check(all(BC.inside_word(c.end, words) is None for c in cuts), "every cutaway comes back between two words")
    check(all(BC.MIN_LEN - 1e-6 <= c.end - c.start <= BC.MAX_LEN + BC.END_ROOM for c in cuts),
          "every cutaway runs one to three seconds")
    check(all(c.start >= BC.OPENING - 1e-6 for c in cuts if c.why == "keyword"), "the opening second stays on you")
    ordered = sorted(cuts, key=lambda c: c.start)
    gaps_ok = all(b.start - a.end >= BC.MIN_SPEAKER - 1e-6 or (a.chained and abs(a.end - b.start) < 1e-6)
                  for a, b in zip(ordered, ordered[1:]))
    check(gaps_ok, "you are on screen at least a second between cutaways, unless one runs straight into the next")
    check(not any(a.chained and b.chained and a.priority <= 1 and b.priority <= 1
                  for a, b in zip(ordered, ordered[1:])), "never three cutaways in a row on keywords alone")
    check(BC.Timing.total(cuts) <= 0.5 * seconds + 1e-6, "the B-roll covers no more than half the film")
    tight, _, _ = BC.decide(words, quiet, None, pool, {}, "keywords", "auto", 0.15, seconds)
    check(BC.Timing.total(tight) <= 0.15 * seconds + 1e-6 and len(tight) < len(cuts), "max_share holds")
    mapped, left2, _ = BC.decide(words, quiet, None, pool, {"mountains": 2}, "map", "auto", 0.5, seconds)
    check(any(c.clip == 2 and c.why == "map" and c.word.lower() == "mountains" for c in mapped),
          "map puts a clip on the word it names")
    check(all(c.why in ("map", "name", "tags") for c in mapped) and "on=map" in left2.get(3, ""),
          "on=map uses only named clips")
    timed_cuts, _, _ = BC.decide(words, quiet, None, pool, {}, "2.6, 9.4", "auto", 0.5, seconds)
    check([c.clip for c in timed_cuts] == [0, 1] and all(round(c.start, 3) in starts for c in timed_cuts),
          "times take the clips in folder order, each moved onto a word")
    print(f"\n{len(cuts)} cutaways: " + "; ".join(f"{c.start:.2f}-{c.end:.2f} “{c.word}” → {pool[c.clip].label}"
                                                 for c in ordered))
    if failures:
        print(f"\n{len(failures)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
