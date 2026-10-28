"""Every effect that chooses by itself passes over what the speaker took back, on made-up speech.

    .venv/bin/python tests/picker_everywhere.py

The shared picker (`fx/core/moments.py`) never sets a word the speaker took back ("the lively
town, or the lively city" -> city), a first attempt said again or an aside about the recording.
Each check here runs one effect's own choosing code on a transcript written for the test (no clip
is read) and says it lands on what was meant: B-roll and pictures by their tags, the backdrop's
ideas, the magazine's title and lines, the captions' accent, the sound effects' key words, the
split layout's lines, a chart's figures, a list's items, the transitions between ideas. Exits 1 on
any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import amounts as num  # noqa: E402
from fx.core import items as found_items  # noqa: E402
from fx.core import match  # noqa: E402
from fx.core import moments as M  # noqa: E402
from fx.core.words import Word  # noqa: E402
from fx.effects import backdrop, captions, lists, magazine_cover, sound_effects, split_layout, transitions  # noqa: E402

failures: list[str] = []

TOWN = "I slowly made my way while watching, the lively town, or the lively city come to life. I went to my hotel."
RETAKE = ("Want something more editorial? Then just ask for the serif. Want something more editorial? "
          "Then just ask for the serif.")
ASIDE = "Give me a second. Testing, testing. The marketing platform of UGC is the product. I think this audio works better."


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def said(text: str) -> tuple[list[Word], list[tuple[float, float]]]:
    """Each word 0.28 s, the next 0.02 s after it; the voice stops 0.35 s after a comma or a stop."""
    words, quiet, t = [], [(0.0, 0.2)], 0.2
    for tok in text.split():
        words.append(Word(tok, round(t, 3), round(t + 0.28, 3), 0.95))
        t += 0.3
        if tok.endswith((",", ".", "?", "!")):
            quiet.append((round(t - 0.02, 3), round(t + 0.35, 3)))
            t += 0.35
    return words, quiet


def keys(words: list[Word], idx) -> list[str]:
    return [words[i].key for i in idx]


def test_match() -> None:
    words, quiet = said(TOWN)
    back = M.taken_back(words, quiet)
    kws = [k.word.lower() for k in match.keywords(words, quiet, back)]
    check("town" not in kws and "city" in kws, f"match.keywords: city is a thing named, town is not ({kws})")
    items = [SimpleNamespace(path=Path("IMG_1.MOV"), names=(), tags=(("town",), ("street",)), named=True),
             SimpleNamespace(path=Path("IMG_2.MOV"), names=(), tags=(("city",),), named=True)]
    left: dict[int, str] = {}
    cands, _ = match.candidates(match.said(words), quiet, items, {}, "map", left, back)
    landed = {c.item: c.word for c in cands}
    check(0 not in landed and "took them back" in left.get(0, ""),
          f"match.candidates: a clip tagged \"town\" never lands on the town taken back ({landed}, {left})")
    check(landed.get(1, "").lower() == "city", f"match.candidates: the clip tagged \"city\" lands on city ({landed})")
    cands, _ = match.candidates(match.said(words), quiet, items, {"town": 0}, "map", {}, back)
    check(any(c.item == 0 and c.why == "map" for c in cands),
          "match.candidates: a word the person maps lands even where it was taken back, when said nowhere else")
    cands, _ = match.candidates(match.said(words), quiet, items, {}, "map", {}, None)
    check(any(c.item == 0 for c in cands), "match.candidates: without what was taken back, nothing changes")


def test_backdrop() -> None:
    words, quiet = said(TOWN)
    found = backdrop.ideas(words, quiet)
    shown = [m.word.lower() for m in found]
    check(not any("town" in s for s in shown) and any("city" in s for s in shown),
          f"backdrop.ideas: an idea on the lively city, none on the town taken back ({shown})")


def test_magazine() -> None:
    words, quiet = said(TOWN)
    _, info = magazine_cover.topic_word(words, quiet)
    named = [info.get("word", "")] + info.get("runners_up", [])
    check("town" not in [n.lower() for n in named], f"magazine-cover: town is neither the title nor a runner-up ({named})")
    got = [s.text for s in magazine_cover.cover_lines(words, quiet, "hotel")]
    check(not any("town" in g.lower() for g in got), f"magazine-cover: no cover line holds town ({got})")
    words, quiet = said(ASIDE)
    got = [s.text for s in magazine_cover.cover_lines(words, quiet)]
    check(not any("second" in g.lower() or "audio" in g.lower() for g in got),
          f"magazine-cover: no cover line is an aside ({got})")


def test_captions_and_sounds() -> None:
    words, quiet = said(TOWN)
    back = M.taken_back(words, quiet)
    town = next(i for i, w in enumerate(words) if w.key == "town")
    phrase = words[town - 2:town + 1]  # "the lively town,"
    k = captions.key_word(phrase, quiet, back)
    check(k is None or phrase[k].key != "town", f"captions.key_word: \"the lively town,\" is never keyed on town ({k})")
    phrase = words[town + 2:town + 5]  # "the lively city"
    k = captions.key_word(phrase, quiet, back)
    check(k is not None and phrase[k].key == "city", "captions.key_word: \"the lively city\" is keyed on city")
    marked = [words[i].key for i in sound_effects._key_words(words, quiet)]
    check("town" not in marked and "city" in marked, f"sound-effects: a sound on city, none on town ({marked})")
    words, quiet = said(RETAKE)
    marked = sound_effects._key_words(words, quiet)
    half = len(words) // 2
    check(all(i >= half for i in marked), f"sound-effects: no sound in the first attempt ({keys(words, marked)})")
    words, quiet = said(ASIDE)
    marked = [words[i].key for i in sound_effects._key_words(words, quiet)]
    check(not ({"second", "testing", "audio"} & set(marked)), f"sound-effects: no sound in an aside ({marked})")


def test_split_layout() -> None:
    words, quiet = said(TOWN)
    span = split_layout._key_line(words, quiet, 0.0, words[-1].end + 1)
    check(span is not None and "town" not in keys(words, range(span[0], span[1] + 1)),
          f"split-layout: the section's line never holds town ({span and keys(words, range(span[0], span[1] + 1))})")
    words, quiet = said(ASIDE)
    span = split_layout._key_line(words, quiet, 0.0, words[-1].end + 1)
    got = keys(words, range(span[0], span[1] + 1)) if span else []
    check(span is not None and not ({"second", "testing", "audio"} & set(got)),
          f"split-layout: the section's line is never an aside ({got})")


def test_figures() -> None:
    for text, want in (("We grew to ten, sorry, twelve thousand followers this year.", 12000),
                       ("It costs ten dollars, sorry, twelve dollars a month.", 12),
                       ("We went from 2,000 followers, no, 3,000 followers to 15,000 followers.", None)):
        words, quiet = said(text)
        back = M.taken_back(words, quiet)
        kept = [a.value for a in num.spoken(words) if not back.holds(words, a.first, a.last)]
        if want is not None:
            check(kept == [want], f"chart and widgets: \"{text}\" gives {want} only ({kept})")
        else:
            check(2000 not in kept and 3000 in kept, f"chart and widgets: \"{text}\" passes over 2,000 ({kept})")


def test_items() -> None:
    words, quiet = said("Number one is Notion, wait. Let me start again. Number one is Notion. Then Figma.")
    first = [i for i, w in enumerate(words) if w.key == "notion"]
    fake = M.TakenBack(why={words[i]: "a first attempt" for i in range(first[0] - 3, first[0] + 2)})
    given = [lists.Given("Notion"), lists.Given("Figma")]
    timed, _ = found_items.time_items(given, words, [], words[-1].end + 1, back=fake)
    at = {t.n: t.at for t in timed}
    check(abs(at[0] - words[first[1]].start) < 1e-6,
          f"items.time_items: Notion lands where it was said again, not in the first attempt ({at[0]:.2f} s)")
    timed, _ = found_items.time_items(given, words, [], words[-1].end + 1)
    check(abs({t.n: t.at for t in timed}[0] - words[first[0]].start) < 1e-6,
          "items.time_items: without what was taken back, nothing changes")
    words, quiet = said(TOWN)
    back = M.taken_back(words, quiet)
    given = [lists.Given("Town"), lists.Given("City"), lists.Given("Hotel")]
    timed, notes = found_items.time_items(given, words, [], words[-1].end + 1, back=back)
    town = next(t for t in timed if t.n == 0)
    landed_on = [w for w in words if abs(w.start - town.at) < 1e-6]
    check(town.how == "placed" and not any(w in back for w in landed_on) and any("took it back" in n for n in notes),
          f"items.time_items: an item said only where it was taken back drops in between, not on it "
          f"({town.how} at {town.at:.2f} s, {notes})")


def test_transitions() -> None:
    words, quiet = said(ASIDE.replace("the product.", "the product. We built it for creators."))
    clip = SimpleNamespace(info=SimpleNamespace(rate=30.0), words=words, quiet=quiet)
    joins = transitions.sentence_moments(clip)
    before = [j.before.split()[-1] for j in joins]
    check(before == ["product."], f"transitions: the one join between ideas is not beside an aside ({before})")


def main() -> int:
    test_match()
    test_backdrop()
    test_magazine()
    test_captions_and_sounds()
    test_split_layout()
    test_figures()
    test_items()
    test_transitions()
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
