"""The magazine cover's reading of the words, checked on transcripts written for the test (no clip needed).

    .venv/bin/python tests/magazine_cover.py

Checks, on the words as a transcriber writes them (a figure in pieces, a comma where the voice
turns, a slip taken back):

- the title is the thing the clip is about, never a verb, an -ly word or a real magazine's name,
  and the run asks for one when nothing said is strong enough;
- the cover lines are whole phrases the speaker said, cut at a clause or a joining word, never
  ending on a small word ("…the marketing campaign and"), keeping a particle the speaker ended on
  ("what draws people in");
- a figure said in pieces is whole again ("12" "%" is 12%), and a line holding one leads;
- a slip the speaker corrects ("the lively town, or the lively city") is left out, and so is a line
  that only says the title again;
- a small line's bold lead-in never ends on a small word and runs to its strongest word;
- a title that is a real magazine's name is refused.

Exits 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import amounts as num  # noqa: E402
from fx.core.words import Word  # noqa: E402
from fx.effects import magazine_cover as M  # noqa: E402
from fx.effects.options import OptionError  # noqa: E402

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def spoken(text: str, pause_after: tuple[str, ...] = (), start: float = 0.3) -> tuple[list[Word], list[tuple[float, float]]]:
    """Words a third of a second apart from `start`, a longer gap after a sentence or a named word;
    and the quiet stretches between them, as the sound would show them."""
    out, t = [], start
    for tok in text.split():
        out.append(Word(tok, t, t + 0.28, 0.95))
        t += 0.33
        if tok.endswith((".", "?", "!")) or tok in pause_after:
            t += 0.45
    quiet = [(a.end, b.start) for a, b in zip(out, out[1:]) if b.start - a.end > 0.2]
    return out, quiet


def lines(text: str, title: str = "", start: float = 0.3) -> list[str]:
    ws, q = spoken(text, start=start)
    return [s.text for s in M.cover_lines(ws, q, title)]


def main() -> int:
    # The title: the thing named, short enough to set large.
    ws, q = spoken("Here's how to make a video just like this and improve your production quality, all in an "
                   "editor that edits for you.")
    title, info = M.topic_word(ws, q)
    check(title is not None and title.lower() in ("video", "editor"), f"the title is a thing named: {title}")
    ws, q = spoken("Hey, it's Eben, and this is a product I've been building for a while. It helps you edit "
                   "your videos professionally.")
    title, _ = M.topic_word(ws, q)
    check(title == "Eben", f"a name said mid-sentence makes the title: {title}")
    ws, q = spoken("I slowly made my way while watching, the lively")
    title, _ = M.topic_word(ws, q)
    check(title is None, f"nothing strong enough is said, so no title is made up: {title}")
    ws, q = spoken("This saves so much time. Time is everything when you edit videos.")
    title, _ = M.topic_word(ws, q)
    check(title is not None and title.lower() == "videos", f"a real magazine's name is never the title, and a "
          f"verb after \"this\" is not a thing named: {title}")

    # The cover lines: whole phrases, cut where a line can start and end.
    got = lines("Here's how to make a video just like this and improve your production quality, all in an "
                "editor that edits for you.", "video")
    check(got[:2] == ["Improve your production quality", "An editor that edits for you"],
          f"the strongest phrases, cut at the joining word: {got}")
    got = lines("to solve that issue. From the perspective of hooks, what draws people in, and all the way down "
                "to the marketing campaign and", "hooks", start=0.0)  # the clip comes in mid-sentence
    check("What draws people in" in got, f"a particle the speaker ended on stays: {got}")
    check(all(not g.lower().endswith((" and", " the", " to")) for g in got), f"no line ends on a small word: {got}")
    check(not any("solve that issue" in g.lower() for g in got),
          f"a scrap of a sentence begun before the clip is left out: {got}")
    got = lines("I slowly made my way while watching, the lively town, or the lively city come to life. I "
                "eventually went to, of course, my hotel.", "hotel")
    check("The lively city come to life" in got and "The lively town" not in got,
          f"a slip taken back is left out, the line meant kept: {got}")
    check("My hotel" not in got, f"a line that only says the title again is left out: {got}")

    # Figures: whole again, and leading.
    ws, q = spoken("So here is the thing. Last year we went from 2 ,000 followers to 15 ,000, and today we are at "
                   "197 ,000. Our revenue grew from 12 % to 40 % margin. We now have 2 .5 million views.")
    glued = [w for w, _, _ in num.glue(ws)]
    found = M.cover_lines(glued, q, "revenue")
    texts = [s.text for s in found]
    check(any("12% to 40%" in t for t in texts), f"a figure said in pieces is whole again: {texts}")
    check(bool(found) and found[0].figure != "", f"a line holding a figure leads: {texts[:1]} "
          f"(figure {found[0].figure if found else ''!r})")
    check(any(s.figure == "2.5M" and s.label == "views" for s in found),
          f"a figure is written short and with what it counts: {[(s.figure, s.label) for s in found]}")

    # A small line's bold lead-in.
    check(M.split_kicker("An editor that edits for you".split(), 1) == (["An", "editor"], ["that", "edits", "for", "you"]),
          "the lead-in runs to the strongest word")
    check(M.split_kicker("Captions in a click".split(), None) == (["Captions"], ["in", "a", "click"]),
          "a lead-in never ends on a small word")
    check(M.split_kicker("What draws people in".split(), 2) == (["What", "draws", "people", "in"], []),
          "a rest that would be one small word stays with the lead-in")

    # Titles that are a real magazine's name are refused, others kept as given.
    try:
        M.OPTIONS["title"].parse("Vogue")
        check(False, "a real magazine's name is refused as a title")
    except OptionError as e:
        check("real magazine" in str(e), f"a real magazine's name is refused as a title: {e}")
    check(M.OPTIONS["title"].parse("Studio") == "Studio", "a title of your own is kept as given")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
