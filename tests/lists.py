"""Lists said in order, read from the words: checked on transcripts written for the test (no clip needed).

    .venv/bin/python tests/lists.py

`tests/fixtures/list-words.json` holds one transcript per way of saying a list: a countdown ("Number
five, Canva … and number one is going to be AskEditor"), steps ("Step one, find a window … and
finally, hit record"), ordinals ("First … Second … Lastly"), a chain of "then"s with no opener, a
journey told as it happened ("First of all, I went … I then boarded … and then eventually I got off
… I eventually went to, of course, my hotel"), "the first thing you need is", a bare count ("Five,
Canva."), steps said done ("Done.", "Check.", "and that's it"), a run after a colon and a run of
actions with a last "and". One more holds sentences that must never be read as a list: a "then"
after "if", "the first time", "give me a second", "one, two, three, test", "a number of" and
"then I was like". Each list must come out whole, in order, with its numbers, each item's words as
said without what only led into them, and each landing on its own first word. Exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import items as I  # noqa: E402
from fx.core.words import Word  # noqa: E402

FIXTURE = PACK / "tests/fixtures/list-words.json"

# (take, [(item, number said or 0)])
EXPECTED = {
    "countdown": [("Canva", 5), ("CapCut", 4), ("Premiere Pro", 3), ("DaVinci Resolve", 2), ("AskEditor", 1)],
    "steps": [("Find a window", 1), ("Clip the mic on your collar", 2), ("Frame yourself at eye level", 3),
              ("Hit record", 0)],
    "ordinals": [("Good lighting", 1), ("Decent mic", 2), ("Tripod", 3), ("Editor that edits for you", 0)],
    "then_chain": [("Drop your clip in", 0), ("Type add captions", 0), ("Pick a look", 0)],
    "journey": [("Went to the train station", 1), ("Boarded the train", 0), ("Got off the train", 0),
                ("Went to my hotel", 0)],
    "things": [("Good mic", 1), ("Light", 2), ("Story people care about", 3)],
    "bare_count": [("Canva", 5), ("CapCut", 4), ("Premiere", 3), ("Resolve", 2), ("AskEditor", 1)],
    "done_words": [("Plug in the mic", 1), ("Open the app", 2), ("Press record", 3)],
    "series": [("Notion", 0), ("Figma", 0), ("Canva", 0)],
    "series_actions": [("Picks your best line", 0), ("Cuts you out", 0), ("Pops you right out of the frame", 0)],
    "not_lists": [],
}
# The word each step is said done on, in the done_words take.
DONE = ["Done.", "Check.", "that's"]


def words_of(items: list[dict]) -> list[Word]:
    return [Word(i["w"], float(i["s"]), float(i["e"])) for i in items]


def main() -> int:
    takes = json.loads(FIXTURE.read_text())["takes"]
    failures = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    for name, want in EXPECTED.items():
        words = words_of(takes[name])
        got = I.in_order(words)
        pairs = [(s.text, s.group) for s in got]
        check(pairs == want, f"{name}: {pairs}" + ("" if pairs == want else f", wanted {want}"))
        if not got:
            continue
        check(all(got[k].at < got[k + 1].at for k in range(len(got) - 1)), f"{name}: in the order said")
        # each lands on its own first word: the first word of the item as said
        firsts = [words[s.anchor].clean.strip(",.") for s in got]
        check(all(f.lower() == s.text.split()[0].lower() for f, s in zip(firsts, got)),
              f"{name}: each lands on its first word ({firsts})")
        check(all(s.at == words[s.anchor].start and s.marker_at <= s.at for s in got),
              f"{name}: its marker comes first, the item lands on its word")

    # Steps said done, and the step after a "then" ticking the one before.
    words = words_of(takes["done_words"])
    steps = I.in_order(words)
    done = I.done_after(words, steps)
    said = [words[d[1]].text if d else None for d in done]
    check(said == DONE, f"done_words: each step said done on {said}")

    # Only inside a list does a "then" in the middle of a sentence start an item.
    loose = words_of(takes["not_lists"][:0]) + [Word(w, 0.5 + 0.3 * k, 0.75 + 0.3 * k) for k, w in enumerate(
        "We had lunch and then we went for a walk.".split())]
    check(I.in_order(loose) == [], "a lone \"and then\" in a story is not a list")

    # Shorter forms keep whole words, and a cut never ends on a word that leads somewhere.
    forms = I.shorter("Clip the mic on your collar")
    check(forms == ["Clip the mic on your collar", "Clip mic on collar"], f"shorter forms: {forms}")
    cut = I.cut_to("Drop your video into the subtitle generator", lambda s: len(s) <= 22)
    check(cut == "Drop your video…", f"cut between words, never after 'into': {cut!r}")

    engine(check)
    print(f"\n{len(failures)} failure(s)" if failures else "\nall passed")
    return 1 if failures else 0


def engine(check) -> None:
    """The lists engine (fx/effects/lists.py) on the same transcripts: when a row is still, when a
    step is ticked, how a countdown is numbered, how items are read, and words never cut inside."""
    from fx.effects import lists as L

    takes = json.loads(FIXTURE.read_text())["takes"]
    fps = 30.0
    # A row is still and whole on the first frame of its word, and moving on the frame before.
    words = words_of(takes["steps"])
    rows, heard, source = L.rows_of(words, [], fps, words[-1].end + 1, [])
    card = L.Card(L.CardSpec("tick", 0.6, 0, 0, False, False), rows, "", "solid", (0, 200, 120), fps, (1080, 1920), None)
    ok = all(r.frame == L.frame_of(r.at, fps) and card.row_state(r, r.frame / fps) == (1.0, 0.0)
             and card.row_state(r, (r.frame - 1) / fps)[1] > 0 for r in rows)
    check(ok, f"steps: each row still on frame {[r.frame for r in rows]}, moving on the frame before")
    # Ticks: where the next step starts (its marker), never on the frame the next row lands, the last
    # where its sentence ends.
    L.done_times(rows, words, fps, words[-1].end + 1, words[-1].end + 1)
    marks = [rows[k + 1].marker_at for k in range(len(rows) - 1)]
    check(all(abs(rows[k].done - max(marks[k], rows[k].land + 0.45)) < 1.0 / fps + 1e-6 or rows[k].done_frame < rows[k + 1].frame
              for k in range(len(rows) - 1)), f"steps: ticked where the next step starts ({[round(r.done, 2) for r in rows]})")
    check(all(rows[k].done_frame < rows[k + 1].frame for k in range(len(rows) - 1)),
          "steps: no tick on the frame the next row lands")
    check(rows[-1].done_how == "its sentence ends", f"steps: the last ticks where its sentence ends ({rows[-1].done_how})")
    words = words_of(takes["done_words"])
    rows, _, _ = L.rows_of(words, [], fps, words[-1].end + 1, [])
    L.done_times(rows, words, fps, words[-1].end + 1, words[-1].end + 1)
    check([r.done_how for r in rows] == ["said done"] * 3, f"done_words: ticked where said done ({[r.done_how for r in rows]})")
    # A countdown counts down in the order said; numbers said win.
    words = words_of(takes["things"])
    rows, _, _ = L.rows_of(words, [], fps, 30, [])
    for r in rows:
        r.number = 0
    L.number_rows(rows, True, [])
    check([r.number for r in rows] == [3, 2, 1], f"a countdown numbers the first said highest: {[r.number for r in rows]}")
    words = words_of(takes["countdown"])
    rows, _, _ = L.rows_of(words, [], fps, 30, [])
    L.number_rows(rows, False, [])
    check([r.number for r in rows] == [5, 4, 3, 2, 1], f"numbers said win: {[r.number for r in rows]}")
    # Items given: the short forms.
    got = L.parse_items("Lighting; Sound@4.2; 5: Canva", PACK, 10)
    check([(g.text, g.at, g.number) for g in got] == [("Lighting", None, 0), ("Sound", 4.2, 0), ("Canva", None, 5)],
          f"items given: {[(g.text, g.at, g.number) for g in got]}")
    got = L.parse_items("Notion, Figma, Canva", PACK, 10)
    check([g.text for g in got] == ["Notion", "Figma", "Canva"], "items given with commas when no ; is used")
    # Words made to fit are whole words.
    from fx.core import text as textmod

    said = "Drop your video into the subtitle generator"
    for room in (900, 700, 500, 400, 300):
        got = L.fit_words(said, room, 50, "poppins-bold", 2)
        if got is None:
            continue
        shown = got[0].rstrip("…").split()
        check(all(w in said.split() for w in shown) and textmod.advance(got[0], "poppins-bold", got[1]) <= room,
              f"{room} px: {got[0]!r} (whole words, inside the room)")


if __name__ == "__main__":
    sys.exit(main())
