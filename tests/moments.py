"""The core's picker, checked: what the speaker takes back is never what an effect sets by itself.

    .venv/bin/python tests/moments.py        # the checks; exits 1 if any fails

- **Corrections and alternatives**, in made-up speech: each of the four ways a word is taken back
  ("the lively town, or the lively city", "X, I mean Y", "X, sorry, Y", "X — no, Y") is found,
  with commas alone and with pauses alone; the alternatives ("the red one, or the blue one", "a
  train or a bus", "now or never") and the look-alikes ("fast, I mean really fast", an apology, an
  answer to a question) are not.
- **D** (`test-clips/D-hotel-plain-wall`): the take says "the lively town, or the lively city come
  to life". `words-behind` must set CITY, never TOWN, and no automatic headline or prompt may land
  on the words he took back.
- **Retakes**: on `A-retake` (the first sentence said twice, then "From the perspective of" broken
  off and begun again) and on E (a line said three times), no word of a first attempt is ever
  picked, by words-behind, headline or prompt-box; and clean-cut cuts with the core's own finder.

The clips are read where they are, through `Clip(...).words`, from the words already cached beside
them; nothing is rendered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import moments as M  # noqa: E402
from fx.core.words import Word  # noqa: E402

# The test clips are not in this repository. Point FX_LAB at a folder holding them (test-clips/,
# raw-takes/), or put them, or a link to them, at tests/lab (ignored by git).
LAB = Path(os.environ.get("FX_LAB") or Path(__file__).resolve().parent / "lab")
if __name__ == "__main__" and not LAB.exists():
    print("skipped: these checks need the test clips, which are not in this repository. "
          "Set FX_LAB to their folder, or put them at tests/lab.")
    raise SystemExit(0)
D = LAB / "test-clips/D-hotel-plain-wall-1080x1920.mp4"
E = LAB / "test-clips/E-closeup-long-pauses-1080x1920.mp4"
A_RETAKE = LAB / "clean-cut/inputs/A-with-retake-and-false-start.mp4"

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def said(text: str, pause: float = 0.0, commas: bool = True) -> tuple[list[Word], list[tuple[float, float]]]:
    """Made-up speech: each word 0.28 s, the next 0.02 s after it. With `pause`, the voice stops
    that long after each word written with a comma, a dash or a stop (the clip's quiet stretches
    are returned to match); without `commas`, those marks are taken off the words, so only the
    pauses tell where the breaks are."""
    words, quiet, t = [], [], 0.2
    for tok in text.split():
        shown = tok if commas or tok[-1:] in ".?!" else tok.rstrip(",—-")
        if not shown:
            continue
        words.append(Word(shown, t, t + 0.28, 0.95))
        t += 0.3
        if pause and tok.endswith((",", ".", "?", "!", "—", "-")):
            quiet.append((t - 0.02, t + pause))
            t += pause
    return words, quiet


def taken(text: str, **kw) -> list[tuple[str, str]]:
    """What `restated` finds in made-up speech, as (word taken back, word said instead)."""
    words, quiet = said(text, **kw)
    return [(words[x].key, words[y].key) for x, y in M.restated(words, quiet).items()]


CORRECTIONS = [
    ("I slowly made my way while watching, the lively town, or the lively city come to life.", ("town", "city")),
    ("We fly out on Tuesday, I mean Wednesday, at noon.", ("tuesday", "wednesday")),
    ("We fly out on Tuesday, I mean on Wednesday.", ("tuesday", "wednesday")),
    ("The meeting is at three, sorry, four this afternoon.", ("three", "four")),
    ("The wall was red— no, blue all along.", ("red", "blue")),
    ("It costs ten, or rather twelve dollars a month.", ("ten", "twelve")),
    # the correction says its last word again: the word before it is the one replaced
    ("It costs ten dollars, sorry, twelve dollars a month.", ("ten", "twelve")),
    ("We went from 2,000 followers, no, 3,000 followers to 15,000.", ("2,000", "3,000")),
    ("It took three days, I mean four days, to get there.", ("three", "four")),
]
# Alternatives and look-alikes: none of these takes a word back.
NOT_CORRECTIONS = [
    "Do you want the red one, or the blue one?",
    "We can take a train or a bus to the airport.",
    "It's now or never.",
    "Do you want the red car, or the red bus?",
    "Either the lively town or the lively city works for me.",
    "We film in the morning, or in the evening.",
    "It's fast, I mean really fast.",
    "I'm late, sorry, the traffic was bad.",
    "Was it red? No, blue.",
    "It's great. I mean, the product is great.",
    "One or two takes is plenty.",
    "We have two cats, I mean two big cats.",
    "I love Paris, I mean Paris is lovely.",
    "Pick the red car, or the blue car.",
]


def test_restated() -> None:
    for text, want in CORRECTIONS:
        for how, kw in (("commas", {}), ("pauses", {"pause": 0.3, "commas": False})):
            got = taken(text, **kw)
            check(got == [want], f"restated ({how}): \"{text}\" takes \"{want[0]}\" back for \"{want[1]}\" (found {got})")
    for text in NOT_CORRECTIONS:
        for how, kw in (("commas", {}), ("commas and pauses", {"pause": 0.3})):
            got = taken(text, **kw)
            check(got == [], f"restated ({how}): \"{text}\" takes nothing back (found {got})")
    # The brief's three alternatives also stay out of what a picker passes over.
    for text in NOT_CORRECTIONS[:3]:
        words, quiet = said(text, pause=0.3)
        check(not M.taken_back(words, quiet).why, f"taken_back: nothing in \"{text}\" is passed over")


def test_pickers_made_up() -> None:
    # D's correction, made up (with no strong word before it, so the correction decides): the word
    # is CITY, and the town's line is never one.
    words, quiet = said("We saw the lively town, or the lively city come to life.", pause=0.3)
    cand, alts = M.pick_word(words, quiet)
    check(cand is not None and cand.word.key == "city", f"pick_word: \"the lively town, or the lively city\" sets "
          f"CITY ({cand.word.key if cand else None})")
    check(all(c.word.key != "town" for c in [cand] + alts if c), "pick_word: TOWN is not even a runner-up")
    back = M.taken_back(words, quiet)
    check({w.key for w in back.why} == {"the", "lively", "town"},
          f"taken_back: the first \"the lively town\" is passed over ({sorted(w.key for w in back.why)})")
    span = M.pick_line(words, quiet)
    shown = [w.key for w in words[span[0]:span[1] + 1]] if span else []
    check(span is not None and "town" not in shown, f"pick_line: the line is not the one taken back ({' '.join(shown)})")
    starts = [words[s.index].key for s in M.strong_starts(words, quiet)]
    check(all(words[s.index] not in back for s in M.strong_starts(words, quiet)),
          f"strong_starts: no moment opens on the words taken back ({starts})")
    # A first attempt broken off and begun again.
    words, quiet = said("So the product is- so the product is really fast and simple.", pause=0.3)
    back = M.taken_back(words, quiet)
    first = [w for w in words[:4]]
    check(all(w in back for w in first), f"taken_back: the first attempt \"So the product is-\" is passed over "
          f"({[w.text for w in back.why]})")
    cand, alts = M.pick_word(words, quiet)
    check(all(c.word not in first for c in [cand] + alts if c), "pick_word: no word of the first attempt is picked "
          f"({[(c.word.text, round(c.word.start, 2)) for c in [cand] + alts if c]})")
    span = M.pick_line(words, quiet)
    check(span is not None and span[0] >= 4, f"pick_line: the line starts after the first attempt ({span})")


def _effect(name: str, clip, given: dict):
    from fx import effects
    from fx.effects.options import parse_all

    module = effects.load(name)
    return module, parse_all(module.OPTIONS, effects.entry(name)["options"], given)


def _words_behind(clip, given: dict) -> dict:
    module, opts = _effect("words-behind", clip, given)
    return module.decide(clip, opts, clip.info.frames / clip.info.rate)[3]


def _headline_words(clip) -> list[Word]:
    """The spoken words the automatic headline sets."""
    module, opts = _effect("headline", clip, {})
    hl = module.Headline(clip, opts, (clip.info.width, clip.info.height), clip.info.rate, 1,
                         lambda j: None, lambda j: None, lambda j: None)
    return list(hl.spoken)


def _sends(clip, given: dict) -> list[float]:
    module, opts = _effect("prompt-box", clip, given)
    return [p["send"] for p in module.prepare(clip, opts).plan["prompts"]]


def test_d() -> None:
    from fx.core.clip import Clip

    clip = Clip(D)
    ws = clip.words
    hits = {ws[x].key: ws[y].key for x, y in M.restated(ws, clip.quiet).items()}
    check(hits == {"town": "city"}, f"D: restated finds \"town\" taken back for \"city\", and nothing else ({hits})")
    back = M.taken_back(ws, clip.quiet)
    town = [w for w in ws if w in back]
    check([w.key for w in town] == ["the", "lively", "town"], f"D: the first \"the lively town\" is passed over "
          f"({[(w.text, round(w.start, 2)) for w in town]})")
    chosen = _words_behind(clip, {})
    check(chosen.get("word", "").lower() == "city", f"D: words-behind sets CITY ({chosen.get('word')} at {chosen.get('said_at')} s)")
    check(all(r["word"].lower() != "town" for r in chosen.get("runners_up", [])),
          f"D: TOWN is not a runner-up ({[r['word'] for r in chosen.get('runners_up', [])]})")
    check(any(p["word"].lower() == "town" for p in chosen.get("passed_over", [])), "D: the plan says TOWN was passed over")
    spoken = _headline_words(clip)
    check(spoken and not any(w in back for w in spoken), f"D: the automatic headline holds nothing taken back "
          f"({' '.join(w.text for w in spoken)})")
    lo, hi = min(w.start for w in town), max(w.end for w in town)
    for given in ({}, {"text": "make it moody | captions on | big title",
                       "answer": "grade look=moody | captions | headline", "when": "auto | auto | auto"}):
        sends = _sends(clip, given)
        check(not any(lo - 0.05 <= s <= hi for s in sends), f"D: prompt-box ({len(sends)} prompt(s)) sends none on "
              f"\"the lively town\" ({lo:.2f}-{hi:.2f} s): {sends}")


def test_retakes() -> None:
    from fx.core.clip import Clip
    from fx.effects import clean_cut

    check(clean_cut.find_retakes is M.find_retakes, "clean-cut cuts the retakes the core finds (one finder)")
    for path, given in ((A_RETAKE, {}), (E, {}), (E, {"when": "19"})):
        clip = Clip(path)
        back = M.taken_back(clip.words, clip.quiet)
        name = "A-retake" if path == A_RETAKE else "E"
        what = " ".join(f"{k}={v}" for k, v in given.items()) or "defaults"
        check(len(back.why) > 0, f"{name}: the first attempts are found ({len(back.why)} words)")
        chosen = _words_behind(clip, given)
        said_at = {round(w.start, 2) for w in back.why}
        picked = [(chosen["word"], round(chosen["said_at"], 2))] + [(r["word"], r["said_at"]) for r in chosen["runners_up"]]
        check(not any(round(t, 2) in said_at for _, t in picked), f"{name}: words-behind ({what}) picks no word of a "
              f"first attempt ({picked})")
        if given:
            continue
        spoken = _headline_words(clip)
        check(spoken and not any(w in back for w in spoken), f"{name}: the automatic headline holds nothing said "
              f"again later ({' '.join(w.text for w in spoken)})")
        spans = [(w.start - 0.05, w.end) for w in back.why]
        for prompts in ({}, {"text": "make it moody | captions on | big title",
                             "answer": "grade look=moody | captions | headline", "when": "auto | auto | auto"}):
            sends = _sends(clip, prompts)
            check(not any(a <= s <= b for s in sends for a, b in spans),
                  f"{name}: prompt-box ({len(sends)} prompt(s)) sends none inside a first attempt: {sends}")
    # E's line said three times: from 19 s on, only the third take's word may be set.
    clip = Clip(E)
    chosen = _words_behind(clip, {"when": "19"})
    check(chosen["said_at"] >= 27.9, f"E: words-behind from 19 s sets the word of the take kept "
          f"({chosen['word']} at {chosen['said_at']} s; the kept take starts at 27.95 s)")


def main() -> int:
    test_restated()
    test_pickers_made_up()
    test_d()
    test_retakes()
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
