"""Asides, checked: talk about the recording is found, cut by clean-cut and never picked, and talk
to the viewer that uses the same words is left alone.

    .venv/bin/python tests/asides.py              # the checks; exits 1 if any fails
    .venv/bin/python tests/asides.py --evidence   # every sentence made of asides' forms on the clips, with its verdict

- **Made-up speech.** True asides of each kind are found (alone, at the start, at the end, with
  pauses); these are never found, wherever they sit and with or without pauses: "Testing our
  product", "The audio was great", "Give me a reason", "Cut the pauses", and look-alikes said to
  the viewer ("Hold on. Don't scroll.", "Was that good? Let me know in the comments.", "Give me two
  minutes."). In a take about editing audio, "I think this audio works better" is the talk.
- **33** (`raw-takes/33-demo-day-pt2-talking-only`): its six asides are found, and nothing else:
  "I think I'm going to get some better audio. Give me a second. Testing, testing. All right. I've
  got the audio set up." and, at the end, "I think this audio works better."
- **Every other clip with its words cached** (A, B, C, D, E, F and A-retake): no aside at all.
- **The picker** never sets a word of an aside on 33 (words-behind, headline, prompt-box with one
  prompt and with three), as it never sets a word taken back.
- **clean-cut** cuts each of 33's asides as a cut of its own, listed as an aside; `asides=keep`
  keeps every one; on clips with no aside both settings plan exactly the same cuts.

The clips are read where they are, through `Clip(...).words`, from the words already cached beside
them; nothing is rendered.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import asides as A  # noqa: E402
from fx.core import moments as M  # noqa: E402
from fx.core.words import Word  # noqa: E402

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
    "D": LAB / "test-clips/D-hotel-plain-wall-1080x1920.mp4",
    "E": LAB / "test-clips/E-closeup-long-pauses-1080x1920.mp4",
    "F": LAB / "test-clips/F-train-moving-1080x1920.mp4",
    "33": LAB / "raw-takes/33-demo-day-pt2-talking-only.mp4",
    "A-retake": LAB / "clean-cut/inputs/A-with-retake-and-false-start.mp4",
}
FIXES = {"C": "Urban=Eben"}
ASIDES_33 = ["I think I'm going to get some better audio.", "Give me a second.", "Testing, testing.", "All right.",
             "I've got the audio set up.", "I think this audio works better."]

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def said(text: str, pause: float = 0.0) -> tuple[list[Word], list[tuple[float, float]]]:
    """Made-up speech: each word 0.28 s, the next 0.02 s after it; with `pause`, the voice stops
    that long after each sentence and each comma (the quiet stretches are returned to match)."""
    words, quiet, t = [], [], 0.2
    for tok in text.split():
        words.append(Word(tok, t, t + 0.28, 0.95))
        t += 0.3
        if pause and tok.endswith((",", ".", "?", "!")):
            quiet.append((t - 0.02, t + pause))
            t += pause
    return words, quiet


def found(text: str, pause: float = 0.0) -> list[str]:
    words, quiet = said(text, pause)
    return [a.said for a in A.asides(words, quiet)]


# A take's real talk, put around the sentences checked: nothing in it is an aside.
TALK = ("So the product is called AskEditor. It edits your videos for you, in seconds. "
        "Thousands of people use it every week.")
PLACES = {"alone": "{s}", "opening": "{s} " + TALK, "middle": TALK + " {s} " + TALK, "closing": TALK + " {s}"}

# Asides of each kind, each found at the start of a take and at its end, set off by pauses.
TRUE = ["Testing, testing.", "Testing, one, two, three.", "Mic check.", "Check, check.", "Is this thing on?",
        "Can you hear me okay?", "Give me a second.", "One sec.", "Hang on a sec.", "Okay, give me one second.",
        "Let me start again.", "Sorry, let me do that again.", "Okay, I'll start again.", "Take two.", "Let's go again.",
        "I lost my train of thought.", "I've got the audio set up.", "I think I'm going to get some better audio.",
        "I think this audio works better.", "Let me just check the audio.", "The mic is on now.", "Is the audio okay?",
        "How does the audio sound?", "Hold on, let me fix the mic.", "Are we rolling?", "Is this recording?",
        "Okay, is it recording?", "Cut.", "From the top.", "One more time."]
# Never asides, wherever they sit, with pauses or without. The first four are the brief's.
NEVER = ["Testing our product takes a week.", "The audio was great.", "Give me a reason.", "Cut the pauses.",
         "Today I'll show you how to fix the audio in your videos.", "First, get the audio set up in your editor.",
         "Now I've got the audio set up, let's add music.", "I think the audio works better with music.",
         "Hold on. Don't scroll.", "Wait. This changes everything.", "Hold on a second. Did you know this?",
         "One second. That's all it takes.", "Give me two minutes.", "Give me sixty seconds.",
         "Was that good? Let me know in the comments.", "Check this out.", "Check out the link below.",
         "One more time, the price is nine dollars.", "Let's get started.", "Let me show you again.", "Let me say that again.",
         "Again, the point is speed.", "Again.", "Speed.", "Check.", "Can you hear the difference?",
         "Is this the best editor?", "The camera is the first thing people notice.", "My mic is a Rode.",
         "The audio is great.", "I have a new camera.", "The lighting is beautiful here.", "Sound on.",
         "Turn the sound on.", "We're recording a podcast.", "I'm recording this from my car.",
         "From the top of the mountain you can see the sea.", "Take two eggs and a cup of flour.", "Cut it in half.",
         "I messed up my first launch.", "One, two, three, that's all it takes.", "Bear with me, this is a long one.",
         "Give me one second of your time and I'll show you.", "Hold on to what matters.", "That's a wrap.",
         "Does it work? Yes.", "Testing is the hardest part.", "We're testing a new feature."]
# A remark about a thing the take talks about is the talk: a take about editing audio, about
# sound design, about testing, about cutting.
ON_TOPIC = [
    ("Today I'll show you how I edit the audio in my videos. First, I clean up the audio and cut the noise.",
     "I think this audio works better."),
    ("Here's how to clean up the sound of your videos. Open the sound panel and pick a preset.", "Now, the sound is better."),
    ("Let's pick a sound effect for the swipe. Here's a whoosh sound, and here's a click.", "This sound works better."),
    ("Testing is how we ship. We spend a week testing each feature.", "Testing."),
    ("In this video I cut the pauses out of my take. Every cut lands in silence.", "Cut."),
]


def test_made_up() -> None:
    for s in TRUE:
        for where in ("opening", "closing"):
            got = found(PLACES[where].format(s=s), pause=0.4)
            check(got == [s], f"made up ({where}, pauses): \"{s}\" is an aside (found {got})")
    for s in NEVER:
        bad = [(where, pause, got) for where, form in PLACES.items() for pause in (0.0, 0.4)
               if (got := found(form.format(s=s), pause))]
        check(not bad, f"made up: \"{s}\" is never an aside, alone, opening, middle or closing, with or without "
              f"pauses (found {bad})")
    for talk, s in ON_TOPIC:
        bad = [(pause, got) for pause in (0.0, 0.4) for text in (talk + " " + s, s + " " + talk)
               if (got := found(text, pause))]
        check(not bad, f"made up: in a take about it, \"{s}\" is the talk ({bad})")
    # The take's own aside still counts in a take about the same thing, beside a mic check.
    got = found("Testing, testing. Is the audio okay? " + ON_TOPIC[0][0], pause=0.4)
    check(got == ["Testing, testing.", "Is the audio okay?"], f"made up: a take about audio still opens on a mic "
          f"check and a question about its own audio, found ({got})")
    # The shape: a weak form needs two of opening/closing, pauses and another aside beside it.
    check(found(TALK + " Cut.", pause=0.4) == ["Cut."], "made up: \"Cut.\" at the end, set off by pauses, is an aside")
    check(found(TALK + " Cut. " + TALK + " " + TALK) == [], "made up: \"Cut.\" in the middle with no pause is not")
    # A faint form needs all three: "Hold on." alone opens a pitch, beside a mic check it waits.
    check(found("Hold on. " + TALK, pause=0.4) == [], "made up: \"Hold on.\" alone at the start is not an aside")
    got = found("Hold on. Is this thing on? " + TALK, pause=0.4)
    check(got == ["Hold on.", "Is this thing on?"], f"made up: \"Hold on.\" beside a mic check is ({got})")
    # Words that say nothing go with the asides beside them, and only then.
    got = found("Testing, testing. All right. Give me a second. " + TALK, pause=0.4)
    check(got == ["Testing, testing.", "All right.", "Give me a second."], f"made up: \"All right.\" between two "
          f"asides goes with them ({got})")
    check(found("All right. " + TALK, pause=0.4) == [], "made up: \"All right.\" alone is not an aside")
    # A long sentence is never one, whatever it is made of.
    long = "Okay, give me a second, testing, testing, one, two, three, is this thing on, can you hear me?"
    check(found(long + " " + TALK, pause=0.4) == [], f"made up: a sentence over {A.MOST_WORDS} words is not an aside")
    # The transcriber lost the full stop before an aside (33 heard afresh): the pause still parts them.
    words, quiet = said(TALK + " And we have an inbuilt video editor for all of them")
    t = words[-1].end + 0.8
    quiet.append((words[-1].end, t))
    for tok in "I think this audio works better".split():
        words.append(Word(tok, t, t + 0.28, 0.95))
        t += 0.3
    got = [a.said for a in A.asides(words, quiet)]
    check(got == ["I think this audio works better"], f"made up: an aside the transcriber ran into the sentence "
          f"before, with no full stop but a pause, is found ({got})")
    run_on = "And we have an inbuilt video editor for all of them I think this audio works better"
    check(found(TALK + " " + run_on) == [], "made up: the same words run on with no pause are not cut apart")


def test_pickers_made_up() -> None:
    words, quiet = said("Testing, testing. Give me a second. The marketing platform for creators is growing fast. "
                        "Distribution is the real problem. I think this audio works better.", pause=0.4)
    back = M.taken_back(words, quiet)
    aside = {w for a in A.asides(words, quiet) for w in words[a.first:a.last + 1]}
    check(aside and aside <= set(back.why), "taken_back: every word of an aside is passed over")
    cand, alts = M.pick_word(words, quiet)
    check(all(c.word not in aside for c in [cand] + alts if c),
          f"pick_word: no word of an aside ({[c.word.text for c in [cand] + alts if c]})")
    span = M.pick_line(words, quiet)
    check(span is not None and not any(w in aside for w in words[span[0]:span[1] + 1]),
          f"pick_line: the line is not an aside ({' '.join(w.text for w in words[span[0]:span[1] + 1]) if span else None})")
    starts = M.strong_starts(words, quiet)
    check(not any(words[s.index] in aside for s in starts), f"strong_starts: no moment opens an aside "
          f"({[words[s.index].text for s in starts]})")


def _clip(name: str):
    from fx.core.clip import Clip

    return Clip(CLIPS[name], fix=FIXES.get(name))


def test_clips() -> None:
    clip = _clip("33")
    got = [a.said for a in A.asides(clip.words, clip.quiet)]
    check(got == ASIDES_33, f"33: the six asides are found, and nothing else ({got})")
    for name in ("A", "B", "C", "D", "E", "F", "A-retake"):
        clip = _clip(name)
        got = [a.said for a in A.asides(clip.words, clip.quiet)]
        check(got == [], f"{name}: no aside ({got})")


def _effect(name: str, given: dict):
    from fx import effects
    from fx.effects.options import parse_all

    module = effects.load(name)
    return module, parse_all(module.OPTIONS, effects.entry(name)["options"], given)


def test_pickers_33() -> None:
    clip = _clip("33")
    ws = clip.words
    spans = [(a.start - 0.05, a.end) for a in A.asides(ws, clip.quiet)]
    inside = lambda t: any(lo <= t <= hi for lo, hi in spans)  # noqa: E731
    module, opts = _effect("words-behind", {})
    chosen = module.decide(clip, opts, clip.info.frames / clip.info.rate)[3]
    picked = [(chosen["word"], chosen["said_at"])] + [(r["word"], r["said_at"]) for r in chosen["runners_up"]]
    check(not any(inside(t) for _, t in picked), f"33: words-behind sets no word of an aside ({picked})")
    module, opts = _effect("headline", {})
    hl = module.Headline(clip, opts, (clip.info.width, clip.info.height), clip.info.rate, 1,
                         lambda j: None, lambda j: None, lambda j: None)
    check(hl.spoken and not any(inside(w.start) for w in hl.spoken),
          f"33: the automatic headline is not an aside ({' '.join(w.text for w in hl.spoken)})")
    for given in ({}, {"text": "make it moody | captions on | big title",
                       "answer": "grade look=moody | captions | headline", "when": "auto | auto | auto"}):
        module, opts = _effect("prompt-box", given)
        sends = [p["send"] for p in module.prepare(clip, opts).plan["prompts"]]
        check(not any(inside(s) for s in sends), f"33: prompt-box ({len(sends)} prompt(s)) sends none on an aside: {sends}")
    # magazine-cover sets its title and lines from the heard words, whose edges differ from the timed
    # ones: "I" of the closing aside is heard from 41.96 s, before the aside's timed start
    from fx.effects import magazine_cover

    said = magazine_cover.said_words(clip)
    near = [(a.start - 0.1, a.end + 0.05) for a in A.asides(ws, clip.quiet)]
    left = [w.text for w in said if any(lo <= (w.start + w.end) / 2 <= hi for lo, hi in near)]
    check(not left and len(said) == 103, f"33: magazine-cover takes no word of an aside for its cover ({len(said)} words; {left})")
    module, opts = _effect("freeze-point", {})
    moment = module.decide_moment(clip, opts, clip.info.frames / clip.info.rate)
    check(moment.said is not None and not inside(moment.said[0]),
          f"33: freeze-point stops on no word of an aside (\"{moment.word}\" at {moment.said})")


def test_clean_cut() -> None:
    from fx.effects import clean_cut

    clip = _clip("33")
    asides = A.asides(clip.words, clip.quiet)
    _, opts = _effect("clean-cut", {})
    check(opts["asides"] == "cut", "clean-cut cuts asides unless asked to keep them")
    removals, kept = clean_cut.plan_cuts(clip, opts)
    rows = [r for r in removals if r.reason == "aside"]
    check(len(rows) == len(asides) and [r.words for r in rows] == [a.said for a in asides],
          f"33: each aside is a cut of its own, listed as an aside ({[r.words for r in rows]})")
    check(all(r.detail for r in rows), f"33: each aside's cut says what it is ({[r.detail for r in rows]})")
    words_out = {w for a in asides for w in clip.words[a.first:a.last + 1]}
    check(not (words_out & set(kept)), "33: no word of an aside is kept")
    _, keep = _effect("clean-cut", {"asides": "keep"})
    removals_keep, kept_keep = clean_cut.plan_cuts(clip, keep)
    check(not any(r.reason == "aside" for r in removals_keep) and words_out <= set(kept_keep),
          "33: asides=keep keeps every aside")
    for name in ("D", "E", "A-retake"):
        clip = _clip(name)
        a = [r.as_dict() for r in clean_cut.plan_cuts(clip, opts)[0]]
        b = [r.as_dict() for r in clean_cut.plan_cuts(clip, keep)[0]]
        check(a == b, f"{name}: no aside, so asides=cut and asides=keep plan the same {len(a)} cuts")


def evidence() -> int:
    for name in CLIPS:
        clip = _clip(name)
        rows = A.report(clip.words, clip.quiet)
        print(f"{name}: {sum(r['aside'] for r in rows)} aside(s), {len(rows)} sentence(s) made of asides' forms")
        for r in rows:
            shape = ", ".join(k for k in ("opening", "closing", "set_off", "beside") if r[k]) or "no shape"
            print(f"  {'ASIDE' if r['aside'] else 'kept '} {r['start']:6.2f}-{r['end']:6.2f}  \"{r['said']}\"  "
                  f"[{r['kind']}; {', '.join(r['forms']) or 'nothing said'}; {shape}"
                  f"{'; its words are the talk' if r['topic_said_elsewhere'] else ''}]")
    return 0


def main() -> int:
    if "--evidence" in sys.argv[1:]:
        return evidence()
    test_made_up()
    test_pickers_made_up()
    test_clips()
    test_pickers_33()
    test_clean_cut()
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
