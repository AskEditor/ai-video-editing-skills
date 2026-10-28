"""The parts of a talk, checked on fixed transcripts.

    .venv/bin/python tests/parts.py

Each case is a talk written out with a time for every word and the quiet between them, the way the
transcriber and the sound would give them. The checks: where the parts begin, that none is shorter
than asked, that a pause between "I" and "got off" is never a boundary, that a signpost said inside a
sentence counts for little, that titles and times given by a person are kept, the YouTube chapter
lines and their rules, and the small parsers. Nothing is rendered and no model runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import parts  # noqa: E402
from fx.core.words import Word  # noqa: E402

STEP = 0.34  # seconds a word


def talk(script: str) -> tuple[list[Word], list[tuple[float, float]], float]:
    """Words and quiet from a script where `[1.2]` is a pause of that many seconds."""
    words, quiet, t = [], [], 0.3
    for token in script.split():
        if token.startswith("[") and token.endswith("]"):
            pause = float(token[1:-1])
            quiet.append((t, t + pause))
            t += pause
            continue
        words.append(Word(token, round(t, 3), round(t + STEP * 0.9, 3)))
        t += STEP
    return words, quiet, round(t + 0.5, 3)


SIGNPOSTED = """
Today I will show you three ways to edit your videos faster. [1.2]
First, cut the pauses out of your talking. Dead air makes people scroll away, and pauses add up to
whole seconds in every clip you post. [0.9]
Second, put captions on every video. Most people watch with the sound off, so captions carry your
message when nobody can hear it. [1.0]
Finally, lay music under your voice. A quiet track keeps the energy up between sentences and makes
the whole reel feel finished.
"""

# The same talk at the length of a short YouTube video: every part well over ten seconds.
LONGER = """
Today I will show you three ways to edit your videos faster, and none of them needs an expensive app
or a long course to learn. [1.2]
First, cut the pauses out of your talking. Dead air makes people scroll away, and pauses add up to
whole seconds in every clip you post. Watch your own video back and you will hear them everywhere. [0.9]
Second, put captions on every video. Most people watch with the sound off, so captions carry your
message when nobody can hear it, and a big clear word on screen holds the eye. [1.0]
Finally, lay music under your voice. A quiet track keeps the energy up between sentences and makes
the whole reel feel finished, as long as it dips whenever you speak.
"""

ONE_IDEA = """
Coffee is the reason I get up in the morning. I grind the beans fresh, I weigh them, and I pour the
water slowly over them. [0.2] The smell fills the kitchen before the coffee is even ready, and that
smell is half of the pleasure. [0.25] A good cup of coffee needs nothing else, not sugar and not milk.
"""

# A pause between "I" and "got off", and a signpost said inside a sentence.
RUN_ON = """
We took the early train to the coast and it was packed with people going to the beach. [0.3]
After a long ride I [1.6] got off the train at the last stop and walked down to the sea. [0.3]
The hotel was small but the room looked right at the water, which was great, [0.6] the next step was
dinner by the harbour. [1.4]
Now the food there is something else. Fresh fish grilled on the street, and bowls of noodles for two
dollars, and fruit everywhere you look.
"""

failures = 0


def check(what: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("ok   " if ok else "FAIL ") + what + (f"  ({detail})" if detail and not ok else ""))
    failures += 0 if ok else 1


def starts(found):
    return [round(p.start, 2) for p in found]


def word_at(words, t):
    """The first word said at or after `t`."""
    return next(w.text for w in words if w.start >= t - 1e-6)


# -- a talk that announces its parts
w, q, d = talk(SIGNPOSTED)
found = parts.find(w, q, d)
check("signposted talk: four parts", len(found) == 4, str([p.title for p in found]))
check("signposted talk: each part begins on its signpost",
      [word_at(w, p.start) for p in found[1:]] == ["First,", "Second,", "Finally,"],
      str([word_at(w, p.start) for p in found[1:]]))
check("signposted talk: titles are the parts' own subjects",
      all(any(k in p.title.lower() for k in ks)
          for ks, p in zip((("faster", "ways"), ("pause",), ("caption",), ("music", "track")), found)),
      str([p.title for p in found]))
check("signposted talk: every part at least min_part long, the opening at least its own shorter minimum",
      found[0].seconds >= parts.opening_min(parts.auto_min_part(d)) - 1e-6
      and all(p.seconds >= parts.auto_min_part(d) - 1e-6 for p in found[1:]),
      str([round(p.seconds, 1) for p in found]))
found = parts.find(w, q, d, min_part=12)
check("min_part=12 keeps every part after the opening 12 s or longer", all(p.seconds >= 12 - 1e-6 for p in found[1:]),
      str([round(p.seconds, 1) for p in found]))

# -- one idea stays one part
w, q, d = talk(ONE_IDEA)
found = parts.find(w, q, d)
check("one idea: one part", len(found) == 1, str([(p.start, p.title) for p in found]))

# -- a run-on story
w, q, d = talk(RUN_ON)
cands = parts.weigh(w, parts.candidates(w, q), window=8)
got_off = next(i for i, x in enumerate(w) if x.text == "got")
check("never a boundary between 'I' and 'got off'", all(c.after != got_off - 1 for c in cands))
step = next(c for c in parts.candidates(w, q) if w[c.after + 1].text == "the" and w[c.after + 2].text == "next")
check("a signpost inside a sentence counts for little", step.strength < 0.5, f"strength {step.strength}")
found = parts.find(w, q, d)
check("run-on story: the food part begins on 'Now'", any(word_at(w, p.start) == "Now" for p in found),
      str([word_at(w, p.start) for p in found]))

# -- what a person gives
w, q, d = talk(SIGNPOSTED)
count, given = parts.parse_titles("Why | Pauses | Captions")
found = parts.find(w, q, d, count=count, titles=given)
check("three titles make three parts", len(found) == 3 and [p.title for p in found] == ["Why", "Pauses", "Captions"],
      str([p.title for p in found]))
check("given titles are marked as given", all(p.given for p in found))
count, given = parts.parse_titles("Intro | 0:05 Cut the pauses | 0:16 Captions | 0:30 Music")
found = parts.find(w, q, d, count=count, titles=given)
check("titles with times pin their parts within 2.5 s",
      [abs(p.start - t) <= 2.5 for p, t in zip(found[1:], (5, 16, 30))] == [True] * 3, str(starts(found)))
check("a pinned part starts on a real boundary", word_at(w, found[1].start) == "First,", word_at(w, found[1].start))
count, given = parts.parse_titles("2")
found = parts.find(w, q, d, count=count)
check("titles=2 makes two parts", len(found) == 2)

# -- parsers
check("parse_titles auto", parts.parse_titles("auto") == (None, []))
check("parse_titles count", parts.parse_titles("4") == (4, []))
n, g = parts.parse_titles("Intro | 0:31 The train | 65s Food | 1:02:03 Late")
check("parse_titles times", n == 4 and [x.at for x in g] == [None, 31.0, 65.0, 3723.0]
      and [x.title for x in g] == ["Intro", "The train", "Food", "Late"], str(g))
check("signpost 'and finally'", parts.signpost(["and", "finally", "we"])[1] == 1.0)
check("signpost 'so' is slight", parts.signpost(["so", "the", "hotel"])[1] < 0.5)
check("signpost 'I then' moves a story on", parts.signpost(["i", "then", "boarded"])[1] == 0.6)
check("signpost 'the second thing'", parts.signpost(["the", "second", "thing", "is"])[1] == 1.0)
check("signpost 'number three'", parts.signpost(["number", "three"])[1] == 1.0)

# -- chapters as YouTube reads them
check("stamp under an hour", parts.stamp(75.9) == "01:15")
check("stamp over an hour", parts.stamp(3725, hours=True) == "1:02:05")
w, q, d = talk(LONGER)
found = parts.find(w, q, d, min_part=parts.YOUTUBE_SHORTEST, first_min=parts.YOUTUBE_SHORTEST)
lines = parts.chapter_lines(found, d)
check("chapters start at 00:00", lines[0].startswith("00:00 "), lines[0])
check("chapters with YouTube's minimum pass YouTube's rules", len(lines) >= 3 and parts.check_chapters(lines, d) == [],
      str(lines) + " " + str(parts.check_chapters(lines, d)))
w, q, d = talk(SIGNPOSTED)
short = parts.chapter_lines(parts.find(w, q, d), d)
check("a reel's short parts are caught as too short for YouTube", parts.check_chapters(short, d) != [])
bad = parts.check_chapters(["00:03 Intro", "00:08 Two"], 15)
check("chapters: a late start, too few and too short are all caught",
      len(bad) == 4 and any("00:00" in b for b in bad) and any("three" in b for b in bad), str(bad))

print(f"\n{failures} failed" if failures else "\nall passed")
sys.exit(1 if failures else 0)
