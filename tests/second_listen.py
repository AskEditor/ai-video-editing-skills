"""The second listen, checked on the clip it invented words on and the one it exists for.

    .venv/bin/python tests/second_listen.py              # the checks; exits 1 if any fails
    .venv/bin/python tests/second_listen.py --evidence   # every word it hears on the test clips, with the values
    .venv/bin/python tests/second_listen.py --calibrate  # what the thresholds are set from (about ten minutes)

- **D** (plain wall): the take says "the lively town, or the lively city … I eventually went to,
  uh, of course, my hotel" (an earlier transcript of the whole take, with its context, agrees).
  Heard again on their own, the stretches after "town" and "hotel" came back as "Town car." and
  "Hotel.", and the edge of the stretch after "went" as a second "went". None of them may be
  shown, by the words or by any caption look; each must be kept, marked unsure.
- **A-retake** (A spliced from itself): "From the perspective of" broken off and begun again,
  written once by the first listen inside a stretched "the". The second listen must still bring
  the false start back, and clean-cut must cut it and keep the restart.
- **Figures** (`chart/inputs/spoken-numbers*.mp4`, a computer voice reading lines with numbers):
  heard again out of their sentences, "19.5%" came back as "19 17 .5", "500,000" as "500 100
  ,000." and "40%" as "40 30". Every figure must reach the words exactly as the first listen
  wrote it, and still read as the same amount.

Each clip is copied, with the words cached beside it, into a fresh folder, and this version's
second listen is taken out of the copy. What is left was written before this version (on D, the
earlier second listen's "Town car." and "Hotel." are in it), so each run also proves a stale
cache is heard again and otherwise left as it was.

`--evidence` reads the clips where they are (adding this version's second listen to their caches)
and prints, for every stretch heard again, each word with its probability, its segment's
no_speech_prob and avg_logprob, the voice under it and the verdict. `--calibrate` hears short
stretches of the same clips the way the second listen does, over speech (each word heard there is
real when the first listen heard it there too) and over the clips' quiet (every word heard there
is invented), and prints how the two compare on each value `words.judge` reads.
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import words as W  # noqa: E402
from fx.core.clip import Clip  # noqa: E402

# The test clips are not in this repository. Point FX_LAB at a folder holding them (test-clips/,
# raw-takes/), or put them, or a link to them, at tests/lab (ignored by git).
LAB = Path(os.environ.get("FX_LAB") or Path(__file__).resolve().parent / "lab")
if __name__ == "__main__" and not LAB.exists():
    print("skipped: these checks need the test clips, which are not in this repository. "
          "Set FX_LAB to their folder, or put them at tests/lab.")
    raise SystemExit(0)
D = LAB / "test-clips/D-hotel-plain-wall-1080x1920.mp4"
A_RETAKE = LAB / "clean-cut/inputs/A-with-retake-and-false-start.mp4"
NUMBERS = [LAB / "chart/inputs/spoken-numbers-2.mp4", LAB / "chart/inputs/spoken-numbers.mp4"]
# The figures each must keep, as the first listen wrote them, and the digits misread on a second listen.
FIGURES = {"spoken-numbers-2.mp4": ([["19", ".5%,"], ["500", ",000,"]], {"17", "100"}, [(19.5, "%"), (500000.0, "")]),
           "spoken-numbers.mp4": ([["40"], ["2", ".5", "million"]], {"30"}, [(2500000.0, "")])}
EVIDENCE = [LAB / f"test-clips/{n}.mp4" for n in (
    "A-demo-day-talk-clean-8s-720x1280", "B-airport-headroom-1080x1920", "C-lake-path-hard-1080x1920",
    "D-hotel-plain-wall-1080x1920", "E-closeup-long-pauses-1080x1920", "F-train-moving-1080x1920")] + [
    LAB / "raw-takes/33-demo-day-pt2-talking-only.mp4", A_RETAKE]

# What is said on D, word by word, as the effects should read it.
D_SAID = ("I slowly made my way while watching uh the lively town or the lively city come to life um "
          "I eventually went to uh of course my hotel").lower().split()

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        failures.append(what)


def fresh_copy(src: Path, folder: Path) -> Path:
    """The clip and what is cached beside it (its words and face), in `folder`, with the words'
    cache made stale: this version's second listen is taken out of the copy, so the run must hear
    the stretches again (an earlier version's results, if the cache has any, stay in it)."""
    for suffix in (".words.json", ".face.json", ".person.mkv"):
        cached = src.with_name(src.stem + suffix)
        if cached.exists():
            shutil.copy2(cached, folder / cached.name)
    words = folder / (src.stem + ".words.json")
    if words.exists():
        data = json.loads(words.read_text())
        data.pop(W.SECOND_KEY, None)
        words.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    shutil.copy2(src, folder / src.name)
    return folder / src.name


def keys(words) -> list[str]:
    return [w.key for w in words]


def test_d(folder: Path) -> Path:
    from fx.effects import captions, entry
    from fx.effects.options import parse_all

    path = fresh_copy(D, folder)
    before = json.loads(path.with_name(path.stem + ".words.json").read_text())
    clip = Clip(path)
    shown = keys(clip.words)
    check(shown == D_SAID, f"D: the words shown are what was said ({' '.join(shown)})")
    for word in ("town", "went", "hotel"):
        check(shown.count(word) == 1, f"D: \"{word}\" is shown once (shown {shown.count(word)} times)")
    check("car" not in shown, "D: no \"car\" is shown")
    unsure = {w.text: w.unsure for w in clip.unsure}
    for word in ("Town", "car.", "Hotel."):
        check(word in unsure, f"D: \"{word}\" is kept, marked unsure ({unsure.get(word, 'missing')})")
    for look in captions.LOOKS:
        opts = parse_all(captions.OPTIONS, entry("captions")["options"], {"look": look})
        plan = captions.prepare(clip, opts).plan
        on_cards = [W.normalize(w["w"]) for c in plan["cards"] for w in c["words"]]
        expect = [k for k in D_SAID if k not in W.HESITATIONS]
        check(on_cards == expect, f"D: captions {look}: the cards show only what was said")
        check({u["w"] for u in plan["unsure"]} >= {"Town", "car.", "Hotel."},
              f"D: captions {look}: the plan lists the unsure words")
    after = json.loads(path.with_name(path.stem + ".words.json").read_text())
    check(W.SECOND_KEY not in before and W.SECOND_KEY in after,
          f"D: the stale cache was heard again and this version's second listen added ({W.SECOND_KEY})")
    check(after.get("second_listen") == before.get("second_listen") and after["words"] == before["words"],
          "D: what the cache held before is still there, unchanged")
    return path


# Each effect with options that make it read the words that have an unsure twin on D ("town",
# "went", "hotel"), besides its defaults.
PROBES = {
    "before-after": [{"at": "town"}], "cta-card": [{"when": "my hotel"}], "grade": [{"start": "town"}],
    "headline": [{"when": "went to"}], "prompt-box": [{"when": "eventually went"}],
    "tier-list": [{"items": "S: town; A: city; B: hotel"}], "words-behind": [{"when": "my hotel"}],
}


def test_every_effect(path: Path) -> None:
    """Every effect decides the same on D with its unsure words kept beside the words as with them
    gone altogether: nothing unsure is shown, laid out round, timed to or cut on."""
    from fx import effects
    from fx.effects.options import parse_all

    kept = Clip(path)
    gone = Clip(path)
    gone.__dict__["heard"] = W.sure(kept.heard)  # as if the second listen had never heard them
    check(bool(kept.unsure) and not gone.unsure, f"D: {len(kept.unsure)} unsure words, none in the copy without them")
    for name in effects.names():
        module = effects.load(name)
        if not hasattr(module, "prepare"):  # a check, not a render (qc-check, review-notes): nothing it lays out
            continue
        for given in [{}] + PROBES.get(name, []):
            opts = parse_all(module.OPTIONS, effects.entry(name)["options"], given)
            decided = []
            for clip in (kept, gone):
                try:
                    plan = {k: v for k, v in module.prepare(clip, opts).plan.items() if k != "unsure"}
                    decided.append(json.dumps(plan, sort_keys=True, default=str))
                except ValueError as e:  # an effect that refuses (no ranking heard) must refuse alike
                    decided.append(f"refused: {e}")
            what = " ".join(f"{k}={v}" for k, v in given.items()) or "defaults"
            check(decided[0] == decided[1], f"D: {name} ({what}) decides the same with or without the unsure words"
                  + ("" if not decided[0].startswith("refused") else f" ({decided[0][:70]})"))


def test_a_retake(folder: Path) -> None:
    from fx.effects import clean_cut, entry
    from fx.effects.options import parse_all

    clip = Clip(fresh_copy(A_RETAKE, folder))
    shown = " ".join(keys(clip.words))
    check("issue from the perspective of from the perspective of hooks" in shown,
          f"A-retake: the second listen brings the false start back ({shown})")
    opts = parse_all(clean_cut.OPTIONS, entry("clean-cut")["options"], {})
    removals, kept = clean_cut.plan_cuts(clip, opts)
    starts = [r for r in removals if r.reason == "false start"]
    check(len(starts) == 1 and " ".join(W.normalize(x) for x in starts[0].words.split()) == "from the perspective of",
          "A-retake: clean-cut cuts the false start \"From the perspective of\" ("
          + "; ".join(f"{r.reason}: {r.words}" for r in removals) + ")")
    kept_text = " ".join(keys(kept))
    check(kept_text.count("from the perspective of hooks") == 1 and kept_text.count("perspective") == 1,
          f"A-retake: the restart is kept, once ({kept_text})")


def test_figures(folder: Path) -> None:
    from fx.core import amounts

    for src in NUMBERS:
        name = src.name
        keep, misread, values = FIGURES[name]
        clip = Clip(fresh_copy(src, folder))
        texts = [w.text for w in clip.words]
        first = [w.text for w in W.timed(W.load(clip.path), clip.quiet)]
        check(texts == first, f"{name}: the words are the first listen's, every figure as it wrote it")
        for figure in keep:
            n = len(figure)
            check(any(texts[i:i + n] == figure for i in range(len(texts))),
                  f"{name}: \"{' '.join(figure)}\" survives both listens unchanged")
        shown = {w.key for w in clip.words}
        check(not misread & shown, f"{name}: no digit misread on the second listen is shown ({sorted(misread & shown)})")
        check(misread <= {w.key for w in clip.unsure},
              f"{name}: the misread digits are kept, marked unsure ({', '.join(sorted(misread))})")
        read = {(round(a.value, 3), a.suffix) for a in amounts.spoken(clip.words)}
        check(all(v in read for v in values), f"{name}: the words still read as {values} ({sorted(read)})")


def evidence() -> None:
    """Every word the second listen hears on the test clips and the figures, with the values it is
    judged by."""
    for path in EVIDENCE + NUMBERS:
        clip = Clip(path)
        clip.heard  # noqa: B018  (runs the second listen, cached beside the clip)
        env = clip.envelope
        spans = clip.second_listen
        print(f"\n{path.name}: voiced level {env.voiced:.1f} dB; {len(spans)} stretch(es) heard again")
        if not spans:
            continue
        second = W.relisten(path, spans)
        verdicts = {(w.text, round(w.start, 3)): w.unsure for w in clip.unsure}
        used = {(w.text, round(w.start, 3)) for w in W.sure(clip.heard)}
        for key, segments in second.items():
            a, b = (float(x) for x in key.split("-"))
            for seg in segments:
                ws = [w for w in seg["words"] if w.key and a - 0.05 <= (w.start + w.end) / 2 <= b + 0.05]
                if not ws:
                    continue
                looped = seg["compression_ratio"] > W.LOOP_RATIO
                head = (f"  [{key}] \"{seg['text'][:60]}{'…' if len(seg['text']) > 60 else ''}\" "
                        f"no_speech {seg['no_speech_prob']:.3f}, avg_logprob {seg['avg_logprob']:+.3f}, "
                        f"compression {seg['compression_ratio']:.2f}")
                print(head)
                for w in ws[:4] if looped else ws:
                    s0, s1 = max(w.start, a), min(w.end, b)
                    if s1 - s0 < 0.02:
                        s0, s1 = w.start, max(w.end, w.start + 0.01)
                    level = W._loudest(env, s0, s1) - env.voiced
                    found = (w.text, round(w.start, 3))
                    if found in verdicts:
                        verdict = "unsure: " + verdicts[found]
                    elif found in used:
                        verdict = "sure, used"
                    else:
                        verdict = "sure; heard as the first listen heard it, which stays"
                    print(f"      {w.text!r:20} {w.start:6.2f}-{w.end:6.2f} s  p {w.prob:.3f}  voice {level:+5.1f} dB  {verdict}")
                if looped and len(ws) > 4:
                    print(f"      … and {len(ws) - 4} more words of the loop")


def calibrate() -> None:
    """Short stretches heard as the second listen hears them, over speech and over quiet."""
    import numpy as np

    rows = []
    for path in EVIDENCE:
        clip = Clip(path)
        first = W.load(path)
        env = clip.envelope
        quiet = clip._quiet_for(first)
        timed = W.timed(first, quiet)
        step = 3 if len(timed) > 80 else 2  # every other word, every third on a long take
        chunks = {(timed[i].start, timed[min(len(timed) - 1, i + n - 1)].end, "speech")
                  for i in range(0, len(timed), step) for n in (1, 3)}
        for q0, q1 in quiet:  # inside each quiet stretch, a second at a time
            t = q0 + 0.05
            while q1 - 0.05 - t >= 0.3:
                chunks.add((t, min(q1 - 0.05, t + 1.0), "quiet"))
                t = min(q1 - 0.05, t + 1.0) + 0.2
        chunks = {(round(a, 2), round(b, 2), kind) for a, b, kind in chunks if b - a >= 0.15}
        heard = W._transcribe_spans(path, sorted((a, b) for a, b, _ in chunks), "small.en")
        for a, b, kind in sorted(chunks):
            nearby = [x for x in first if a - W.NEARBY <= x.start <= b + W.NEARBY]
            for seg in heard.get(W._key((a, b)), []):
                ws = W.from_json(seg["words"])
                for i, w in enumerate(ws):
                    if not w.key or not a - 0.05 <= (w.start + w.end) / 2 <= b + 0.05:
                        continue
                    s0, s1 = (max(w.start, a), min(w.end, b)) if min(w.end, b) - max(w.start, a) >= 0.02 \
                        else (w.start, max(w.end, w.start + 0.01))
                    real = any(x.key == w.key and x.start - 0.25 <= (w.start + w.end) / 2 <= x.end + 0.25 for x in first)
                    cut = w.text.rstrip().endswith(W._DASHES) or (i + 1 < len(ws) and ws[i + 1].text.strip() in W._DASHES)
                    rows.append({"label": "invented" if kind == "quiet" else ("real" if real else "unmatched"),
                                 "p": w.prob, "nsp": seg["no_speech_prob"], "alp": seg["avg_logprob"],
                                 "loop": seg["compression_ratio"] > W.LOOP_RATIO, "zero": w.end - w.start < 0.02,
                                 "voice": W._loudest(env, s0, s1) - env.voiced, "nearby": W._said_nearby(w, nearby, cut)})
        print(f"{path.name}: {len(chunks)} stretches heard", flush=True)
    print(f"\n{len(rows)} words heard in all. real: what the first listen heard there too; invented: heard over "
          f"the quiet; unmatched: heard over speech, but not what the first listen heard there.")
    for label in ("real", "invented", "unmatched"):
        rs = [r for r in rows if r["label"] == label]
        rest = [r for r in rs if not r["loop"] and not r["zero"]]
        if not rs:
            continue
        print(f"\n{label}: {len(rs)} words, {sum(r['voice'] < 0 for r in rs)} of them under the voiced level (rule 1); "
              f"{len(rs) - len(rest)} in a loop or of no length (caught before the bounds are read)")
        if not rest:
            continue
        print(f"  the other {len(rest)}, percentiles (min, 1st, 5th, 50th, max):")
        for k in ("p", "nsp", "alp", "voice"):
            v = np.array([r[k] for r in rest])
            print(f"    {k:6} " + "  ".join(f"{np.percentile(v, q):+.3f}" for q in (0, 1, 5, 50, 100)))
        print(f"    probability under {W.SURE_PROB}: {sum(r['p'] < W.SURE_PROB for r in rest)}; no_speech over "
              f"{W.SURE_NO_SPEECH}: {sum(r['nsp'] > W.SURE_NO_SPEECH for r in rest)}; avg_logprob under "
              f"{W.SURE_LOGPROB}: {sum(r['alp'] < W.SURE_LOGPROB for r in rest)}")
    other = [r for r in rows if r["label"] == "unmatched" and not r["loop"] and not r["zero"] and r["voice"] >= 0]
    print(f"\nheard over speech with voice under them, not in a loop, and not what the first listen heard there: "
          f"{len(other)}; not said within {W.NEARBY:.0f} s (rule 4): {sum(not r['nearby'] for r in other)}")


def test_fixes_with_unsure() -> None:
    """`fix=` on words with unsure ones among them (found on take 33 on 28 October: every run with
    `fix=` on a clip that had an unsure word recursed until Python gave up)."""
    Word = W.Word
    spec = "Urban=Eben; ask editor=AskEditor"
    words = [Word("I'm", 0.0, 0.2), Word("Urban,", 0.2, 0.5), Word("Urban.", 0.6, 0.9, 0.5, "heard again"),
             Word("ask", 1.0, 1.2), Word("editor", 1.2, 1.5), Word("ask", 1.6, 1.7, 0.4, "a loop")]
    try:
        fixed = W.apply_fixes(words, spec)
    except RecursionError:
        check(False, "fix= on words with unsure ones among them returns")
        return
    check([w.text for w in fixed] == ["I'm", "Eben,", "Eben.", "AskEditor", "ask"],
          f"fix= puts the sure and the unsure words right, in time order ({[w.text for w in fixed]})")
    check([bool(w.unsure) for w in fixed] == [False, False, True, False, True],
          "fix= keeps each word's unsure mark")
    only = [w for w in words if w.unsure]
    check([w.text for w in W.apply_fixes(only, spec)] == ["Eben.", "ask"], "fix= on words that are all unsure returns")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", action="store_true", help="print every word the second listen hears on the test clips")
    ap.add_argument("--calibrate", action="store_true", help="print what the thresholds are set from (about ten minutes)")
    args = ap.parse_args(argv)
    if args.evidence:
        evidence()
        return 0
    if args.calibrate:
        calibrate()
        return 0
    test_fixes_with_unsure()
    with tempfile.TemporaryDirectory() as tmp:
        test_every_effect(test_d(Path(tmp)))
        test_a_retake(Path(tmp))
        test_figures(Path(tmp))
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
