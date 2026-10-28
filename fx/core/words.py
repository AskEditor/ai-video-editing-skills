"""Word times from the clip's speech, cached next to the clip as JSON.

The transcription runs once per clip, in its own process (the speech package and the picture
package each carry their own copy of the video libraries, and loading both in one process makes
macOS print warnings). The result is `<clip>.words.json` beside the clip: the transcriber's words
exactly as it heard them, and what a second listen heard (`relisten`). Corrections (`fix`), the
judging of the second listen (`merge`) and the timing checked against the sound (`timed`) are
applied each time the words are read, never written back, so the cache stays the raw record.

A word the second listen heard but could not be sure of is kept with the words, marked `unsure`
with the reason, and is never shown or cut on: `timed` (what every effect reads) leaves it out.

    python -m fx.core.words <clip> [--model small.en] [--refresh]   # transcribe and print
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

from . import paths

CACHE_VERSION = 1
# A short, neutral prompt in the way people really talk. It nudges the transcriber into keeping
# the "um"s, the restarts and the repeated words instead of tidying them away, which is what lets
# clean-cut find them.
PROMPT = "Um, so, uh, I- I think, like... okay."

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_PUNCT_EDGES = re.compile(r"^[^\w']+|[^\w']+$")
FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm", "mhm", "eh"}
# The sounds that hold a place while the speaker thinks: never shown as a caption, and cut by
# clean-cut unless asked to keep them. ("ah" and "eh" can be real interjections, so they stay.)
HESITATIONS = {"um", "umm", "uhm", "uh", "uhh", "erm", "er", "hmm", "hm", "mm", "mhm"}


@dataclass(frozen=True)
class Word:
    text: str  # as heard, with its punctuation ("product.")
    start: float
    end: float
    prob: float = 1.0
    # Why the word is doubted, or "" when it is not: a word the second listen heard and could not
    # be sure of (`judge`). It stays with the words, so nothing heard is lost, and is never shown
    # or cut on.
    unsure: str = ""

    @property
    def key(self) -> str:
        """The word for comparing: lower case, no punctuation, no trailing hyphen."""
        return normalize(self.text)

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def ends_sentence(self) -> bool:
        return self.text.rstrip().endswith((".", "?", "!"))

    @property
    def ends_clause(self) -> bool:
        return self.text.rstrip().endswith((",", ";", ":", "—", "-", "…")) or self.ends_sentence

    @property
    def is_filler(self) -> bool:
        return self.key in FILLERS

    @property
    def clean(self) -> str:
        """The word to show on screen: as heard, without a dangling hyphen or ellipsis."""
        t = self.text.strip().translate(_APOSTROPHES)
        return t.rstrip("-—…").rstrip(".") if t.endswith(("-", "—", "…", "...")) else t


def normalize(text: str) -> str:
    t = text.strip().lower().translate(_APOSTROPHES)
    t = _PUNCT_EDGES.sub("", t)
    return t.strip("'")


def to_json(words: Iterable[Word]) -> list[dict]:
    return [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.prob, 3),
             **({"unsure": w.unsure} if w.unsure else {})} for w in words]


def from_json(items: Iterable[dict]) -> list[Word]:
    return [Word(text=str(i["w"]), start=float(i["s"]), end=float(i["e"]), prob=float(i.get("p", 1.0)),
                 unsure=str(i.get("unsure", ""))) for i in items]


def sure(words: Iterable[Word]) -> list[Word]:
    """The words that may be shown and cut on: all but the unsure ones."""
    return [w for w in words if not w.unsure]


def in_time(words: list[Word], extra: Iterable[Word]) -> list[Word]:
    """`words` in their own order, with each of `extra` put in after the last word that starts no
    later than it does."""
    import bisect

    out = list(words)
    for w in sorted(extra, key=lambda w: (w.start, w.end)):
        out.insert(bisect.bisect_right([x.start for x in out], w.start), w)
    return out


def unsure_stretches(words: Iterable[Word], most: int = 8) -> list[dict]:
    """The unsure words, for a person to read: one entry per run of them doubted for the same
    reason (a loop of forty words is one line), each with what was heard, when, and why."""
    runs: list[list[Word]] = []
    for w in sorted((w for w in words if w.unsure), key=lambda w: (w.start, w.end)):
        if runs and runs[-1][-1].unsure == w.unsure and w.start - runs[-1][-1].end < 0.3:
            runs[-1].append(w)
        else:
            runs.append([w])
    out = []
    for run in runs:
        said = " ".join(w.text for w in run[:most]) + (" …" if len(run) > most else "")
        out.append({"w": said, "s": round(run[0].start, 3), "e": round(max(w.end for w in run), 3),
                    "why": run[0].unsure + (f" ({len(run)} words)" if len(run) > 1 else "")})
    return out


def _cache_path(clip: Path) -> Path:
    return paths.beside(clip, "words.json")


def _valid(data: dict, clip: Path, model: str) -> bool:
    return (
        data.get("version") == CACHE_VERSION
        and data.get("model") == model
        and data.get("source", {}).get("bytes") == paths.source_signature(clip)["bytes"]
    )


def load(clip: str | os.PathLike, model: str = "small.en", refresh: bool = False,
         out: Optional[Path] = None) -> list[Word]:
    """The clip's words: from the cache beside it, or by transcribing it now (once)."""
    clip = Path(clip).resolve()
    cache = out or _cache_path(clip)
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if _valid(data, clip, model):
                return from_json(data["words"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    env = dict(os.environ, PYTHONPATH=str(paths.PACK) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    run = subprocess.run(
        [sys.executable, "-m", "fx.core.words", str(clip), "--model", model, "--out", str(cache), "--quiet"]
        + (["--refresh"] if refresh else []),
        env=env, capture_output=True, text=True,
    )
    if run.returncode != 0 or not cache.exists():
        lines = [ln for ln in (run.stderr or "").splitlines() if ln.strip() and "objc[" not in ln]
        raise RuntimeError("could not find the words: " + (lines[-1] if lines else "the transcriber stopped"))
    return from_json(json.loads(cache.read_text())["words"])


def _engine(model: str):
    from faster_whisper import WhisperModel

    from . import models

    threads = max(1, min(8, (os.cpu_count() or 4)))
    return WhisperModel(str(models.speech_path(model)), device="cpu", compute_type="int8", cpu_threads=threads)


def _listen(engine, samples) -> list[dict]:
    """What the transcriber hears in `samples`, one entry per segment: its text, its words, and its
    own confidence in it (`no_speech_prob`: how likely it thought the sound was not speech at all;
    `avg_logprob`: how sure it was of the words, 0 being certain; `compression_ratio`: above about
    2.4 the text is the same few words over and over)."""
    segments, _ = engine.transcribe(
        samples,
        language="en",
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,  # no sampling: the same clip gives the same words every time
        condition_on_previous_text=False,
        initial_prompt=PROMPT,
        vad_filter=False,
    )
    out = []
    for seg in segments:
        words = [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                 for w in seg.words or [] if w.word.strip()]
        out.append({"text": seg.text.strip(), "no_speech_prob": float(seg.no_speech_prob),
                    "avg_logprob": float(seg.avg_logprob), "compression_ratio": float(seg.compression_ratio),
                    "words": words})
    return out


def _hear(engine, samples) -> tuple[list[Word], str]:
    segments = _listen(engine, samples)
    return [w for s in segments for w in s["words"]], " ".join(s["text"] for s in segments)


def transcribe(clip: Path, model: str = "small.en") -> dict:
    """Runs the transcriber (in this process). Use `load()` from everywhere else."""
    import numpy as np  # noqa: F401  (the speech package needs it loaded first)

    from . import video

    info = video.probe(clip)
    samples = video.read_audio(info, rate=16000, channels=1)[:, 0]
    started = time.time()
    words, text = _hear(_engine(model), samples)
    return {
        "version": CACHE_VERSION,
        "model": model,
        "source": dict(paths.source_signature(clip), seconds=round(info.duration, 3)),
        "seconds_to_transcribe": round(time.time() - started, 2),
        "text": text,
        "words": to_json(words),
    }


# ---- a second listen -----------------------------------------------------------------------------
#
# The transcriber tidies speech up. A false start it hears as part of the take ("from the
# perspective of- from the perspective of hooks") often comes out written once, with one word
# stretched across both attempts and the rest of the voice left without words. Nothing that reads
# only the words can find a retake that was never written down. So the stretches the words cannot
# explain are listened to again, each on its own, and what is heard there replaces what was there.
#
# Out of its sentence, a second of sound is also where the transcriber invents. On clip D it heard
# "Town car." where the take says "town, or", "Hotel." again in the tail of "hotel", and a second
# "went" at the edge of a stretch that began just after "went"; on clean speech it hears "I'll be
# safe." for "lively city". How sure it says it is does not tell these apart from real words, so
# nothing heard the second time is used until it is judged (`judge`, below). A word that fails is
# kept with the words, marked `unsure` with the reason, and never shown or cut on.

SECOND_PAD = 0.12  # sound either side of a stretch, so its first and last words are whole
# How a stretch is heard again, and what is kept of it. Each version's results live in the clip's
# cache under their own name: a cache from before this version (whose "second_listen" holds words
# without the transcriber's confidence) is heard again automatically, and an older copy of the pack
# reading the same clip keeps its own results and never undoes these.
SECOND_VERSION = 2
SECOND_KEY = f"second_listen_v{SECOND_VERSION}"

# What a word heard the second time must pass to be used (`judge`). The numbers come from hearing
# 280 short stretches of the eight test clips exactly as the second listen does
# (`tests/second_listen.py --calibrate`, 28 October 2026): 503 words heard over speech were what the
# first listen heard there, and 709 were heard over the clips' own quiet, where nothing is said.
#
# - The voice under a word tells them apart: 707 of the 709 sit under the clip's voiced level
#   (rule 1), against 4 of the 503.
# - The transcriber's confidence barely does. Real words go down to 0.001 in probability (13 of them
#   under 0.05, nearly all the first word of a stretch, heard without its sentence), up to 0.63 in
#   no_speech_prob and down to -1.12 in avg_logprob, so the bounds of rule 2 sit at that edge and
#   catch only what almost no real word looks like.
# - The words invented on D sit on real voice with real-looking confidence (probability 0.41-0.99,
#   no_speech_prob 0.002-0.09, avg_logprob -0.43 to -0.49): rules 3 and 4 catch them, and rule 4
#   catches 99 of the 121 other words heard over speech, with voice under them, that the first
#   listen did not hear there.
#
# `tests/second_listen.py --evidence` prints every word the second listen hears on the test clips,
# with these values and its verdict.
SURE_PROB = 0.05  # 2. the word's own probability, at least
SURE_NO_SPEECH = 0.60  # 2. its segment's no_speech_prob, at most
SURE_LOGPROB = -1.00  # 2. its segment's avg_logprob, at least
LOOP_RATIO = 2.4  # 3. a segment that compresses more than this is a few words over and over
MOST_PER_SECOND = 5.0  # 3. more words than this per second of the stretch is a loop too
NEARBY = 8.0  # 4. a word added is one said within this many seconds (clean-cut's retake window)
# 3. What the transcriber says to sound it cannot make out.
_HALLUCINATIONS = {"thank you", "thanks for watching", "you", "bye", "thank you for watching", "so"}


def plausible_length(word: Word) -> float:
    """The longest a word of this many letters is usually said for (a slow speaker, plus slack)."""
    return 0.18 + 0.1 * max(1, len(word.key))


def _key(span: tuple[float, float]) -> str:
    return f"{span[0]:.2f}-{span[1]:.2f}"


def doubtful_spans(words: list[Word], quiet: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Stretches the words cannot explain: one word lasting far longer than its letters allow, or
    voice that no word covers."""
    spans = []
    for w in words:
        if w.end - w.start > max(0.6, plausible_length(w) + 0.3):
            spans.append((w.start, w.end))
    # Voice: everything between the quiet stretches.
    voiced, t = [], 0.0
    for q0, q1 in quiet:
        if q0 > t:
            voiced.append((t, q0))
        t = max(t, q1)
    if t < duration:
        voiced.append((t, duration))
    for v0, v1 in voiced:
        if v1 - v0 < 0.3:
            continue
        cuts = sorted((max(v0, w.start), min(v1, w.end)) for w in words if w.end > v0 and w.start < v1)
        pos = v0
        for a, b in cuts + [(v1, v1)]:
            if a - pos >= 0.3:
                spans.append((pos, a))
            pos = max(pos, b)
    spans.sort()
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(round(float(a), 2), round(float(b), 2)) for a, b in merged]


def relisten(clip: str | os.PathLike, spans: list[tuple[float, float]], model: str = "small.en") -> dict:
    """What is heard in each span on its own, as {"a-b": [segments]}: each segment's text, the
    transcriber's confidence in it and its words (`_listen`). Kept in the clip's cache under
    `SECOND_KEY`, so each span is heard once per version."""
    clip = Path(clip).resolve()
    cache = _cache_path(clip)
    want = {_key(s) for s in spans}
    have = _second_store(json.loads(cache.read_text()), model)["spans"]
    todo = [s for s in spans if _key(s) not in have]
    if todo:
        env = dict(os.environ, PYTHONPATH=str(paths.PACK) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        tmp = cache.with_name(f"{cache.stem}.second-{os.getpid()}.json.part")
        run = subprocess.run(
            [sys.executable, "-m", "fx.core.words", str(clip), "--model", model, "--quiet", "--out", str(tmp),
             "--spans", ",".join(_key(s) for s in todo)],
            env=env, capture_output=True, text=True,
        )
        if run.returncode == 0 and tmp.exists():
            heard = json.loads(tmp.read_text())
            tmp.unlink(missing_ok=True)
            # Read again before writing: another run may have added to the cache meanwhile.
            data = json.loads(cache.read_text())
            store = _second_store(data, model)
            store["spans"].update(heard)
            data[SECOND_KEY] = store
            part = cache.with_name(f"{cache.name}.{os.getpid()}.part")
            part.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
            part.replace(cache)
            have = store["spans"]
    return {k: [_segment_from_json(seg) for seg in have[k]] for k in sorted(want) if k in have}


def _second_store(data: dict, model: str) -> dict:
    """This version's second listen from a cache, or an empty one when it has none (or another's)."""
    store = data.get(SECOND_KEY)
    if (isinstance(store, dict) and store.get("version") == SECOND_VERSION and store.get("model") == model
            and store.get("pad") == SECOND_PAD and isinstance(store.get("spans"), dict)):
        return store
    return {"version": SECOND_VERSION, "model": model, "pad": SECOND_PAD, "spans": {}}


def _segment_json(seg: dict, offset: float) -> dict:
    return {"text": seg["text"], "no_speech_prob": round(seg["no_speech_prob"], 4),
            "avg_logprob": round(seg["avg_logprob"], 4), "compression_ratio": round(seg["compression_ratio"], 3),
            "words": to_json(Word(w.text, w.start + offset, w.end + offset, w.prob) for w in seg["words"])}


def _segment_from_json(seg: dict) -> dict:
    return dict(seg, words=from_json(seg.get("words", [])))


def _transcribe_spans(clip: Path, spans: list[tuple[float, float]], model: str) -> dict:
    from . import video

    info = video.probe(clip)
    samples = video.read_audio(info, rate=16000, channels=1)[:, 0]
    engine = _engine(model)
    out = {}
    for a, b in spans:
        lo = max(0.0, a - SECOND_PAD)
        chunk = samples[int(lo * 16000):int(min(len(samples) / 16000, b + SECOND_PAD) * 16000)]
        out[_key((a, b))] = [_segment_json(seg, lo) for seg in _listen(engine, chunk)]
    return out


def _middle(w: Word) -> float:
    return (w.start + w.end) / 2


def _overlap(a: Word, b: Word) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def _loudest(env, t0: float, t1: float) -> float:
    """The level the sound between t0 and t1 reaches, in the envelope's dB: the loudest tenth of
    it, so one click does not count as a voice."""
    import numpy as np

    i0 = int(min(len(env.db) - 1, max(0, round(t0 / env.hop))))
    i1 = int(min(len(env.db), max(i0 + 1, round(t1 / env.hop))))
    return float(np.percentile(env.db[i0:i1], 90))


def _sentence(seg: dict) -> bool:
    """A whole sentence of one or two words: "Hotel.", "Town car.", "Thanks.", "Why?"."""
    ws = [w for w in seg["words"] if w.key]
    return (1 <= len(ws) <= 2 and ws[0].text.lstrip("\"'")[:1].isupper()
            and ws[-1].text.rstrip("\"'").endswith((".", "!", "?")) and not ws[-1].text.endswith("..."))


_DASHES = ("-", "—", "–")


def _said_nearby(w: Word, said: list[Word], cut_off: bool) -> bool:
    """Is this word said in `said` (the first listen's words around the stretch)? A word cut off
    part-way ("perspec-") counts when a word said nearby begins with it."""
    if any(x.key == w.key for x in said):
        return True
    return cut_off and len(w.key) >= 2 and any(x.key.startswith(w.key) for x in said)


def judge(segments: list[dict], span: tuple[float, float], first: list[Word], settled: list[Word],
          env, voiced: float) -> tuple[list[Word], list[Word]]:
    """What the second listen heard in `span`, each word left sure or marked unsure with the reason:
    (the stretch's own words, words it heard again from just outside the stretch).

    `first` is the first listen's words, `settled` those of them no stretch is heard again over,
    and `voiced` the level a word's sound must reach. A word is sure when all of these hold:

    1. **There is voice under it.** The loudest tenth of its sound, inside the stretch, reaches
       `voiced`: the level the clip's own words reach (the core's speech envelope, halfway from
       the quiet between words to a typical syllable). A word with no length has no sound to hear.
    2. **The transcriber was not unsure.** Its probability is at least `SURE_PROB`, and its
       segment's no_speech_prob at most `SURE_NO_SPEECH` and avg_logprob at least `SURE_LOGPROB`.
    3. **It is not one of the transcriber's known inventions.** Kept narrow, because a real
       false start comes back from a second listen as a fragment ("From the perspective of-"),
       never as these:
       - a loop: the same few words over and over (`LOOP_RATIO`, `MOST_PER_SECOND`);
       - a stock phrase ("Thank you.", "Bye.");
       - a whole sentence of one or two words, the only thing heard in the stretch ("Hotel.",
         "Town car."): what it says to a sound it cannot place;
       - the first listen's own word beside the stretch, heard again: the same word, over the
         same sound ("Hotel." inside the first listen's "hotel.", 16.6-18.1 s on D).
    4. **It is said nearby.** The second listen is there to find words said twice (a false start
       and its restart, a retake, a stammer), so a word it adds is one the speaker says within
       `NEARBY` seconds. One the speaker never says there is the transcriber guessing.
    """
    a, b = span
    heard = []  # (segment, word, cut off): a dash after a word comes back as a word of its own
    for seg in segments:
        ws = seg["words"]
        for i, w in enumerate(ws):
            if w.key:
                after = ws[i + 1].text.strip() if i + 1 < len(ws) else ""
                heard.append((seg, w, w.text.rstrip().endswith(_DASHES) or after in _DASHES))
    # The padding is there to make the first and last words whole: a word belongs to the stretch
    # when its middle is inside it.
    inside = [(seg, w, cut) for seg, w, cut in heard if a - 0.05 <= _middle(w) <= b + 0.05]
    alone = len([seg for seg in segments if any(w.key for w in seg["words"])]) == 1
    said = " ".join(w.key for _, w, _ in inside)
    nearby = [x for x in first if a - NEARBY <= x.start <= b + NEARBY]
    content, echoes = [], []
    # Each word keeps the first reason it fails, the most telling first.
    for seg, w, cut in inside:
        twin = next((x for x in settled if x.key == w.key and _overlap(x, w) > 0.02), None)
        if twin is not None:
            echoes.append(replace(w, unsure=f"the first listen's \"{twin.text}\" ({twin.start:.2f}-{twin.end:.2f} s) "
                                             f"heard again"))
            continue
        why = ""
        s0, s1 = max(w.start, a), min(w.end, b)
        if s1 - s0 < 0.02:
            s0, s1 = w.start, w.end
        if seg["compression_ratio"] > LOOP_RATIO or len(inside) > MOST_PER_SECOND * max(0.4, b - a):
            why = "a loop: the same few words over and over"
        elif said in _HALLUCINATIONS:
            why = f"\"{said}\" is what the transcriber says to sound it cannot make out"
        elif alone and _sentence(seg):
            why = f"\"{seg['text']}\", a whole sentence of one or two words alone in the stretch"
        elif w.end - w.start < 0.02:
            why = "no length of its own"
        elif _loudest(env, s0, s1) < voiced:
            why = f"no voice under it ({_loudest(env, s0, s1) - voiced:+.1f} dB from the clip's voiced level)"
        elif w.prob < SURE_PROB:
            why = f"the transcriber gave it {w.prob:.2f}"
        elif seg["no_speech_prob"] > SURE_NO_SPEECH:
            why = f"the transcriber put {seg['no_speech_prob']:.0%} on there being no speech"
        elif seg["avg_logprob"] < SURE_LOGPROB:
            why = f"the transcriber was unsure of the whole stretch (avg_logprob {seg['avg_logprob']:.2f})"
        elif not _said_nearby(w, nearby, cut):
            why = f"\"{w.clean}\" is not said within {NEARBY:.0f} s"
        content.append(replace(w, unsure=why))
    return content, echoes


_FIGURE = re.compile(r"[\d%$£€]")


def _figure(w: Word) -> bool:
    """A word that is a figure or a piece of one ("19", ".5%,", ",000", "$", "%")."""
    return bool(_FIGURE.search(w.text))


def merge(words: list[Word], second: dict, env, voiced: Optional[float] = None) -> list[Word]:
    """The words, with what the second listen heard (`relisten`) judged (`judge`) and used.

    A stretch whose words are all sure replaces what the first listen wrote there; one where any
    word is unsure keeps the first listen's words, so a real word is never lost to a half-trusted
    hearing. Every word heard there that is not used is kept beside them, marked unsure.

    Figures are the first listen's. A figure read aloud is long ("197" is "one hundred and
    ninety-seven"), so it is often taken for a stretched word, and heard again out of its sentence
    its digits come back wrong: "19.5%" as "19 17 .5", "500,000" as "500 100,000". So where a
    stretch holds a figure, in either listen, and the second listen does not hear there exactly the
    words the first wrote, the first listen's words stay."""
    voiced = env.voiced if voiced is None else voiced
    spans = sorted((tuple(float(x) for x in k.split("-")), k) for k in second)
    heard_again = [any(a <= _middle(w) <= b for (a, b), _ in spans) for w in words]
    settled = [w for w, again in zip(words, heard_again) if not again]
    replaced: set[int] = set()
    added: list[Word] = []
    doubted: list[Word] = []
    for (a, b), key in spans:
        content, echoes = judge(second[key], (a, b), words, settled, env, voiced)
        doubted += echoes
        if not content:
            continue
        old = [i for i, w in enumerate(words) if heard_again[i] and a <= _middle(w) <= b]
        # A piece with no letters ("%", "-") is not a word heard differently: it never makes a
        # stretch differ, so a stretch heard the same keeps it.
        same = [w.key for w in content] == [words[i].key for i in old if words[i].key]
        figure = [x for x in words if _figure(x) and min(x.end, b) - max(x.start, a) > 0]
        if not same and (figure or any(_figure(w) for w in content)):
            first_said = " ".join(x.text for x in figure) or "no figure"
            content = [replace(w, unsure=f"inside a figure, heard as \"{' '.join(c.text for c in content)}\" where "
                                         f"the first listen heard \"{first_said}\", which stays") for w in content]
        why = next((w.unsure for w in content if w.unsure), "")
        if why:
            doubted += [w if w.unsure else replace(w, unsure=f"heard with an unsure word ({why})") for w in content]
        elif not same:
            replaced.update(old)
            added += content
        # (heard the same again: nothing changes)
    kept = [w for i, w in enumerate(words) if i not in replaced]
    # A word added that is the same as a word kept over the same sound would be shown twice.
    for w in list(added):
        twin = next((x for x in kept if x.key == w.key and _overlap(x, w) > 0.02), None)
        if twin is not None:
            added.remove(w)
            doubted.append(replace(w, unsure=f"the first listen's \"{twin.text}\" heard again"))
    return in_time(kept, added + doubted)


# ---- corrections ---------------------------------------------------------------------------

def parse_fixes(spec: str | None) -> list[tuple[list[str], list[str]]]:
    """`"Jon=John; ask editor=AskEditor"` -> [(['jon'], ['John']), (['ask', 'editor'], ['AskEditor'])]."""
    fixes = []
    for part in re.split(r"[;\n]", spec or ""):
        if "=" not in part:
            continue
        wrong, right = part.split("=", 1)
        wrong_keys = [normalize(x) for x in wrong.split() if normalize(x)]
        right_words = right.split()
        if wrong_keys and right_words:
            fixes.append((wrong_keys, right_words))
    return fixes


def apply_fixes(words: list[Word], spec: str | None) -> list[Word]:
    """Replace misheard words. Punctuation after the last replaced word is kept; the times of the
    words replaced are shared out across the words that replace them."""
    fixes = parse_fixes(spec)
    if not fixes:
        return list(words)
    if any(w.unsure for w in words) and not all(w.unsure for w in words):
        # The sure and the unsure words are corrected apart, so an unsure word heard in the middle
        # of a name never stops the name being put right. Each half is then all one kind and is
        # corrected below: splitting a half that is all unsure again would never end.
        return in_time(apply_fixes(sure(words), spec), apply_fixes([w for w in words if w.unsure], spec))
    out: list[Word] = []
    i = 0
    while i < len(words):
        for wrong, right in fixes:
            n = len(wrong)
            if [w.key for w in words[i:i + n]] == wrong:
                group = words[i:i + n]
                tail = re.search(r"[^\w']+$", group[-1].text)
                span0, span1 = group[0].start, group[-1].end
                step = (span1 - span0) / len(right)
                for k, r in enumerate(right):
                    text = r + (tail.group(0) if tail and k == len(right) - 1 else "")
                    out.append(Word(text, span0 + k * step, span0 + (k + 1) * step, min(w.prob for w in group),
                                    group[0].unsure))
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


# ---- timing, refined against the sound ---------------------------------------------------------

# A speech recogniser marks a word's start after its sound has already begun. AskEditor measured
# about 50 ms on its own transcriber and pulls every start 0.1 s earlier; on this pack's test clips
# the transcriber here marked starts a median 0.13 s late after a real pause (39 onsets, 28 October
# 2026). Uncorrected, a cut or a word that appears on its start lands one to four frames late. So
# every start is pulled 0.1 s earlier, and then the sound has the last word: a start that the pull
# puts in the quiet before the word moves forward again to where the voice begins (`refine`).
HEARD_LATE = 0.10
# A quiet stretch at least this long inside a word's span is a real pause the transcriber folded
# into the word, not the short closure inside a word like "ex-pect".
FOLDED_PAUSE = 0.18


def pull_starts(words: list[Word], by: float = HEARD_LATE) -> list[Word]:
    """Each start `by` seconds earlier, never before the word before it began. Ends are kept as heard."""
    out: list[Word] = []
    floor = 0.0
    for w in words:
        s = max(floor, w.start - by, 0.0)
        out.append(replace(w, start=s, end=max(w.end, s + 0.02)))
        floor = s + 0.02
    return out


def refine(words: list[Word], quiet: list[tuple[float, float]]) -> list[Word]:
    """Pull word edges out of the quiet around them.

    The transcriber's word edges drift: it reports nearly every word as touching the next, so a
    word often 'starts' in the pause before it, 'ends' in the pause after it, or carries a whole
    pause inside its span. Where the sound says an edge sits in a quiet stretch, the edge moves to
    where the speech really begins or stops.
    """
    out = []
    for w in words:
        s, e = w.start, w.end
        for q0, q1 in quiet:
            if q1 <= s or q0 >= e:
                continue
            if q0 <= s and e <= q1:  # the whole word sits in quiet: the sound is too soft to say
                continue
            if q0 <= s < q1:  # starts in a quiet stretch: speech begins at its end
                s = q1 - 0.02
            elif q0 < e <= q1:  # ends in a quiet stretch: speech stops at its start
                e = q0 + 0.03
            elif q1 - q0 >= FOLDED_PAUSE:  # a pause folded inside the word
                # The transcriber stretches a word until the next one begins, so the word's own
                # sound is the part before the pause, unless almost nothing comes before it.
                if q0 - s >= 0.10:
                    e = q0 + 0.03
                else:
                    s = q1 - 0.02
        if e - s < 0.05:
            s, e = w.start, w.end
        out.append(replace(w, start=s, end=e))
    # A start still marked a little after the voice came back from a pause moves back to where it
    # came back, as long as no other word sits in between.
    for k, w in enumerate(out):
        prev_end = out[k - 1].end if k else 0.0
        for q0, q1 in quiet:
            if w.start - 0.15 <= q1 < w.start and q1 - q0 >= 0.08 and prev_end <= q1:
                out[k] = replace(w, start=max(prev_end, q1 - 0.02))
                break
    # Keep them in order and never overlapping. The start is the better-known edge (corrected, then
    # checked against the sound), so where two words overlap the earlier one's end gives way.
    for k in range(1, len(out)):
        if out[k].start < out[k - 1].end:
            prev = out[k - 1]
            if out[k].start - prev.start >= 0.05:
                out[k - 1] = replace(prev, end=out[k].start)
            else:  # the earlier word would all but vanish: split the overlap instead
                mid = (out[k].start + prev.end) / 2
                out[k - 1] = replace(prev, end=max(prev.start + 0.02, mid))
                out[k] = replace(out[k], start=min(out[k].end - 0.02, mid))
    return out


def unheard(word: Word, quiet: list[tuple[float, float]], prob: float = 0.5) -> bool:
    """A word the transcriber is unsure of whose whole span is quiet: a breath or a knock it put a
    word to. It is never captioned, and clean-cut treats it as part of the silence around it."""
    if word.prob >= prob:
        return False
    return any(q0 <= word.start + 0.02 and word.end - 0.02 <= q1 for q0, q1 in quiet)


def timed(words: list[Word], quiet: list[tuple[float, float]]) -> list[Word]:
    """The words as the effects use them: starts corrected for the transcriber's lateness, both
    edges checked against the sound, and words nobody said left out, the unsure ones with them."""
    kept = [w for w in words if w.key and not w.unsure and not unheard(w, quiet)]  # a lone "-" is not a word
    return refine(pull_starts(kept), quiet)


def pause_after(words: list[Word], i: int, quiet: list[tuple[float, float]]) -> float:
    """How long the voice really stops after word `i`, read from the sound: the transcriber's own
    gaps are nearly always zero. 0 when the next word follows straight on."""
    if i + 1 >= len(words):
        return 0.0
    a, b = words[i].end, words[i + 1].start
    best = 0.0
    for q0, q1 in quiet:
        lo, hi = max(q0, a - 0.06), min(q1, b + 0.06)
        if hi > lo:
            best = max(best, hi - lo)
    return best


def numbered(words: list[Word], per_line: int = 12) -> str:
    """The transcript with word numbers and times, for reading a plan."""
    lines, line = [], []
    for i, w in enumerate(words):
        if not line:
            line.append(f"[{i:>3} {w.start:6.2f}s]")
        line.append(w.text)
        if len(line) > per_line or w.ends_sentence:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fx.core.words", description="Find the words of a clip.")
    ap.add_argument("clip")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--out", help="where to write the JSON (default: next to the clip)")
    ap.add_argument("--refresh", action="store_true", help="transcribe again even if a cache exists")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--spans", help="listen again to these stretches only (a-b,c-d) and write what is heard")
    args = ap.parse_args(argv)
    clip = Path(args.clip).resolve()
    if args.spans:
        spans = [tuple(float(x) for x in s.split("-")) for s in args.spans.split(",") if s]
        Path(args.out).write_text(json.dumps(_transcribe_spans(clip, spans, args.model), ensure_ascii=False) + "\n")
        return 0
    out = Path(args.out) if args.out else _cache_path(clip)
    try:
        fresh = out.exists() and _valid(json.loads(out.read_text()), clip, args.model)
    except (json.JSONDecodeError, OSError):
        fresh = False
    if args.refresh or not fresh:
        data = transcribe(clip, args.model)
        tmp = out.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(out)
    if not args.quiet:
        words = from_json(json.loads(out.read_text())["words"])
        print(numbered(words))
        print(f"\n{len(words)} words · {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
