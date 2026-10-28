"""sound-effects: our own sounds (whoosh, swoosh-in, pop, click, tick, hit, riser, ding) on the moments
that matter, each landing on the first frame of its moment and sitting well under the voice.

Where the sounds go:

- `sounds`: a list of `{at, kind}`, `at` being seconds or words that are said ("quality").
- `on=cuts`: every cut. The cuts are read from the plan of the edit that made them (the
  `<video>.plan.json` that `clean-cut` writes beside its video, or `events=<plan>`), and each is
  checked against the picture itself: the frame where it really changes. Without a plan, the cuts
  are found in the picture.
- `on=cards`: every card or graphic a plan lays: a tier list's cards as each lands, a call to
  action, a headline, a word behind the head, a before-and-after wipe.
- `on=words`: the words in `words`, every time they are said; with none given, the most specific
  word of each sentence. Word times are the core's: corrected for the transcriber's lateness and
  checked against the sound, with `fix` for misheard words.
- `on=auto` (the default): the plan beside the video decides (cuts for a cut, cards for cards),
  then cuts found in the picture.

How a sound lands: every moment is moved to the first frame that shows it (the frame where the cut
has happened, the card has landed, or the word has begun), and the sound is placed so that its own
landing point falls on the start of that frame, to the sample. A struck sound lands on its attack; a
swell (whoosh, swoosh-in) on its peak, so it rises into the moment; the riser ends on it. Sounds
closer together than they can be told apart are merged into the first.

How loud: the voice's loudness is measured, and every sound's loudest tenth of a second is set
`level` dB under it, the same for every kind (the library is levelled by the same measure), so a
click and a whoosh sit at one height. `variety` lets each repeat differ a little in pitch, and
whooshes alternate their direction, so a run of them does not sound like a machine.

After the render the file is checked: each sound is found again in the encoded video and its
landing measured against its frame, and its level against the voice; a sound sheet shows the
voice, the sounds and a close-up of every landing beside its frame.
"""

from __future__ import annotations

import json
import math
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import loudness, moments, sound, soundsheet, video
from ..core import framing as framing_mod
from ..core import words as wordsmod
from ..core.paths import PACK
from ..core.timeline import Timeline
from . import captions as cap
from . import options as O

NAME = "sound-effects"
RATE = video.AUDIO_RATE
SOUNDS = PACK / "sounds"
KINDS = ("whoosh", "swoosh-in", "pop", "click", "tick", "hit", "riser", "ding")
SWELLS = ("whoosh", "swoosh-in")
AUTO_KIND = {"cuts": "whoosh", "cards": "pop", "words": "pop"}
# Two sounds closer than this (seconds) blur into one: the later is dropped. A swell needs more room.
APART = {"whoosh": 0.45, "swoosh-in": 0.35, "riser": 0.6}
APART_STRUCK = 0.18

OPTIONS = {
    "sounds": O.Text("sounds at given moments: [{\"at\": 3.2, \"kind\": \"pop\"}, {\"at\": \"quality\", \"kind\": "
                     "\"ding\"}], or 3.2 pop; quality ding"),
    "on": O.Choice(("auto", "cuts", "cards", "words", "none"),
                   "which moments get a sound: cuts, cards, words, none, or auto (what the video's plan holds)"),
    "kind": O.Choice(("auto",) + KINDS, "the sound for those moments: whoosh, swoosh-in, pop, click, tick, hit, riser, "
                                        "ding, or auto (a whoosh on a cut, a pop on a card or a word)"),
    "words": O.Text("for on=words: the words to mark, by commas (empty: the key word of each sentence)"),
    "events": O.Text("the plan that says where the cuts or cards are (empty: the one beside the video)"),
    "level": O.Number(-30, 0, "how far under your voice the sounds sit, in dB (every kind at the same height)"),
    "variety": O.Number(0, 1, "how much each repeat differs in pitch and direction: 0 the same every time, 1 the most"),
    "fix": O.Text("misheard words, as wrong=right; …"),
}

LAB = "{lab}"
MATRIX = [
    ("E-cut", "E-cut", {"on": "cuts", "events": LAB + "/clean-cut/E.plan.json"}),
    ("tier-B", "tier-B", {"on": "cards", "events": LAB + "/tier-list/B.plan.json", "kind": "pop"}),
    ("A-words", "A", {"on": "words", "kind": "pop"}),
    ("B-given", "B", {"on": "none", "sounds": "video tick; quality riser; editor ding"}),
    ("C-words", "C", {"on": "words", "words": "product, professionally", "kind": "click"}),
    # Phrases, not single words: the transcriber's second listen invents "Town car." and "Hotel." in D's
    # pauses (a core fix is on its way), and a sound must never mark a word nobody said.
    ("D-words", "D", {"on": "words", "words": "lively town, lively city, my hotel", "kind": "swoosh-in"}),
]


# ---- the library -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    kind: str
    samples: np.ndarray  # (n, 2) float32
    anchor: float  # seconds into it where it lands
    lands: str  # attack, peak or end
    loudest: float  # LUFS of its loudest tenth of a second


_LIBRARY: dict[str, Sample] = {}


def library() -> dict[str, Sample]:
    if not _LIBRARY:
        manifest = json.loads((SOUNDS / "sounds.json").read_text())
        for kind, m in manifest.items():
            with wave.open(str(SOUNDS / m["file"])) as w:
                if w.getframerate() != RATE or w.getsampwidth() != 2:
                    raise ValueError(f"sounds/{m['file']} must be 48 kHz 16-bit; run sounds/make_sounds.py")
                pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels())
            x = pcm.astype(np.float32) / 32768.0
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            _LIBRARY[kind] = Sample(kind, x, float(m["anchor_s"]), m["lands"], float(m["loudest_lufs"]))
    return _LIBRARY


# ---- the moments -------------------------------------------------------------------------------------

@dataclass
class Event:
    t: float  # when the moment happens (seconds into the video)
    kind: str
    label: str
    source: str  # where it came from: given, plan, picture, words
    frame: int = 0  # the first frame that shows it
    at: float = 0.0  # that frame's start: where the sound lands
    gain_db: float = 0.0
    pitch: float = 1.0
    mirror: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {"at": round(self.at, 4), "frame": self.frame, "kind": self.kind, "event": self.label,
                "moment": round(self.t, 4), "from": self.source, "gain_db": round(self.gain_db, 2),
                "pitch": round(self.pitch, 4), "mirror": self.mirror, **({"note": self.note} if self.note else {})}


def picture_cuts(info: video.VideoInfo) -> list[int]:
    """Frames where the picture jumps: a new shot, or a jump cut inside one. The whole picture changes
    far more between those two frames than between any of their neighbours."""
    w = 90
    h = max(2, int(round(w * info.height / info.width / 2)) * 2)
    diffs, prev = [], None
    for f in video.FrameReader(info, size=(w, h), gray=True):
        g = f.astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(diffs)
    cuts = []
    for j in range(1, len(d)):
        around = np.concatenate([d[max(1, j - 12):j], d[j + 1:j + 13]])
        if len(around) and d[j] >= 8.0 and d[j] >= 5.0 * max(0.5, float(np.median(around))):
            cuts.append(j)
    return cuts


def _plan_for(clip, given: str) -> tuple[Optional[dict], Optional[Path]]:
    from . import PlanError

    path = Path(given).expanduser() if given.strip() else Path(clip.path).with_suffix(".plan.json")
    if not path.exists():
        if given.strip():
            raise PlanError(f"there is no plan at {given}")
        return None, None
    try:
        plan = json.loads(path.read_text())
    except ValueError:
        raise PlanError(f"{path} is not a plan this pack wrote") from None
    seconds = float((plan.get("output") or {}).get("seconds", -1))
    mine = clip.info.frames / clip.info.rate
    if seconds > 0 and abs(seconds - mine) > 1.5 / clip.info.rate:
        raise PlanError(f"that plan is for a video {seconds:.2f} s long, and this one is {mine:.2f} s: give the "
                        f"video that {plan.get('effect', 'the effect')} made, not the one it started from")
    return plan, path


def _plan_cuts(plan: dict) -> list[tuple[float, str]]:
    if plan.get("effect") != "clean-cut" or not isinstance(plan.get("removals"), list):
        return []
    seconds_in = float(plan.get("summary", {}).get("seconds_in", 0)) or max(r["end"] for r in plan["removals"])
    tl = Timeline.keeping(seconds_in + 1e-3, [(float(r["start"]), float(r["end"])) for r in plan["removals"]])
    return [(j, f"cut {k + 1}") for k, j in enumerate(tl.joins)]


def _plan_cards(plan: dict) -> list[tuple[float, str, Optional[str]]]:
    """(moment, label, the kind that suits it or None) for each card or graphic the plan lays."""
    e = plan.get("effect")
    out: list[tuple[float, str, Optional[str]]] = []
    if e == "tier-list":
        for c in plan.get("cards", []):
            out.append((float(c.get("lands", c.get("at", 0))), f"{c.get('text', 'card')} ({c.get('tier', '')})", None))
    elif e == "cta-card" and plan.get("card"):
        out.append((float(plan["card"]["start"]), f"{plan['card'].get('kind', 'card')} card", None))
    elif e == "headline" and plan.get("headline"):
        h = plan["headline"]
        out.append((float(h["start"]), "headline", "hit" if h.get("animation") == "slam" else None))
    elif e == "words-behind":
        pieces = plan.get("pieces") or [{"text": plan.get("text", "word"), "land": plan.get("land", 0)}]
        out += [(float(p["land"]), f"“{p.get('text', '')}” behind", "hit") for p in pieces]
    elif e == "before-after" and plan.get("wipe"):
        w = plan["wipe"]
        out.append((float(w.get("at", 0)) + float(w.get("duration", 0)) / 2, "the wipe", "whoosh"))
    return out


def _parse_given(raw: str) -> list[tuple[str, str]]:
    """`sounds` as (at, kind) pairs: JSON, a JSON file, or `at kind; at kind`."""
    raw = raw.strip()
    if not raw:
        return []
    if not raw.startswith("[") and Path(raw).expanduser().suffix.lower() == ".json" and Path(raw).expanduser().exists():
        raw = Path(raw).expanduser().read_text()
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except ValueError:
            raise O.OptionError("sounds: that list is not valid JSON") from None
        out = []
        for it in items:
            if not isinstance(it, dict) or "at" not in it:
                raise O.OptionError(f"sounds: each item needs an at (got {it!r})")
            out.append((str(it["at"]), str(it.get("kind", "pop"))))
        return out
    out = []
    for part in re.split(r"[;\n]", raw):
        part = part.strip()
        if not part:
            continue
        bits = part.rsplit(None, 1)
        if len(bits) == 2 and bits[1].lower() in KINDS:
            out.append((bits[0].strip().strip("\"'"), bits[1].lower()))
        elif len(bits) == 2 and re.fullmatch(r"\d+(\.\d+)?s?", bits[0].strip()):
            raise O.OptionError(f"sounds: no sound called {bits[1]!r}; the sounds are {', '.join(KINDS)}")
        else:
            out.append((part.strip("\"'"), "pop"))
    return out


def _find_words(words: list[wordsmod.Word], phrase: str) -> list[int]:
    keys = [wordsmod.normalize(x) for x in phrase.split() if wordsmod.normalize(x)]
    if not keys:
        return []
    have = [w.key for w in words]
    return [i for i in range(len(have) - len(keys) + 1) if have[i:i + len(keys)] == keys]


def _key_words(words: list[wordsmod.Word], quiet=None) -> list[int]:
    """The key word of each sentence, as the captions set it (the shared picker's `moments.key_word`):
    never a word the speaker took back, a first attempt said again or an aside, which are read from
    `words`, the clip's whole words, fillers and all."""
    back = moments.taken_back(words, quiet or [])
    out, start = [], 0
    for i, w in enumerate(words):
        if w.ends_sentence or i == len(words) - 1:
            k = cap.key_word(words[start:i + 1], quiet, back)
            if k is not None:
                out.append(start + k)
            start = i + 1
    return out


def gather(clip, opts: dict, notes: list[str]) -> tuple[list[Event], str]:
    """Every moment to mark, before they are put on frames and thinned."""
    from . import PlanError

    events: list[Event] = []
    want = opts["on"]
    plan, plan_path = (None, None)
    if want in ("auto", "cuts", "cards"):
        plan, plan_path = _plan_for(clip, opts["events"])
    source = "given"
    if want == "auto":
        if plan is not None and plan.get("effect") == "clean-cut":
            want = "cuts"
        elif plan is not None and _plan_cards(plan):
            want = "cards"
        else:
            want = "cuts" if not opts["sounds"].strip() else "none"
    kind_for = opts["kind"]

    if want == "cuts":
        cuts = [(t, label, "plan") for t, label in _plan_cuts(plan)] if plan is not None else []
        if plan is not None and not cuts and plan.get("effect") != "clean-cut":
            raise PlanError(f"the plan at {plan_path} is a {plan.get('effect')} plan, which makes no cuts; try on=cards")
        seen = picture_cuts(clip.info)
        if not cuts:
            cuts = [(j / clip.info.rate, f"cut {k + 1}", "picture") for k, j in enumerate(seen)]
            source = "the picture (no plan beside the video)"
        else:
            source = f"the plan {plan_path.name}, checked against the picture"
        for t, label, src in cuts:
            events.append(Event(t, AUTO_KIND["cuts"] if kind_for == "auto" else kind_for, label, src))
        if not cuts and opts["on"] == "cuts":
            notes.append("no cut was found in the picture")
        clip._seen_cuts = seen  # kept for the check below
    elif want == "cards":
        if plan is None:
            raise PlanError("on=cards reads the plan of the effect that laid the cards (<video>.plan.json beside "
                            "the video); none is there, so give its path with events=…")
        cards = _plan_cards(plan)
        if not cards:
            raise PlanError(f"the plan at {plan_path} ({plan.get('effect')}) lays no cards")
        for t, label, suits in cards:
            k = kind_for if kind_for != "auto" else (suits or AUTO_KIND["cards"])
            events.append(Event(t, k, label, "plan"))
        source = f"the plan {plan_path.name}"
    elif want == "words":
        ws = clip.words
        if opts["words"].strip():
            for phrase in [p.strip() for p in re.split(r"[,;\n]", opts["words"]) if p.strip()]:
                found = _find_words(ws, phrase)
                if not found:
                    notes.append(f"“{phrase}” is never said (check the words with fx.run words, and fix=)")
                for i in found:
                    events.append(Event(ws[i].start, AUTO_KIND["words"] if kind_for == "auto" else kind_for,
                                        f"“{ws[i].clean}”", "words"))
            source = "the words given"
        else:
            for i in _key_words(ws, clip.quiet):
                events.append(Event(ws[i].start, AUTO_KIND["words"] if kind_for == "auto" else kind_for,
                                    f"“{ws[i].clean}”", "words"))
            source = "the key word of each sentence"

    given = _parse_given(opts["sounds"])
    for at, kind in given:
        if kind not in KINDS:
            raise O.OptionError(f"sounds: no sound called {kind!r}; the sounds are {', '.join(KINDS)}")
        try:
            t = float(at.rstrip("s"))
            label = f"{t:.2f} s"
        except ValueError:
            found = _find_words(clip.words, at)
            if not found:
                notes.append(f"“{at}” is never said, so its {kind} was left out")
                continue
            t = clip.words[found[0]].start
            label = f"“{clip.words[found[0]].clean}”"
            if len(found) > 1:
                notes.append(f"“{at}” is said {len(found)} times; the {kind} is on the first")
        events.append(Event(t, kind, label, "given"))
    if given:
        source = "the sounds given" if want == "none" else f"{source}, and the sounds given"
    return sorted(events, key=lambda e: e.t), source


def on_frames(events: list[Event], clip, notes: list[str]) -> tuple[list[Event], list[Event]]:
    """Each moment on the first frame that shows it, a cut checked against the picture; then the
    ones too close to the one before dropped. Returns (kept, dropped)."""
    fps = clip.info.rate
    n = clip.info.frames
    seen = getattr(clip, "_seen_cuts", None) or []
    for e in events:
        j = int(math.ceil(e.t * fps - 1e-6))
        if e.source == "plan" and e.label.startswith("cut") and seen:
            near = [c for c in seen if abs(c - j) <= 2]
            if near:
                c = min(near, key=lambda c: abs(c - j))
                e.note = "on the planned frame, where the picture changes" if c == j else \
                    f"moved {c - j:+d} frame(s) to where the picture changes"
                j = c
            else:
                e.note = "no change in the picture there"
        e.frame = int(min(n - 1, max(0, j)))
        e.at = e.frame / fps
    kept: list[Event] = []
    dropped: list[Event] = []
    for e in sorted(events, key=lambda e: e.at):
        if kept:
            prev = kept[-1]
            room = max(APART.get(prev.kind, APART_STRUCK), APART.get(e.kind, APART_STRUCK))
            if e.at - prev.at < room:
                e.note = f"{(e.at - prev.at) * 1000:.0f} ms after the {prev.kind} on {prev.label}: dropped"
                dropped.append(e)
                continue
        kept.append(e)
    if dropped:
        notes.append(f"{len(dropped)} moment(s) too close to the one before were left without a sound")
    return kept, dropped


# ---- the mix -------------------------------------------------------------------------------------------

@dataclass
class Sounds:
    clip: object
    opts: dict
    events: list[Event]
    voice_lufs: float
    limited_db: float = 0.0
    lag_ms: float = 0.0
    _stem: Optional[np.ndarray] = None
    _voice: Optional[np.ndarray] = None
    measured: list[dict] = field(default_factory=list)

    def variant(self, e: Event) -> tuple[np.ndarray, float]:
        s = library()[e.kind]
        x, anchor = s.samples, s.anchor
        if abs(e.pitch - 1.0) > 1e-4:
            x = sound.resample(x, e.pitch)
            anchor = anchor / e.pitch
        if e.mirror:
            x = x[:, ::-1]
        return np.ascontiguousarray(x), anchor

    def mix(self, voice: np.ndarray, rate: int) -> np.ndarray:
        stem = np.zeros_like(voice)
        for e in self.events:
            x, anchor = self.variant(e)
            sound.lay(stem, x, e.at, rate, anchor=anchor, level=sound.gain(e.gain_db))
        out = voice + stem
        # A sound on a loud word could push the mix over full scale. The mix is held to the voice's own
        # highest true peak (or -1 dBTP when the voice peaks lower), so where no sound was added the
        # clip's own sound is never touched, however near full scale it already was.
        if stem.any():
            ceiling = max(-1.0, loudness.true_peak_db(voice) + 0.05)
            if loudness.true_peak_db(out) > ceiling:
                out, self.limited_db = loudness.limit(out, ceiling)
        self._stem, self._voice = stem, voice.copy()
        return out.astype(np.float32)

    def stems(self) -> tuple[np.ndarray, np.ndarray]:
        if self._stem is None:
            self.mix(self.clip.sound, RATE)
        return self._voice, self._stem

    # -- after the render: each sound found again in the file
    def check_file(self, out: Path) -> list[dict]:
        y = sound.read(out)
        x = self.clip.sound
        n = min(len(x), len(y))
        # The voice in the file against the voice given: an encoder delay would move both the voice
        # and the sounds, so the sounds are found in the file's own time, the voice taken away.
        lag = _lag(x[:n].mean(axis=1), y[:n].mean(axis=1))
        self.lag_ms = lag * 1000.0 / RATE
        shifted = np.zeros((n, x.shape[1]), np.float32)
        if lag >= 0:
            shifted[lag:] = x[: n - lag]
        else:
            shifted[: n + lag] = x[-lag:n]
        res = (y[:n] - shifted).mean(axis=1)
        fps = self.clip.info.rate
        rows = []
        for e in self.events:
            tmpl, anchor = self.variant(e)
            tmpl = tmpl.mean(axis=1) * sound.gain(e.gain_db)
            start = int(round((e.at - anchor) * RATE))
            a, b = max(0, start - int(0.05 * RATE)), min(len(res), start + len(tmpl) + int(0.05 * RATE))
            found = None
            if b - a > len(tmpl) // 4:
                seg = res[a:b]
                xc = np.correlate(seg, tmpl[: max(1, min(len(tmpl), b - a - 1))], mode="valid")
                if len(xc):
                    k = int(np.argmax(xc))
                    found = (a + k) / RATE + anchor  # where its landing point is in the file
            onset = None
            if found is not None and library()[e.kind].lands == "attack":
                lo, hi = int((found - 0.02) * RATE), int((found + 0.04) * RATE)
                part = np.abs(res[max(0, lo):hi])
                if len(part):
                    over = np.flatnonzero(part > 0.3 * part.max())
                    onset = (max(0, lo) + int(over[0])) / RATE if len(over) else None
            lands = onset if onset is not None else found
            piece = self._stem[max(0, start):max(0, start) + len(tmpl)]
            loud = sound.loudest(piece) if len(piece) else sound.SILENCE_DB
            rows.append({"event": e.label, "kind": e.kind, "frame": e.frame, "frame_at": round(e.at, 4),
                         "lands_in_file": None if lands is None else round(lands, 4),
                         "off_ms": None if lands is None else round((lands - e.at) * 1000, 1),
                         "off_frames": None if lands is None else round((lands - e.at) * fps, 2),
                         "under_voice_db": round(self.voice_lufs - loud, 1), "note": e.note})
        self.measured = rows
        return rows


def _lag(a: np.ndarray, b: np.ndarray, most: int = 4800) -> int:
    """How many samples `b` runs late against `a` (the encoder's delay, if any), from their correlation."""
    n = min(len(a), len(b), 10 * RATE)
    if n < 2 * most:
        return 0
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(b[:n], size) * np.conj(np.fft.rfft(a[:n], size)), size)
    lags = np.concatenate([xc[:most + 1], xc[-most:]])
    k = int(np.argmax(lags))
    return k if k <= most else k - len(lags)


# ---- the effect -------------------------------------------------------------------------------------

def _from_plan(plan: dict, clip) -> list[Event]:
    from . import PlanError

    items = plan.get("sounds")
    if not isinstance(items, list):
        raise PlanError("the plan has no sounds list; run once without --plan to get one to edit")
    out = []
    for it in items:
        try:
            t, kind = float(it["at"]), str(it["kind"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"each sound needs an at (seconds) and a kind: {it!r}") from None
        if kind not in KINDS:
            raise PlanError(f"no sound called {kind!r}; the sounds are {', '.join(KINDS)}")
        out.append(Event(t, kind, str(it.get("event", f"{t:.2f} s")), "plan (edited)"))
    return out


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job

    info = clip.info
    fps = info.rate
    L = info.frames / fps
    tl = Timeline.whole(L)
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, None, shape="keep")
    notes: list[str] = []
    if plan is not None:
        events, source = _from_plan(plan, clip), "the plan given"
    else:
        events, source = gather(clip, opts, notes)
    if not events:
        raise O.OptionError("nothing to put a sound on: " + ("; ".join(notes) + "; " if notes else "")
                            + "say on=words words=…, give sounds=\"3.2 pop\", or run it on a video an effect made")
    kept, dropped = on_frames(events, clip, notes)
    voice = sound.integrated(sound.blocks(clip.sound)) if info.has_audio else sound.SILENCE_DB
    reference = voice if voice > -70 else -23.0
    counts: dict[str, int] = {}
    for e in kept:
        lib = library()[e.kind]
        e.gain_db = (reference + opts["level"]) - lib.loudest
        k = counts.get(e.kind, 0)
        counts[e.kind] = k + 1
        if opts["variety"] > 0 and k > 0:
            u = float(np.random.default_rng(7919 * (KINDS.index(e.kind) + 1) + k).uniform(-1, 1))
            e.pitch = float(2 ** (u * opts["variety"] * 1.5 / 12))
            e.mirror = e.kind in SWELLS and k % 2 == 1
    sfx = Sounds(clip, opts, kept, reference)

    hits = [e.at for e in kept][:6]
    sheet_times = sorted(set(round(min(L - 0.5 / fps, t + 0.5 / fps), 3) for t in hits))
    while len(sheet_times) < 6:
        sheet_times.append(round(L * (len(sheet_times) + 0.5) / 6, 3))
    sheet_times = sorted(sheet_times)[:6]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(fps, 3), "seconds": round(L, 3),
                           "framing": fr.note},
                "moments_from": source, "voice_lufs": round(voice, 1), "level_db": opts["level"],
                "sounds": [e.as_dict() for e in kept], "dropped": [e.as_dict() for e in dropped],
                "notes": notes, "sheet_times": sheet_times}
    by_kind = ", ".join(f"{n} {k}" for k, n in counts.items())
    checks = {"moments": f"{len(kept)} from {source}" + (f" ({len(dropped)} too close, dropped)" if dropped else ""),
              "sounds": by_kind or "none",
              "level": f"each sound's loudest tenth of a second {-opts['level']:g} dB under the voice "
                       f"({reference:.1f} LUFS)",
              "frames": "; ".join(f"{e.label} on frame {e.frame} ({e.at:.3f} s)" + (f", {e.note}" if e.note else "")
                                  for e in kept[:8]) + (" …" if len(kept) > 8 else "")}
    if notes:
        checks["notes"] = "; ".join(notes)

    def stills() -> list[float]:
        return [min(L - 0.5 / fps, e.at + 0.5 / fps) for e in kept[:4]] or [L / 2]

    def frames_of(path: str):
        vinfo = video.probe(path)
        return lambda t: video.read_frame(vinfo, vinfo.frame_at(t))

    def preview(out: Path) -> dict:
        sheet = _sheet(sfx, frames_of(clip.path), out.with_suffix(".sound.jpg"), checked=False)
        return {"sound sheet": str(sheet)}

    def after(result: dict, out: Path) -> dict:
        rows = sfx.check_file(out)
        sheet = _sheet(sfx, frames_of(str(out)), out.with_suffix(".sound.jpg"), checked=True)
        offs = [abs(r["off_frames"]) for r in rows if r["off_frames"] is not None]
        unders = [r["under_voice_db"] for r in rows]
        lines = {"landing": (f"{sum(1 for o in offs if o < 1.0)} of {len(rows)} sounds land within a frame of their "
                             f"moment in the encoded file (largest {max(offs) * 1000 / fps:.1f} ms)" if offs else
                             "no sound could be found again in the file"),
                 "under the voice": (f"{min(unders):.1f} to {max(unders):.1f} dB" if unders else "-"),
                 "voice in the file": ("in step with the video given" if abs(sfx.lag_ms) < 0.05 else
                                       f"{sfx.lag_ms:+.2f} ms against the video given")}
        if sfx.limited_db > 0:
            lines["limited"] = f"peaks held under -1 dBFS ({sfx.limited_db:.1f} dB at most)"
        lines["sound sheet"] = str(sheet)
        return lines

    return Job(NAME, clip, tl, fr, None, out_plan, checks=checks, still_times=stills, after=after,
               sound=sfx.mix, preview=preview)


def _sheet(sfx: Sounds, frame_at, path: Path, checked: bool) -> Path:
    voice, stem = sfx.stems()
    clip = sfx.clip
    L = clip.info.frames / clip.info.rate
    lanes = [soundsheet.Lane("voice", voice, (196, 196, 206)),
             soundsheet.Lane("sounds", stem, (255, 150, 70))]
    marks = [soundsheet.Mark(e.at, e.kind, (255, 206, 84)) for e in sfx.events]
    by_label = {r["event"] + str(r["frame"]): r for r in sfx.measured}
    zooms = []
    for e in sfx.events[:12]:
        r = by_label.get(e.label + str(e.frame))
        lands = library()[e.kind].lands
        before, after = (0.55, 0.12) if lands in ("peak", "end") else (0.08, 0.2)
        if e.kind == "riser":
            before = 1.0
        note = (f"lands {r['off_ms']:+.1f} ms ({r['off_frames']:+.2f} frames)" if r and r["off_ms"] is not None
                else "planned on the frame's start")
        zooms.append(soundsheet.Zoom(e.at, f"{e.label}: {e.kind}", lanes=("voice", "sounds"),
                                     onset=r["lands_in_file"] if r else None, before=before, after=after, note=note))
    unders = [r["under_voice_db"] for r in sfx.measured]
    lines = [f"{len(sfx.events)} sounds; each {-sfx.opts['level']:g} dB under the voice ({sfx.voice_lufs:.1f} LUFS); "
             f"variety {sfx.opts['variety']:g}",
             ("checked in the encoded file: " + (f"every landing within {max(abs(r['off_ms']) for r in sfx.measured if r['off_ms'] is not None):.1f} ms of its frame; "
                                                 if any(r["off_ms"] is not None for r in sfx.measured) else "")
              + (f"{min(unders):.1f} to {max(unders):.1f} dB under the voice" if unders else ""))
             if checked else "planned (not rendered yet): each sound's landing on the start of its frame",
             "close-ups: yellow is the frame where the moment starts; green is where the sound was found in the file"]
    fps = clip.info.rate
    tiles = [min(L - 0.5 / fps, e.at + 0.5 / fps) for e in sfx.events[:6]]
    k = 0
    while len(tiles) < 6:  # fewer than six sounds: fill in across the film
        tiles.append(L * (k + 0.5) / 6)
        k += 1
    sheet = soundsheet.Sheet(title=f"sound-effects · {Path(clip.path).name}", duration=L, fps=fps,
                             frame_at=frame_at, lanes=lanes, frames=sorted(tiles),
                             marks=marks, zooms=zooms, lines=lines)
    return soundsheet.draw(sheet, path)
