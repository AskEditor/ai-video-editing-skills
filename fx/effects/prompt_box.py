"""prompt-box: a prompt types itself over the clip, and the edit answers it on the same beat.

How a prompt becomes an edit on screen:

1. **The ask and its answer.** `text` is what is typed ("make it moody"); several prompts are
   separated by `|`. Each is answered by another effect of this pack with its options (`answer`,
   "grade look=moody"), or by the effect its own words ask for (`answer=auto`: "make it black and
   white" is grade, black-and-white). Any effect that keeps the clip's length and shape can answer.
2. **The moment.** `when` is when the prompt is sent, and so when its answer lands: `auto` (the
   start of a spoken line, spread across the clip when there are several), seconds, or words that
   are said ("moody", found in the corrected word times). `auto` takes, near each prompt's share
   of the clip, the start of the strongest line (`moments.strong_starts`: a sentence's start before
   a clause's), never a line the speaker took back or a first attempt. The answer is told the moment through
   its own options, or shown from it, and the run measures the first frame the answer changes: it
   is the frame the prompt is sent on. `until` ends an answer; by default it stays (a look and
   captions to the end, a headline for as long as it lasts), so several prompts build the edit up.
3. **The box** is the pack's own composer bar, `light`, `dark` or `glass`: a placeholder and a
   blinking caret, the prompt typed in a person's rhythm (quick runs inside a word, a longer
   reach between words, the same every run), the send button lighting once there is text. On the
   send the button is pressed, the bar pulses and a ripple leaves it; a change to the whole
   picture also gets a soft flash. Then it leaves.
4. **The place.** Above the head or below the chin, inside the safe zone, clear of the whole head
   over every frame it is up, and clear of the other graphics where the band has room. Where it
   has not, an earlier graphic steps aside while the box is up; a graphic that lands where an
   earlier one stands takes its place while it shows.
5. **The sound** is the clip's own, with soft key ticks and a send tick laid well under the voice
   (their level is measured against the voice and printed); `sound=off` keeps it exactly.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, onto_layer, over
from ..core.layout import Rect, SafeZone
from ..core.words import Word, normalize
from . import layers
from . import options as O

NAME = "prompt-box"
STYLES = ("light", "dark", "glass")
BEATS = ("auto", "ripple", "flash", "both", "off")

# The test matrix's runs (tests/matrix.py): one prompt on each controlled clip, its answer read from
# its own words, in each style; the edit building itself on the plain wall (three prompts, the
# title sent on the words it sets); and a prompt sent on a spoken word.
MATRIX = [
    ("A", "A", {"text": "make it black and white"}),
    ("B", "B", {"text": "make it warm", "style": "glass"}),
    ("C", "C", {"text": "captions on", "style": "dark"}),
    ("D", "D", {"text": "make it moody | captions on | big title",
                "answer": "grade look=moody | captions | headline", "when": "auto | auto | my hotel"}),
    ("A-on-hooks", "A", {"text": "black and white", "when": "hooks", "style": "dark"}),
]


# ---- options of this effect's own kinds ------------------------------------------------------

@dataclass(frozen=True)
class Pace:
    """How fast the prompt types: slow, normal or fast, or a number (1 is a quick, natural typist)."""
    help: str

    def parse(self, raw: Any) -> float:
        v = str(raw).strip().lower()
        named = {"slow": 0.7, "normal": 1.0, "natural": 1.0, "fast": 1.4, "quick": 1.4}
        if v in named:
            return named[v]
        try:
            x = float(v.rstrip("x"))
        except ValueError:
            raise O.OptionError(f"say slow, normal or fast, or a number from 0.5 to 2.5 (got {raw!r})") from None
        if not 0.5 <= x <= 2.5:
            raise O.OptionError(f"give a number from 0.5 to 2.5 (got {raw!r})")
        return x


OPTIONS = {
    "text": O.Text("what is typed: one prompt, or several separated by | (\"make it moody | captions on | big "
                   "title\"), each sent in turn"),
    "answer": O.Text("what each prompt does: an effect of this pack with its options (\"grade look=moody\", "
                     "\"headline style=serif\"), several separated by | in the same order as the prompts; auto "
                     "reads it from the prompt's own words"),
    "when": O.Text("when each prompt is sent and its answer lands: auto (the start of a spoken line, spread "
                   "across the clip), seconds such as 2.5, or words that are said (\"moody\"); several "
                   "separated by |"),
    "until": O.Text("when each answer ends: auto (as long as the answer itself lasts: a look or captions to the "
                    "end), end, next (when the next prompt is sent), seconds, or words that are said"),
    "style": O.Choice(STYLES, "light: a white bar with dark type · dark: a near-black bar with white type · "
                              "glass: frosted glass over the picture, tinted so the type stays readable"),
    "position": O.Choice(("auto", "top", "bottom"), "auto: above the head when there is room, else below the "
                                                    "chin · top · bottom. Never over the face"),
    "speed": Pace("how fast it types: slow, normal or fast, or a number from 0.5 to 2.5 (1 is a quick, natural "
                  "typist)"),
    "sound": O.Switch("on lays soft key ticks and a send tick well under the voice; off keeps the clip's sound "
                      "exactly"),
    "beat": O.Choice(BEATS, "what ties the send to the answer: auto (a ripple from the box, plus a soft flash when "
                            "the answer changes the whole picture) · ripple · flash · both · off"),
    "accent": O.Colour("the send button and the ripple: auto (yellow), a name such as green or pink, or a hex "
                       "colour"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  (matters when when or until are words, and is "
                  "passed on to the answers)"),
}

CPS = 15.0  # letters a second at speed 1
ENTER = 0.22  # the box comes in
THINK = 0.2  # a beat before the first letter
BEFORE_SEND = 0.32  # the finished prompt waits this long before it is sent
PRESS = 0.09  # the send button goes down this long before the send
HOLD = 0.55  # the box stays this long after the send
QUICK = 0.0  # ... or not at all, when its own answer lands where it sits: it dissolves under it
EXIT = 0.24
EXIT_QUICK = 0.16
GAP = 0.2  # between one box leaving and the next coming
TAIL = 1.6  # the last answer is on screen at least this long
MAX_TYPING = 2.6  # a long prompt types faster rather than taking longer than this
REACH_LEAST = 1.5  # auto: seconds a prompt may always move to land on a stronger line
DRIFT = 0.5  # auto: what each second away from its share of the clip costs a line's strength
SENTENCE = 3.0  # auto: what opening a sentence adds, over opening a clause inside one
PULSE = 0.26
GLOW = 0.4
RIPPLE = 0.5
RIPPLE_REACH = 84  # how far the ripple travels from the bar, at 1080 wide
FLASH = 0.42
STEP_ASIDE = 0.15  # how long a graphic takes to step aside, and to come back
PLACEHOLDER = "Ask for an edit…"
FACE = "poppins-semibold"
GLASS_SMOKE = 0.24  # the least a glass bar is smoked
CLEAR = 0.02  # how far (a share of the height) the box keeps from its own answer to stay while it lands
ACCENT = "#ffd400"
ON_ACCENT = (22, 22, 24)


# ---- easing --------------------------------------------------------------------------------------

def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _ease_out(x: float) -> float:
    x = _clamp(x)
    return 1.0 - (1.0 - x) ** 3


def _smooth(x: float) -> float:
    x = _clamp(x)
    return x * x * (3 - 2 * x)


# ---- typing --------------------------------------------------------------------------------------

def rhythm(text: str, speed: float) -> tuple[list[float], float]:
    """When each letter appears, in seconds from the first, and how long the typing takes: a
    person's rhythm, quick inside a word and a longer reach between words, the same every run."""
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    base = 1.0 / (CPS * speed)
    t, out = 0.0, []
    for i, ch in enumerate(text):
        out.append(t)
        step = base * float(np.exp(rng.normal(0.0, 0.32)))
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch == " ":
            step *= 1.5
        elif ch in ",.;:!?":
            step *= 2.0
        elif nxt == ch:
            step *= 0.75
        t += min(3.0 * base, max(0.4 * base, step))
    total = t
    if total > MAX_TYPING:
        k = MAX_TYPING / total
        out = [x * k for x in out]
        total = MAX_TYPING
    return out, total


def lead_time(typing: float, speed: float) -> float:
    """From the box coming in to the send."""
    return ENTER + THINK / speed + typing + BEFORE_SEND / speed


# ---- finding words and moments ---------------------------------------------------------------

def _key(w: str) -> str:
    k = normalize(w)
    return k[:-1] if len(k) >= 4 and k.endswith("s") else k


def find_words(words: list[Word], phrase: str, after: float = -1.0) -> Optional[int]:
    """The first word of the first place `phrase` is said, starting at or after `after`."""
    want = [_key(x) for x in phrase.split() if normalize(x)]
    if not want:
        return None
    keys = [_key(w.text) for w in words]
    for i in range(len(keys) - len(want) + 1):
        if words[i].start >= after - 1e-6 and keys[i:i + len(want)] == want:
            return i
    return None


def _moment(raw: str) -> Any:
    """auto, seconds, or words: as given."""
    v = raw.strip().strip("\"'“”").strip()
    if v.lower() in ("", "auto"):
        return "auto"
    try:
        x = float(v.lower().rstrip("s"))
    except ValueError:
        return v
    if x < 0:
        raise O.OptionError(f"when: give seconds from the start, 0 or more (got {raw!r})")
    return x


# ---- the look --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Look:
    fill: Optional[tuple]  # None: frosted glass
    fill_a: float
    edge: tuple
    edge_a: float
    ink: tuple
    hint: tuple
    hint_a: float
    off: tuple
    off_a: float
    off_icon: tuple
    shadow_a: float


LOOKS = {
    "light": Look((250, 250, 251), 0.97, (0, 0, 0), 0.10, (22, 22, 26), (128, 128, 138), 1.0,
                  (224, 224, 229), 1.0, (150, 150, 160), 0.30),
    "dark": Look((20, 20, 24), 0.9, (255, 255, 255), 0.14, (246, 246, 248), (138, 138, 148), 1.0,
                 (58, 58, 66), 1.0, (150, 150, 160), 0.38),
    "glass": Look(None, 0.0, (255, 255, 255), 0.5, (255, 255, 255), (255, 255, 255), 0.66,
                  (255, 255, 255), 0.24, (255, 255, 255), 0.2),
}


def _marks(plan: Any) -> list[Rect]:
    """Every box an effect's plan says it draws in: each `rect` (four numbers) anywhere in it."""
    out: list[Rect] = []
    if isinstance(plan, dict):
        for k, v in plan.items():
            if k == "rect" and isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                out.append(Rect.of(v))
            else:
                out += _marks(v)
    elif isinstance(plan, list):
        for v in plan:
            out += _marks(v)
    return out


def _rrect_sdf(w: float, h: float, r: float, pad: float, dy: float = 0.0) -> np.ndarray:
    """Signed distance in pixels to a w×h box with corners of radius r, its top-left at (pad, pad + dy)
    of a canvas (w + 2·pad) × (h + 2·pad): negative inside, positive outside."""
    W, H = int(math.ceil(w + 2 * pad)), int(math.ceil(h + 2 * pad))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    qx = np.abs(xs + 0.5 - (pad + w / 2)) - (w / 2 - r)
    qy = np.abs(ys + 0.5 - (pad + dy + h / 2)) - (h / 2 - r)
    return (np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r).astype(np.float32)


def _paint(alpha: np.ndarray, rgb) -> Sprite:
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return Sprite(rgb=a[..., None] * np.asarray(rgb, np.float32)[None, None, :], a=a)


def _icon(kind: str, s: float, colour) -> Sprite:
    """The send button's mark, drawn four times larger and brought down: an arrow, or a tick."""
    from PIL import Image, ImageDraw

    k = 4
    S = max(4, int(round(s * k)))
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(2, int(round(S * 0.11)))
    fill = tuple(int(c) for c in colour) + (255,)
    if kind == "arrow":
        d.line([(S * 0.24, S * 0.5), (S * 0.74, S * 0.5)], fill=fill, width=lw)
        d.line([(S * 0.52, S * 0.27), (S * 0.75, S * 0.5), (S * 0.52, S * 0.73)], fill=fill, width=lw, joint="curve")
    else:
        d.line([(S * 0.26, S * 0.53), (S * 0.43, S * 0.7), (S * 0.75, S * 0.33)], fill=fill, width=lw, joint="curve")
    img = img.resize((max(1, int(round(s))), max(1, int(round(s)))), Image.LANCZOS)
    return Sprite.from_rgba(np.asarray(img))


def _disc(d: float, colour, opacity: float = 1.0) -> Sprite:
    n = int(math.ceil(d)) + 2
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    dist = np.hypot(xs + 0.5 - n / 2, ys + 0.5 - n / 2) - d / 2
    return _paint(np.clip(0.5 - dist, 0, 1) * opacity, colour)


class Bar:
    """The composer bar's measures and its pieces, drawn once for a run: every prompt in a run uses
    the same bar, so it never changes size between prompts."""

    def __init__(self, W: int, H: int, texts: list[str], style: str, accent: tuple):
        u = W / 1080.0
        self.u = u
        self.style = style
        self.look = LOOKS[style]
        self.accent = accent
        self.h = 128 * u
        self.r = 0.3 * self.h
        self.pad_l = 42 * u
        self.btn = 0.6 * self.h
        self.btn_pad = (self.h - self.btn) / 2
        zone_w = SafeZone(W, H).rect.w
        px = int(round(51 * u))

        def need(p: int) -> float:
            longest = max(textmod.advance(s, FACE, p) for s in texts + [PLACEHOLDER])
            return self.pad_l + longest + 28 * u + self.btn + self.btn_pad

        while need(px) > zone_w and px > 38 * u:
            px -= 1
        if need(px) > zone_w:
            long = max(texts, key=lambda s: textmod.advance(s, FACE, px))
            raise O.OptionError(f"text: “{long}” is too long to type on one line of the box; keep a prompt to "
                                f"about 34 letters")
        self.px = px
        self.w = min(zone_w, max(0.88 * zone_w, need(px)))
        self.cap = textmod.cap_height(FACE, px)
        self.baseline = self.h / 2 + self.cap / 2
        self.pad = math.ceil(40 * u)  # room round the bar on its canvas, for the shadow and the glow
        # white type on glass carries a soft shadow, so it reads on whatever shows through
        self.shadow = (0.0, 2.0 * u, 3.0 * u, 0.5) if style == "glass" else None
        P = self.pad
        d = _rrect_sdf(self.w, self.h, self.r, P)
        self.sdf = d
        fill = np.clip(0.5 - d, 0, 1)
        self.fill_mask = fill.astype(np.float32)
        sigma = 14 * u
        ds = _rrect_sdf(self.w, self.h, self.r, P, dy=8 * u)
        shadow = np.exp(-np.maximum(ds, 0) ** 2 / (2 * sigma * sigma)) * (1.0 - fill) * self.look.shadow_a
        bw = max(1.0, 1.6 * u)
        edge = np.clip(0.5 - d, 0, 1) - np.clip(0.5 - (d + bw), 0, 1)
        if style == "glass":  # glass catches the light along its top
            rows = np.linspace(1.0, 0.45, edge.shape[0], dtype=np.float32)[:, None]
            edge = edge * rows
        H_, W_ = d.shape
        base_rgb = np.zeros((H_, W_, 3), np.float32)
        base_a = np.zeros((H_, W_), np.float32)
        layer = (base_rgb, base_a)
        onto_layer(layer, _paint(shadow, (0, 0, 0)), 0, 0)
        if self.look.fill is not None:
            onto_layer(layer, _paint(fill * self.look.fill_a, self.look.fill), 0, 0)
        onto_layer(layer, _paint(edge * self.look.edge_a, self.look.edge), 0, 0)
        self.base = layer
        self.glow = _paint(np.exp(-np.maximum(d, 0) / (9 * u)) * (d > -0.5) * (1.0 - fill), accent)
        # the send button, in its three states, and where it sits on the canvas
        self.btn_at = (P + self.w - self.btn_pad - self.btn, P + (self.h - self.btn) / 2)
        icon = 0.62 * self.btn
        lk = self.look
        self.btn_off = self._button(_disc(self.btn, lk.off, lk.off_a), _icon("arrow", icon, lk.off_icon))
        self.btn_on = self._button(_disc(self.btn, accent), _icon("arrow", icon, ON_ACCENT))
        self.btn_done = self._button(_disc(self.btn, accent), _icon("tick", icon, ON_ACCENT))
        self.text_x = P + self.pad_l
        self.text_room = self.w - self.pad_l - self.btn - self.btn_pad - 20 * u
        self.placeholder = textmod.draw_word(PLACEHOLDER, FACE, px, fill=lk.hint, shadow=self.shadow)

    def _button(self, disc: Sprite, icon: Sprite) -> Sprite:
        n = disc.w
        rgb = disc.rgb.copy()
        a = disc.a.copy()
        layer = (rgb, a)
        onto_layer(layer, icon, (n - icon.w) / 2, (n - icon.h) / 2)
        return Sprite(rgb=rgb, a=a)


# ---- one prompt --------------------------------------------------------------------------------

@dataclass
class Prompt:
    index: int
    text: str
    spec: layers.Spec
    chosen: str  # how the answer was chosen
    when: Any  # "auto", seconds or words, as given
    until: Any
    send: float = 0.0
    how: str = ""
    keys: list = field(default_factory=list)  # output seconds of each letter
    typing: float = 0.0
    lead: float = 0.0
    appear: float = 0.0
    hold: float = HOLD
    exit: float = EXIT
    under: bool = False  # drawn under its own answer, which lands where it sits
    end: Optional[float] = None
    layer: Optional[layers.Layer] = None
    first: Optional[float] = None  # the first output time the answer changes the frame
    rect: Optional[Rect] = None  # the bar on the frame, at its full size
    where: str = ""
    scale: float = 1.0
    reach: float = 0.0  # how far the ripple travels from the bar
    keep_off: Optional[Rect] = None  # what the ripple fades out before reaching: the head
    warnings: list = field(default_factory=list)
    word: Optional[Sprite] = None
    word_ox: float = 0.0
    word_oy: float = 0.0
    cuts: list = field(default_factory=list)  # where the typed text ends after each letter
    offs: list = field(default_factory=list)  # each letter's time from the first, as typed

    @property
    def leave(self) -> float:
        return self.send + self.hold + self.exit

    def on(self, t: float) -> bool:
        return self.appear <= t < self.leave


# ---- the effect ----------------------------------------------------------------------------------

class PromptBox:
    def __init__(self, clip, opts: dict, base, timeline, plan: Optional[dict] = None):
        from .captions import head_in_output

        self.clip = clip
        self.opts = opts
        self.fr = base
        self.tl = timeline
        info = clip.info
        self.fps = info.rate
        self.n = timeline.output_frames(self.fps)
        self.seconds = self.n / self.fps
        self.W, self.H = base.out_w, base.out_h
        self.zone = SafeZone(self.W, self.H)
        head_at = head_in_output(clip, timeline, base)
        self.heads = [head_at(j) for j in range(self.n)]
        self.speed = float(opts["speed"])
        self.accent = textmod.colour(ACCENT if opts["accent"] == "auto" else opts["accent"])
        self.warnings: list[str] = []
        self.words = [w for w in clip.words if w.key]
        self._fields: dict = {}
        self.prompts = self._prompts(plan)
        self.bar = Bar(self.W, self.H, [p.text for p in self.prompts], opts["style"], self.accent)
        for p in self.prompts:
            self._type(p)
        if plan is None:
            self._send_times()
        self._answers(plan)
        self._probe()
        for p in self.prompts:
            self._place(p, plan)
        self._own_landing()
        self._step_aside()
        self._glass_tint()

    # -- reading the options
    def _prompts(self, plan: Optional[dict]) -> list[Prompt]:
        if plan is not None:
            return self._prompts_from_plan(plan)
        texts = [t for t in layers.split(self.opts["text"])]
        if not any(texts) or any(not t.strip() for t in texts):
            raise O.OptionError("text: give the prompt to type, such as text=\"make it moody\" (several separated by |)")
        n = len(texts)
        if n > 6:
            raise O.OptionError(f"text: at most six prompts in one clip (got {n})")

        def per(name: str) -> list[str]:
            items = layers.split(self.opts[name])
            if len(items) == 1:
                return items * n
            if len(items) != n:
                raise O.OptionError(f"{name}: give one for every prompt ({n}), separated by |, or one for all "
                                    f"(got {len(items)})")
            return items

        answers, whens, untils = per("answer"), per("when"), per("until")
        out = []
        for i, (text, ans, when, until) in enumerate(zip(texts, answers, whens, untils)):
            text = text.strip().strip("\"“”")
            if ans.strip().lower() in ("", "auto"):
                spec, why = layers.guess(text, exclude=(NAME,))
                if spec is None:
                    raise O.OptionError(f"answer: {why}; say which effect answers it, such as "
                                        f"answer=\"grade look=moody\"")
                chosen = f"read from the prompt: {why}"
            else:
                spec = layers.parse(ans)
                chosen = "as given"
            if spec.name == NAME:
                raise O.OptionError("answer: a prompt box cannot answer a prompt")
            out.append(Prompt(i, text, spec, chosen, _moment(when), _moment(until)))
        return out

    def _prompts_from_plan(self, plan: dict) -> list[Prompt]:
        from . import PlanError

        items = plan.get("prompts")
        if not isinstance(items, list) or not items:
            raise PlanError("the plan has no prompts; run once without --plan to get one to edit")
        out, last = [], -1.0
        for i, it in enumerate(items):
            try:
                text, send = str(it["text"]).strip(), float(it["send"])
                spec = layers.parse(str(it["answer"]))
            except (KeyError, TypeError, ValueError) as e:
                raise PlanError(f"prompt {i + 1} of the plan needs text, answer and send ({e})") from None
            if not text:
                raise PlanError(f"prompt {i + 1} of the plan has no text")
            if not 0 < send < self.seconds:
                raise PlanError(f"prompt {i + 1} is sent at {send:.2f} s, outside the video's {self.seconds:.2f} s")
            if send <= last:
                raise PlanError(f"prompt {i + 1} is sent before the one before it; keep them in order")
            last = send
            end = it.get("until")
            p = Prompt(i, text, spec, "from the plan", send, end if end is None else float(end))
            p.send, p.how = send, "from the plan"
            out.append(p)
        return out

    # -- typing and timing
    def _type(self, p: Prompt) -> None:
        offs, p.typing = rhythm(p.text, self.speed)
        p.lead = lead_time(p.typing, self.speed)
        p.offs = offs
        w = textmod.draw_word(p.text, FACE, self.bar.px, fill=self.bar.look.ink, shadow=self.bar.shadow)
        p.word, p.word_ox, p.word_oy = w.sprite, w.ox, w.oy
        p.cuts = [w.ox + textmod.advance(p.text[:k + 1], FACE, self.bar.px) for k in range(len(p.text))]

    def _set_keys(self, p: Prompt) -> None:
        start = p.send - BEFORE_SEND / self.speed - p.typing
        p.keys = [start + o for o in p.offs]
        p.appear = p.send - p.lead

    def _after(self, p: Prompt) -> float:
        return HOLD + EXIT + GAP

    def _send_times(self) -> None:
        """Each prompt's send: as given, or (auto) the start of a spoken line, spread from the first
        moment the first prompt can be typed to the last that leaves its answer time to be seen."""
        ps = self.prompts
        n = len(ps)
        # Where a spoken line begins, each weighed by the line that begins there (the core's picker:
        # the start of a line the speaker took back, or of a first attempt said again, is never one).
        starts = moments.strong_starts(self.words, self.clip.quiet, moments.taken_back(self.clip.words, self.clip.quiet))
        # The earliest each can be sent (each box typed after the one before has gone), and the
        # latest (each leaving room for the ones after it, and the last for its answer to be seen).
        lo = [0.0] * n
        hi = [0.0] * n
        for k, p in enumerate(ps):
            lo[k] = (0.25 if k == 0 else lo[k - 1] + self._after(ps[k - 1])) + p.lead
        for k in range(n - 1, -1, -1):
            hi[k] = self.seconds - TAIL if k == n - 1 else hi[k + 1] - ps[k + 1].lead - self._after(ps[k])
        if any(h < l for h, l in zip(hi, lo)):
            raise O.OptionError(f"the clip is too short for {n} prompt{'s' if n > 1 else ''}: each needs about "
                                f"{ps[0].lead + HOLD + EXIT:.1f} s; use fewer prompts or a faster speed")
        span = max(0.0, hi[-1] - lo[0])
        # How far from its share of the clip a prompt may move to land on a stronger line: half the
        # way to the next prompt's share (the whole clip's half for one prompt), never under REACH_LEAST.
        reach = max(REACH_LEAST, 0.5 * (span * 0.85 / (n - 1) if n > 1 else span))
        prev_send = None
        for k, p in enumerate(ps):
            low = lo[k] if prev_send is None else max(lo[k], prev_send + self._after(ps[k - 1]) + p.lead)
            if isinstance(p.when, float):
                p.send, p.how = p.when, f"at {p.when:.2f} s"
                if p.send - p.lead < 0.05:
                    self._squeeze(p, p.send - 0.05)
            elif p.when != "auto":
                i = find_words(self.words, p.when, after=prev_send if prev_send is not None else -1.0)
                if i is None:
                    raise O.OptionError(f"when: the words \"{p.when}\" are not said in this video"
                                        + (" after the prompt before it" if prev_send is not None else "")
                                        + "; read the words with `python -m fx.run words <video>`, or give seconds")
                p.send = self.words[i].start
                p.how = f"on “{self.words[i].clean}”, as it is said"
                if p.send - p.lead < 0.05:
                    self._squeeze(p, p.send - 0.05)
            else:
                target = lo[0] + (span * 0.85 * k / (n - 1) if n > 1 else 0.2 * span)
                target = max(target, low)
                top = hi[k]
                fits = [s for s in starts if low - 1e-6 <= s.at <= top + 1e-6]
                strong = [s for s in fits if s.strength is not None and abs(s.at - target) <= reach]
                if strong:  # the strongest line near this prompt's share of the clip, the nearer on a tie
                    s = max(strong, key=lambda s: s.strength + (SENTENCE if s.sentence else 0.0)
                            - DRIFT * abs(s.at - target))
                    said = " ".join(w.clean for w in self.words[s.line[0]:s.line[1] + 1])
                    p.send = s.at
                    p.how = f"auto: the start of a line, “{self.words[s.index].clean}” (the line “{said}”)"
                elif fits:  # no line stands near: the nearest start
                    s = min(fits, key=lambda s: abs(s.at - target))
                    p.send = s.at
                    p.how = f"auto: the start of a line, “{self.words[s.index].clean}” (no strong line near)"
                else:
                    p.send = min(max(target, low), max(low, top))
                    p.how = f"auto: {p.send:.2f} s (no line starts there)"
            if prev_send is not None and p.send - p.lead < prev_send + HOLD + EXIT + 0.05:
                # Too close to the one before: that box leaves sooner, and this one types faster.
                prev = ps[k - 1]
                room = p.send - prev_send - EXIT - 0.05
                prev.hold = max(QUICK, min(prev.hold, room - p.lead))
                if p.send - p.lead < prev_send + prev.hold + EXIT + 0.05:
                    self._squeeze(p, p.send - (prev_send + prev.hold + EXIT + 0.05))
                p.warnings.append("sent soon after the prompt before it: that box leaves sooner and this one "
                                  "types faster")
            if prev_send is not None and p.send <= prev_send + 0.3:
                raise O.OptionError(f"when: prompt {k + 1} is sent at {p.send:.2f} s, too close to the one before "
                                    f"it ({prev_send:.2f} s)")
            if p.send >= self.seconds - 0.3:
                raise O.OptionError(f"when: prompt {k + 1} would be sent at {p.send:.2f} s, at the very end of the "
                                    f"video")
            self._set_keys(p)
            prev_send = p.send

    def _squeeze(self, p: Prompt, room: float) -> None:
        """Type faster so the whole lead fits in `room` seconds."""
        if room < 0.6:
            raise O.OptionError(f"when: prompt {p.index + 1} is sent at {p.send:.2f} s, too early to type "
                                f"“{p.text}” before it; send it later")
        fixed = ENTER + THINK / self.speed + BEFORE_SEND / self.speed
        k = max(0.2, (room - fixed) / max(1e-3, p.typing))
        p.offs = [o * k for o in p.offs]
        p.typing *= k
        p.lead = lead_time(p.typing, self.speed)
        if k < 0.999:
            p.warnings.append(f"typed {1 / k:.1f}× faster to fit before its moment")

    def _ends(self, p: Prompt, nxt: Optional[Prompt]) -> Optional[float]:
        u = p.until
        if u == "auto" or u is None:
            return None
        if isinstance(u, float):
            if u <= p.send:
                raise O.OptionError(f"until: prompt {p.index + 1}'s answer would end ({u:.2f} s) before it lands "
                                    f"({p.send:.2f} s)")
            return min(u, self.seconds)
        low = str(u).lower()
        if low in ("end", "the end"):
            return self.seconds
        if low == "next":
            return nxt.send if nxt is not None else None
        i = find_words(self.words, str(u), after=p.send + 0.05)
        if i is None:
            raise O.OptionError(f"until: the words \"{u}\" are not said after prompt {p.index + 1} is sent")
        return self.words[i].start

    # -- the answers
    def _answers(self, plan: Optional[dict]) -> None:
        size = (self.W, self.H)
        saved = (plan or {}).get("prompts") or []
        for k, p in enumerate(self.prompts):
            nxt = self.prompts[k + 1] if k + 1 < len(self.prompts) else None
            if plan is not None:
                self._set_keys(p)
                p.end = p.until if isinstance(p.until, float) else None
                inner = None
                lay = saved[k].get("layer") if k < len(saved) else None
                if isinstance(lay, dict) and abs(float(lay.get("moment", -1)) - p.send) < 1e-3:
                    inner = lay.get("plan")
            else:
                p.end = self._ends(p, nxt)
                inner = None
            try:
                p.layer = layers.prepare(self.clip, p.spec, p.send, p.end, self.fr, fix=self.opts["fix"], plan=inner)
            except ValueError as e:  # the answer's own plain words, said for the prompt that asked for it
                raise O.OptionError(f"answer {k + 1} ({p.spec.name}): {e}") from None
            # The first frame the answer changes, as it will be drawn.
            near = layers.first_change(p.layer, size, self.fps, self.n, p.send - 0.5, min(self.seconds, p.send + 3.0),
                                       gated=True)
            raw = layers.first_change(p.layer, size, self.fps, self.n, p.send - 1.0, p.send + 0.02) \
                if p.layer.told else None
            # An answer that begins its entrance before its moment (a word that rises into place) is
            # told a moment later by that much, so its entrance starts on the send.
            if raw is not None and raw < p.send - 1.5 / self.fps and plan is None:
                lead = p.send - raw
                try:
                    later = layers.prepare(self.clip, p.spec, p.send + lead, p.end, self.fr, fix=self.opts["fix"])
                    later.start = p.send
                    again = layers.first_change(later, size, self.fps, self.n, p.send - 1.0, p.send + 0.02)
                    if again is None or again >= p.send - 1.5 / self.fps:
                        p.layer = later
                        near = layers.first_change(later, size, self.fps, self.n, p.send - 0.5,
                                                   min(self.seconds, p.send + 3.0), gated=True)
                except ValueError:
                    pass
            p.first = near
            late = None if near is None else near - p.send
            if near is None:
                p.warnings.append(f"{p.spec.name} shows nothing within 3 s of the send")
            elif late > 2.5 / self.fps:
                if p.when == "auto" and plan is None and late <= 1.2 and self._can_move(k, near):
                    p.send = near
                    p.how += f"; moved to {near:.2f} s, where {p.spec.name} lands"
                    self._set_keys(p)
                    p.layer.start = near
                else:
                    p.warnings.append(f"{p.spec.name} lands {late:.2f} s after the send (it waits for its own moment)")

    def _can_move(self, k: int, t: float) -> bool:
        p = self.prompts[k]
        nxt = self.prompts[k + 1] if k + 1 < len(self.prompts) else None
        if nxt is not None and t + HOLD + EXIT + GAP > nxt.send - nxt.lead:
            return False
        return t < self.seconds - TAIL

    # -- where the other graphics are
    def _probe(self) -> None:
        """What each graphic answer covers, on the frames where something else could meet it: while a
        box is up, and while a later answer shows."""
        size = (self.W, self.H)
        stride = max(1, int(round(0.1 * self.fps)))
        self.cover: dict[int, dict[int, Optional[Rect]]] = {}
        self.fixed: set[int] = set()  # picture answers: kept clear of, but never stepping aside
        for k, p in enumerate(self.prompts):
            L = p.layer
            if L is None or L.job.draw is None:
                continue
            marks = None
            if L.kind == "picture":
                # A change to the whole picture cannot be told apart from the labels drawn on it, so
                # what it draws on top is read from where its own plan says it draws.
                marks = Rect.around(_marks(L.job.plan))
                if marks is None:
                    continue
                self.fixed.add(k)
            lo = int(math.floor(L.start * self.fps))
            hi = self.n - 1 if L.end is None else int(min(self.n - 1, math.ceil(L.end * self.fps)))
            want = set()
            for q in self.prompts[k:]:
                a = int(max(lo, math.floor(q.appear * self.fps)))
                b = int(min(hi, math.ceil(q.leave * self.fps)))
                want.update(range(a, b + 1, stride))
            for q in self.prompts[k + 1:]:
                if q.layer is None:
                    continue
                a = int(max(lo, math.floor(q.layer.start * self.fps)))
                b = hi if q.layer.end is None else int(min(hi, math.ceil(q.layer.end * self.fps)))
                want.update(range(a, b + 1, stride))
            if marks is not None:
                self.cover[k] = {j: marks for j in sorted(want)}
            else:
                self.cover[k] = {j: layers.change_on(L, size, j, self.fps, gated=True).rect for j in sorted(want)}

    def _covered(self, k: int, j: int) -> Optional[Rect]:
        """Answer k's box at frame j, from the nearest frame probed (None: not probed, or nothing)."""
        got = self.cover.get(k)
        if not got:
            return None
        if j in got:
            return got[j]
        keys = list(got)
        i = int(np.searchsorted(keys, j))
        near = [got[keys[x]] for x in (i - 1, i) if 0 <= x < len(keys) and abs(keys[x] - j) <= 0.12 * self.fps]
        return Rect.around(r for r in near if r is not None)

    # -- where the box goes
    def _heads_over(self, t0: float, t1: float) -> Optional[Rect]:
        a = int(max(0, math.floor(t0 * self.fps)))
        b = int(min(self.n - 1, math.ceil(t1 * self.fps)))
        return Rect.around(h for h in self.heads[a:b + 1] if h is not None)

    def _place(self, p: Prompt, plan: Optional[dict]) -> None:
        bar = self.bar
        z = self.zone.rect
        saved = ((plan or {}).get("prompts") or [])
        if plan is not None and p.index < len(saved) and isinstance(saved[p.index].get("box"), dict):
            b = saved[p.index]["box"]
            try:
                p.rect = Rect.of(b["rect"])
                p.where, p.scale = str(b.get("where", "from the plan")), float(b.get("scale", 1.0))
                self._ripple_room(p)
                return
            except (KeyError, TypeError, ValueError):
                pass
        margin = 0.012 * self.H
        gap = 0.022 * self.H
        head = self._heads_over(p.appear, p.leave)
        grow = 1.04  # room for the pulse
        if head is None:
            bands = {"top": (z.y0 + margin, z.y0 + 0.3 * self.H), "bottom": (z.y1 - 0.3 * self.H, z.y1 - margin)}
        else:
            bands = {"top": (z.y0 + margin, head.y0 - gap), "bottom": (head.y1 + gap, z.y1 - margin)}
        pos = self.opts["position"]
        order = ["top", "bottom"] if pos in ("auto", "top") else ["bottom", "top"]
        stride = max(1, int(round(0.1 * self.fps)))
        frames = list(range(int(math.floor(p.appear * self.fps)), int(math.ceil(p.leave * self.fps)) + 1, stride))
        earlier, own = [], []  # earlier answers' graphics while it is up; its own answer once sent
        for k, q in enumerate(self.prompts):
            if k not in self.cover or k > p.index:
                continue
            for j in frames:
                if j / self.fps < q.layer.start:
                    continue
                r = self._covered(k, j)
                if r is not None:
                    (own if k == p.index else earlier).append(r)
        prev = next((q for q in reversed(self.prompts[:p.index]) if q.rect is not None), None)
        # Full size where it fits; a little smaller only to find a spot clear of the earlier graphics,
        # or to fit at all. In order: clear of the earlier graphics (which otherwise step aside while
        # it is up), clear of its own answer (so it can stay while the answer lands; otherwise it
        # dissolves under it), where the box before it sat (a box that keeps its place reads as one
        # composer), then nearest its edge of the frame.
        best, fallback = None, None
        for scale in (1.0, 0.9, 0.8, 0.7):
            w, h = bar.w * scale, bar.h * scale
            gw, gh = w * grow, h * grow
            cands = []
            for rank, side in enumerate(order):
                r0, r1 = bands[side]
                if r1 - r0 < gh:
                    continue
                step = max(2.0, 6 * bar.u)
                ys = np.arange(r0, r1 - gh + 1e-3, step)
                if side == "bottom":
                    ys = ys[::-1]
                for n_off, y in enumerate(ys):
                    rect = Rect(z.cx - gw / 2, y, z.cx + gw / 2, y + gh)
                    hit = any(rect.intersects(o) for o in earlier)
                    mine = any(rect.intersects(o, margin=CLEAR * self.H) for o in own)
                    moved = abs(rect.cy - prev.rect.cy) if prev is not None and prev.where.startswith(side) else math.inf
                    cands.append((hit, rank, mine, moved, n_off, rect, side, scale))
            if not cands:
                continue
            free = [c for c in cands if not c[0]]
            if free:
                c = min(free, key=lambda c: c[1:5])
                best = ((False,), c[5], c[6], c[7])
                break
            if fallback is None:
                c = min(cands, key=lambda c: c[1:5])
                fallback = ((True,), c[5], c[6], c[7])
            if scale < 0.88:
                break
        best = best or fallback
        if best is None:  # no band holds the bar: the margin on the roomier side, smaller; never the face
            above = head.y0 if head is not None else self.H
            below = self.H - head.y1 if head is not None else self.H
            scale = 0.6
            w, h = bar.w * scale, bar.h * scale
            if above >= below:
                y = max(0.015 * self.H, min(z.y0 + margin, (head.y0 if head else self.H) - gap - h))
                side = "top"
            else:
                y = min(0.985 * self.H - h, max(z.y1 - margin - h, (head.y1 if head else 0) + gap))
                side = "bottom"
            rect = Rect(z.cx - w / 2, y, z.cx + w / 2, y + h)
            best = ((True, 0, 0, 0, 0, 0), rect, side, scale)
            p.warnings.append("there was no room for the box inside the safe zone clear of the head; it sits in the "
                              "margin, smaller")
            self.warnings.append(f"prompt {p.index + 1}: the box sits outside the safe zone")
        _, rect, side, scale = best
        # the bar itself sits in the middle of the room it was given (that room allows for the pulse)
        w, h = bar.w * scale, bar.h * scale
        p.rect = Rect(rect.cx - w / 2, rect.cy - h / 2, rect.cx + w / 2, rect.cy + h / 2)
        p.where = side if pos in ("auto", side) else f"{side} (no room {'above the head' if pos == 'top' else 'below the chin'})"
        if pos in ("top", "bottom") and side != pos:
            p.warnings.append(f"there is no room {'above the head' if pos == 'top' else 'below the chin'}; the box "
                              f"went {'below' if pos == 'top' else 'above'}")
        p.scale = scale
        self._ripple_room(p)

    def _ripple_room(self, p: Prompt) -> None:
        """The ripple travels the same way from every bar, and fades out before it reaches the head:
        it is kept off the head's box (taken over the ripple's own frames) and a margin round it."""
        p.reach = RIPPLE_REACH * self.bar.u * p.scale
        head = self._heads_over(p.send, p.send + RIPPLE + 0.12)
        p.keep_off = None if head is None else head.grow(*(6 * self.bar.u,) * 4)

    def _own_landing(self) -> None:
        """A box whose own answer lands where it sits leaves at once, so the answer is seen whole."""
        for k, p in enumerate(self.prompts):
            if k not in self.cover or p.rect is None:
                continue
            a = int(math.floor(p.send * self.fps))
            b = int(math.ceil((p.send + HOLD + EXIT) * self.fps))
            if any(layers.overlap(self._covered(k, j), p.rect, margin=CLEAR * self.H) for j in range(a, b + 1)):
                p.hold, p.exit, p.under = QUICK, EXIT_QUICK, True

    def _step_aside(self) -> None:
        """How much of each graphic answer shows on each frame: an earlier one steps aside while a box
        or a later graphic would land on it."""
        n = self.n
        self.seen: dict[int, np.ndarray] = {}
        ramp = max(1.0, STEP_ASIDE * self.fps)
        for k, p in enumerate(self.prompts):
            if k not in self.cover or k in self.fixed:
                continue
            hidden = np.zeros(n, bool)
            for j in self.cover[k]:
                mine = self._covered(k, j)
                if mine is None:
                    continue
                t = j / self.fps
                for q in self.prompts[k + 1:]:
                    if q.on(t) and q.rect is not None and mine.intersects(q.rect.grow(*(0.012 * self.H,) * 4)):
                        hidden[j] = True
                    if q.layer is not None and (m2 := self.prompts.index(q)) in self.cover and t >= q.layer.start:
                        theirs = self._covered(m2, j)
                        if theirs is not None and mine.intersects(theirs, margin=0.01 * self.H):
                            hidden[j] = True
            if not hidden.any():
                continue
            stride = max(1, int(round(0.1 * self.fps)))
            idx = np.flatnonzero(hidden)
            for j in idx:  # a probed frame stands for the frames round it
                hidden[max(0, j - stride):min(n, j + stride + 1)] = True
            # distance in frames to the nearest hidden frame, both ways: a soft step aside and back
            dist = np.full(n, np.inf)
            last = -np.inf
            for j in range(n):
                if hidden[j]:
                    last = j
                dist[j] = j - last
            last = np.inf
            for j in range(n - 1, -1, -1):
                if hidden[j]:
                    last = j
                dist[j] = min(dist[j], last - j)
            self.seen[k] = np.clip(dist / ramp, 0.0, 1.0).astype(np.float32)

    def _glass_tint(self) -> None:
        """Glass is tinted so the white type stays readable on what is behind it: a light frost on a
        dark picture, smoked glass on a bright one."""
        from ..core import video

        self.tints: dict[int, tuple[tuple, float]] = {}
        if self.opts["style"] != "glass":
            return
        info = self.clip.info
        for p in self.prompts:
            lum = []
            for t in np.linspace(p.appear + ENTER, p.send, 3):
                j = int(min(self.n - 1, max(0, round(t * self.fps))))
                k = self.tl.source_frame(j, self.fps, info.frames)
                f = self.fr.apply(video.read_frame(info, k), k)
                r = p.rect
                y0, y1 = int(max(0, r.y0)), int(min(self.H, r.y1))
                x0, x1 = int(max(0, r.x0)), int(min(self.W, r.x1))
                region = f[y0:y1, x0:x1].astype(np.float32) / 255.0
                lin = np.where(region <= 0.04045, region / 12.92, ((region + 0.055) / 1.055) ** 2.4)
                lum.append(float((lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()))
            L = float(np.mean(lum)) if lum else 0.3
            # White type wants at least 4:1 against the glass's average (its shadow carries the
            # brightest spots of what shows through); the glass is always at least lightly smoked.
            target = 1.05 / 4.0 - 0.05
            dark = 0.004
            a = (L - target) / max(1e-3, L - dark) if L > target else 0.0
            self.tints[p.index] = ((14, 14, 18), min(0.8, max(GLASS_SMOKE, a)))

    # -- drawing
    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        ps = self.prompts
        beat = self.opts["beat"]
        ripples = beat in ("auto", "ripple", "both")
        for p in ps:
            if p.layer is not None and p.layer.kind == "picture":
                p.layer.draw(frame, j, t)
                if p.under:  # a picture's own labels cannot go over its box: the box sits on them, leaving at once
                    self._draw_under(frame, p, t, ripples)
        for kind in ("behind", "over"):
            for k, p in enumerate(ps):
                if p.layer is None or p.layer.kind != kind:
                    continue
                if p.under:  # its answer lands where it sits: the box goes under it as it dissolves
                    self._draw_under(frame, p, t, ripples)
                seen = float(self.seen[k][j]) if k in self.seen and j < len(self.seen[k]) else 1.0
                p.layer.draw(frame, j, t, seen)
        for p in ps:
            flash = beat in ("flash", "both") or (beat == "auto" and p.layer is not None and p.layer.kind == "picture")
            if flash and p.send <= t < p.send + FLASH:
                a = 0.2 * math.exp(-(t - p.send) / 0.11)
                frame[:] = np.clip(frame + (255.0 - frame) * a + 0.5, 0, 255).astype(np.uint8)
        for p in ps:
            if p.on(t) and not p.under:
                self._draw_box(frame, p, t)
        for p in ps:
            if ripples and not p.under and p.send <= t < p.send + RIPPLE + 0.12:
                self._draw_ripple(frame, p, t)

    def _draw_under(self, frame: np.ndarray, p: Prompt, t: float, ripples: bool) -> None:
        if p.on(t):
            self._draw_box(frame, p, t)
        if ripples and p.send <= t < p.send + RIPPLE + 0.12:
            self._draw_ripple(frame, p, t)

    def _state(self, p: Prompt, t: float) -> tuple[float, float, float]:
        """The box's opacity, scale and slide at `t`."""
        u = self.bar.u
        toward = -1.0 if p.where.startswith("top") else 1.0  # its own edge of the frame
        q_in = _ease_out((t - p.appear) / ENTER)
        out_at = p.send + p.hold
        q_out = _smooth((t - out_at) / p.exit) if t >= out_at else 0.0
        alpha = q_in * (1.0 - q_out)
        scale = (0.965 + 0.035 * q_in) * (1.0 - 0.035 * q_out)
        if p.send <= t < p.send + PULSE:
            scale *= 1.0 + 0.03 * math.sin(math.pi * (t - p.send) / PULSE)
        slide = toward * (16 * u * (1.0 - q_in) + 8 * u * q_out)
        return alpha, scale, slide

    def _canvas(self, p: Prompt, t: float) -> tuple[Sprite, Optional[np.ndarray]]:
        """The bar as it looks at `t`, at its full size, on its own canvas."""
        bar = self.bar
        rgb = bar.base[0].copy()
        a = bar.base[1].copy()
        layer = (rgb, a)
        typed = sum(1 for x in p.keys if x <= t)
        sent = t >= p.send + 0.06
        bx = bar.text_x
        by = bar.pad + bar.baseline
        if typed == 0:
            ph = bar.placeholder
            onto_layer(layer, ph.sprite, bx - ph.ox, by - ph.oy, bar.look.hint_a)
            caret_x = bx
        else:
            cut = p.cuts[typed - 1]
            keep = int(min(p.word.w, math.ceil(cut + 1.5)))
            spr = Sprite(p.word.rgb[:, :keep], p.word.a[:, :keep]) if keep < p.word.w else p.word
            onto_layer(layer, spr, bx - p.word_ox, by - p.word_oy, 0.72 if sent else 1.0)
            caret_x = bx + (cut - p.word_ox) + 2.5 * bar.u
        # the caret: steady while letters come, blinking while it waits; gone once sent
        if not sent:
            last = p.keys[typed - 1] if typed else p.appear + ENTER
            idle = t - last
            if typed < len(p.keys) and typed > 0 and idle < 0.5:
                show = True
            else:
                show = (idle % 1.0) < 0.55
            if show:
                ch = bar.cap * 1.22
                cw = max(2.0, 3.0 * bar.u)
                caret = Sprite(np.full((int(round(ch)), int(round(cw)), 3), bar.look.ink, np.float32),
                               np.ones((int(round(ch)), int(round(cw))), np.float32))
                onto_layer(layer, caret, caret_x, by - bar.cap * 1.1)
        # the send button: grey until there is text, lit once there is, pressed, then a tick
        if sent:
            btn = bar.btn_done
        elif typed > 0:
            btn = bar.btn_on
        else:
            btn = bar.btn_off
        press = 1.0
        if p.send - PRESS <= t < p.send + 0.1:
            q = (t - (p.send - PRESS)) / (PRESS + 0.1)
            press = 1.0 - 0.14 * math.sin(math.pi * q)
        bxx, byy = bar.btn_at
        if abs(press - 1.0) > 1e-3:
            sb = btn.scaled(press)
            onto_layer(layer, sb, bxx + (btn.w - sb.w) / 2, byy + (btn.h - sb.h) / 2)
        else:
            onto_layer(layer, btn, bxx, byy)
        # the pulse: a glow round the bar as it is sent
        if p.send <= t < p.send + GLOW:
            onto_layer(layer, bar.glow, 0, 0, 0.6 * (1.0 - _ease_out((t - p.send) / GLOW)))
        return Sprite(rgb=rgb, a=a), (bar.fill_mask if bar.look.fill is None else None)

    def _draw_box(self, frame: np.ndarray, p: Prompt, t: float) -> None:
        import cv2

        alpha, scale, slide = self._state(p, t)
        if alpha <= 0.002:
            return
        sprite, glass = self._canvas(p, t)
        s = scale * p.scale
        if abs(s - 1.0) > 1e-3:
            sprite = sprite.scaled(s)
            if glass is not None:
                glass = cv2.resize(glass, (sprite.w, sprite.h), interpolation=cv2.INTER_LINEAR)
        cx, cy = p.rect.cx, p.rect.cy + slide
        X, Y = cx - sprite.w / 2, cy - sprite.h / 2
        if glass is not None:
            self._frost(frame, glass, X, Y, alpha, self.tints.get(p.index, ((14, 14, 18), 0.5)))
        over(frame, sprite, X, Y, alpha)

    def _frost(self, frame: np.ndarray, mask: np.ndarray, X: float, Y: float, alpha: float, tint) -> None:
        """Frosted glass: the picture under the bar blurred, then tinted."""
        import cv2

        H, W = frame.shape[:2]
        x0, y0 = int(round(X)), int(round(Y))
        mh, mw = mask.shape
        sigma = 18 * self.bar.u
        m = int(math.ceil(3 * sigma))
        ax0, ay0, ax1, ay1 = max(0, x0 - m), max(0, y0 - m), min(W, x0 + mw + m), min(H, y0 + mh + m)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        region = frame[ay0:ay1, ax0:ax1]
        k = 4
        small = cv2.resize(region, (max(1, (ax1 - ax0) // k), max(1, (ay1 - ay0) // k)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / k)
        blur = cv2.resize(small, (ax1 - ax0, ay1 - ay0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        rgb, ta = tint
        frosted = blur * (1.0 - ta) + np.asarray(rgb, np.float32) * ta
        # the mask, where it falls on the frame
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + mw), min(H, y0 + mh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        mm = mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None] * alpha
        part = frame[cy0:cy1, cx0:cx1].astype(np.float32)
        fr = frosted[cy0 - ay0:cy1 - ay0, cx0 - ax0:cx1 - ax0]
        frame[cy0:cy1, cx0:cx1] = np.clip(part * (1.0 - mm) + fr * mm + 0.5, 0, 255).astype(np.uint8)

    def _field(self, scale: float) -> tuple[np.ndarray, float]:
        """The distance to the bar's outline (at `scale`) over a canvas wide enough for the ripple."""
        key = round(scale, 3)
        if key not in self._fields:
            bar = self.bar
            pad = float(math.ceil((RIPPLE_REACH + 24) * bar.u * scale))
            self._fields[key] = (_rrect_sdf(bar.w * scale, bar.h * scale, bar.r * scale, pad), pad)
        return self._fields[key]

    def _ring(self, p: Prompt, t: float) -> Optional[np.ndarray]:
        """The ripple's opacity at `t` over the bar's ripple canvas (None when there is none): two
        rings leave the bar's outline as it is sent, each a crisp line in a soft glow, fading as
        they go and kept off the head."""
        d, off = self._field(p.scale)
        u = self.bar.u * p.scale
        out = None
        for delay, strength in ((0.0, 0.95), (0.11, 0.55)):
            q = (t - p.send - delay) / RIPPLE
            if not 0.0 <= q < 1.0:
                continue
            e = 3 * u + (p.reach - 3 * u) * _ease_out(q)
            fade = strength * (1.0 - q) ** 1.5
            core = np.clip(1.0 - np.abs(d - e) / ((4.5 - 2.5 * q) * u), 0.0, 1.0)
            halo = np.exp(-((d - e) / ((9.0 - 3.0 * q) * u)) ** 2) * 0.45
            ring = np.maximum(core, halo) * fade
            out = ring if out is None else np.maximum(out, ring)
        if out is None:
            return None
        return out * self._clear_of_head(p, d.shape, off)

    def _clear_of_head(self, p: Prompt, shape: tuple, off: float) -> np.ndarray:
        """1 away from the head, easing to 0 at the edge of the box kept off it (cached per prompt)."""
        key = ("clear", p.index)
        if key not in self._fields:
            if p.keep_off is None:
                self._fields[key] = np.ones(shape, np.float32)
            else:
                H, W = shape
                ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
                x = xs + 0.5 + p.rect.x0 - off
                y = ys + 0.5 + p.rect.y0 - off
                k = p.keep_off
                dx = np.maximum(np.maximum(k.x0 - x, x - k.x1), 0)
                dy = np.maximum(np.maximum(k.y0 - y, y - k.y1), 0)
                dist = np.hypot(dx, dy)
                self._fields[key] = np.clip(dist / (20 * self.bar.u), 0.0, 1.0).astype(np.float32)
        return self._fields[key]

    def _draw_ripple(self, frame: np.ndarray, p: Prompt, t: float) -> None:
        ring = self._ring(p, t)
        if ring is None:
            return
        _, off = self._field(p.scale)
        over(frame, _paint(ring, self.accent), p.rect.x0 - off, p.rect.y0 - off)

    def _ring_over_head(self, p: Prompt, t: float, head: Optional[Rect]) -> bool:
        """Whether the ripple drawn at `t` puts any visible light inside `head`."""
        ring = self._ring(p, t) if head is not None else None
        if ring is None:
            return False
        _, off = self._field(p.scale)
        ox, oy = p.rect.x0 - off, p.rect.y0 - off
        x0, y0 = int(max(0, math.floor(head.x0 - ox))), int(max(0, math.floor(head.y0 - oy)))
        x1, y1 = int(min(ring.shape[1], math.ceil(head.x1 - ox))), int(min(ring.shape[0], math.ceil(head.y1 - oy)))
        return x1 > x0 and y1 > y0 and float(ring[y0:y1, x0:x1].max()) > 0.01

    # -- sound
    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        out = sound
        for p in self.prompts:
            if p.layer is not None and p.layer.job.sound is not None:
                out = np.asarray(p.layer.job.sound(out, rate), np.float32)
        if not self.opts["sound"]:
            return out
        out = out.copy()
        voice = self.voice_rms(sound, rate)
        rng = np.random.default_rng(11)
        for p in self.prompts:
            for i, kt in enumerate(p.keys):
                s = _tick(rate, rng, space=p.text[i] == " ") * voice * 10 ** (KEY_DB / 20)
                _add(out, s, kt, rate)
            _add(out, _send(rate, rng) * voice * 10 ** (SEND_DB / 20), p.send - 0.01, rate)
        return np.clip(out, -1.0, 1.0)

    def voice_rms(self, sound: np.ndarray, rate: int) -> float:
        """How loud the voice is while words are said (RMS of the clip's own sound)."""
        if getattr(self, "_voice", None) is not None:
            return self._voice
        mono = sound.mean(axis=1) if sound.ndim == 2 else sound
        parts = [mono[int(w.start * rate):int(w.end * rate)] for w in self.words if w.end > w.start]
        parts = [x for x in parts if len(x)]
        if parts:
            v = float(np.sqrt(np.mean(np.concatenate(parts) ** 2)))
        else:
            v = 0.0
        self._voice = v if v > 1e-4 else 10 ** (-24 / 20)
        return self._voice

    # -- checks
    def report(self) -> dict:
        over_face = outside = 0
        for j in range(self.n):
            t = j / self.fps
            h = self.heads[j]
            for p in self.prompts:
                if not p.on(t):
                    continue
                alpha, scale, slide = self._state(p, t)
                if alpha <= 0.02:
                    continue
                s = scale
                r = p.rect
                w, hh = r.w * s, r.h * s
                drawn = Rect(r.cx - w / 2, r.cy + slide - hh / 2, r.cx + w / 2, r.cy + slide + hh / 2)
                ripple = (self.opts["beat"] in ("auto", "ripple", "both") and p.send <= t < p.send + RIPPLE + 0.12
                          and self._ring_over_head(p, t, h))
                if h is not None and (drawn.intersects(h) or ripple):
                    over_face += 1
                if not self.zone.contains(Rect(r.cx - w / 2, r.cy + slide - hh / 2, r.cx + w / 2, r.cy + slide + hh / 2)):
                    outside += 1
        lines = []
        for p in self.prompts:
            L = p.layer
            if p.first is None:
                land = "nothing seen"
            elif abs(p.first - p.send) <= 1.5 / self.fps:
                land = "on the send frame"
            elif 0 < p.first - p.send <= 0.05:
                land = f"{(p.first - p.send) * 1000:.0f} ms after the send (the answer's own ease in)"
            else:
                land = f"{(p.first - p.send) * self.fps:+.0f} frames from the send"
            lines.append(f"“{p.text}” sent at {p.send:.2f} s ({p.how}) → {L.describe() if L else '?'}; "
                         f"first change {land}; box {p.where}" + (f" at {p.scale:.0%}" if p.scale < 0.999 else "")
                         + (", dissolves under its answer (which lands where it sits)" if p.under else ""))
        stepped = [f"answer {k + 1} for {float((v < 0.999).mean()) * self.seconds:.2f} s"
                   for k, v in self.seen.items() if (v < 0.999).any()]
        out: dict = {f"prompt {i + 1}": line for i, line in enumerate(lines)}
        out.update({"frames_over_face": over_face, "frames_outside_safe_zone": outside,
                    "placed": ", ".join(p.where for p in self.prompts),
                    "stepped_aside": "; ".join(stepped) or "none"})
        for p in self.prompts:
            if p.layer is None:
                continue
            c = p.layer.job.checks
            keep = [f"{k} {v}" for k, v in c.items() if k in ("frames_over_face", "frames_outside_safe_zone", "placed",
                                                              "look", "skin", "lands", "on_screen", "span")
                    and v not in ([], "", None)]
            out[f"answer {p.index + 1}"] = (f"{p.spec.name}: " + "; ".join(keep)) if keep else p.spec.name
            for w in c.get("warnings") or []:
                p.warnings.append(f"{p.spec.name} says: {w}")
        out["warnings"] = list(self.warnings) + [f"prompt {p.index + 1}: {w}" for p in self.prompts for w in p.warnings]
        return out

    def sound_line(self, sound: np.ndarray, rate: int) -> str:
        """The level of the sounds laid in, measured: the mix less the clip's own sound, each key
        tick and the send against the voice (50 ms windows while words are said)."""
        added = (self.mix(sound, rate) - sound).mean(axis=1)
        mono = sound.mean(axis=1)
        win = int(0.05 * rate)

        def rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0

        voice = [rms(mono[i:i + win]) for w in self.words
                 for i in range(int(w.start * rate), int(w.end * rate) - win, win)]
        if not voice:
            return "nothing is said, so the sounds are at their own quiet level"
        v = float(np.median(voice))
        ticks = [rms(added[int(t * rate):int(t * rate) + win]) for p in self.prompts for t in p.keys]
        sends = [rms(added[int((p.send - 0.01) * rate):int((p.send + 0.15) * rate)]) for p in self.prompts]

        def under(x: float) -> float:
            return 20 * math.log10(max(v, 1e-9) / max(x, 1e-9))

        return (f"key ticks {under(float(np.median(ticks))):.0f} dB and the send {under(float(np.median(sends))):.0f} dB "
                f"under the voice (measured on the sound laid in, alone)")

    # -- the plan
    def plan(self) -> list[dict]:
        out = []
        for p in self.prompts:
            L = p.layer
            out.append({
                "text": p.text, "answer": str(p.spec), "chosen": p.chosen, "send": round(p.send, 3), "how": p.how,
                "until": None if p.end is None else round(p.end, 3),
                "appear": round(p.appear, 3), "leave": round(p.leave, 3), "hold": round(p.hold, 3),
                "keys": [round(x, 3) for x in p.keys],
                "box": {"rect": p.rect.as_list(), "where": p.where, "scale": round(p.scale, 3)},
                "first_change": None if p.first is None else round(p.first, 3),
                "layer": None if L is None else {
                    "effect": L.name, "kind": L.kind, "moment": round(p.send, 3), "told": L.told,
                    "options": {k: v for k, v in L.opts.items()},
                    "window": [round(L.start, 3), None if L.end is None else round(L.end, 3)],
                    "plan": L.job.plan},
            })
        return out

    def _showing(self, p: Prompt, t0: float, t1: float) -> float:
        """The first moment from t0 to t1 when p's answer is on screen (t0 when it cannot be told)."""
        L = p.layer
        if L is None or L.kind == "picture":
            return t0
        for t in np.arange(t0, t1, 0.1):
            j = int(min(self.n - 1, round(t * self.fps)))
            if layers.change_on(L, (self.W, self.H), j, self.fps, gated=True).share > 0.0005:
                return float(t)
        return t0

    def sheet_times(self) -> list[float]:
        """Six moments that tell it: the box typing, the send, and each answer on screen once the box
        has gone."""
        ps = self.prompts
        e = self.seconds - 0.5 / self.fps
        if len(ps) == 1:
            p = ps[0]
            mid = p.keys[len(p.keys) // 2] + 0.02
            gone = p.leave + 0.1
            picks = [p.appear + ENTER + 0.05, mid, p.send - 0.04, p.send + 0.12, self._showing(p, gone, gone + 1.5),
                     self._showing(p, min(e, gone + 1.4), e)]
        elif len(ps) == 2:
            picks = []
            for p in ps:
                picks += [p.keys[len(p.keys) // 2] + 0.02, p.send + 0.12, self._showing(p, p.leave + 0.1, p.leave + 1.5)]
        else:
            picks = []
            for p in ps[:3]:
                picks += [p.keys[len(p.keys) * 2 // 3] + 0.02, self._showing(p, p.send + 0.4, p.send + 1.6)]
        return sorted(round(min(e, max(0.0, x)), 3) for x in picks)[:6]

    def still_times(self) -> list[float]:
        ps = self.prompts
        e = self.seconds - 0.5 / self.fps
        p0, pn = ps[0], ps[-1]
        picks = [p0.keys[len(p0.keys) // 2] + 0.02, p0.send + 0.15]
        if len(ps) > 1:
            picks += [pn.send + 0.15]
        picks += [min(e, pn.send + 1.0)]
        return [round(min(e, max(0.0, x)), 3) for x in picks]


# ---- the sounds, made here ---------------------------------------------------------------------

KEY_DB = -23.0  # each key tick's level against the voice (its RMS against the voice's)
SEND_DB = -20.0


def _unit(s: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(s ** 2)))
    return (s / r).astype(np.float32) if r > 0 else s.astype(np.float32)


def _tick(rate: int, rng: np.random.Generator, space: bool = False) -> np.ndarray:
    """A soft key: a short bright click with a little body; the space bar lower and duller."""
    n = int((0.02 if space else 0.014) * rate)
    t = np.arange(n) / rate
    noise = rng.standard_normal(n)
    click = np.diff(np.concatenate([[0.0], noise])) * np.exp(-t / (0.0022 if space else 0.0016))
    tone = np.sin(2 * np.pi * rng.uniform(1700, 2400) * (0.6 if space else 1.0) * t) * np.exp(-t / 0.0022)
    body = np.sin(2 * np.pi * rng.uniform(170, 230) * t) * np.exp(-t / 0.005)
    s = click * 0.8 + 0.3 * tone + (0.5 if space else 0.25) * body
    s *= np.minimum(1.0, t / 0.0007)
    return _unit(s) * float(rng.uniform(0.75, 1.0))


def _send(rate: int, rng: np.random.Generator) -> np.ndarray:
    """The send: a soft rising tick with a breath of air."""
    n = int(0.16 * rate)
    t = np.arange(n) / rate
    f = 620 + 420 * np.minimum(1.0, t / 0.05)
    tone = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t / 0.04) * np.minimum(1.0, t / 0.003)
    air = rng.standard_normal(n)
    air = np.convolve(air, np.ones(24) / 24, mode="same") * np.exp(-((t - 0.035) / 0.03) ** 2) * 0.6
    return _unit(tone + air)


def _add(out: np.ndarray, s: np.ndarray, at: float, rate: int) -> None:
    i = int(round(at * rate))
    if i < 0 or i >= len(out):
        return
    m = min(len(s), len(out) - i)
    out[i:i + m] += s[:m, None]


# ---- the effect ----------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, None, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    try:
        box = PromptBox(clip, opts, fr, tl, plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    checks = box.report()
    if opts["sound"]:
        checks["sound"] = box.sound_line(clip.sound, 48000)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "style": opts["style"], "prompts": box.plan(),
                "sheet_times": box.sheet_times()}

    def after(result: dict, out) -> dict:
        extra = {}
        for p in box.prompts:
            if p.layer is not None and p.layer.job.after is not None:
                got = p.layer.job.after(result, out) or {}
                extra.update({f"answer {p.index + 1} {k}": v for k, v in got.items()})
        return extra

    return Job(NAME, clip, tl, fr, box.draw, out_plan, checks=checks, still_times=box.still_times, after=after,
               sound=box.mix)
