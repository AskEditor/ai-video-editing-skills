"""The kinds of option an effect takes, each able to read what a person typed and say plainly
what is wrong with it.

An effect lists its options as `{name: kind}`; the defaults live in `effects.json`, the one
manifest, and `run.py` checks both agree before anything renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..core.text import colour


class OptionError(ValueError):
    """An option value that cannot be used; the message says what would work."""


@dataclass(frozen=True)
class Choice:
    values: Sequence[str]
    help: str

    def parse(self, raw: Any) -> str:
        v = str(raw).strip().lower()
        if v not in self.values:
            raise OptionError(f"choose one of {', '.join(self.values)} (got {raw!r})")
        return v


@dataclass(frozen=True)
class Number:
    lo: float
    hi: float
    help: str

    def parse(self, raw: Any) -> float:
        try:
            v = float(str(raw).strip().rstrip("s"))
        except ValueError:
            raise OptionError(f"give a number between {self.lo:g} and {self.hi:g} (got {raw!r})") from None
        if not self.lo <= v <= self.hi:
            raise OptionError(f"give a number between {self.lo:g} and {self.hi:g} (got {raw!r})")
        return v


@dataclass(frozen=True)
class Whole:
    lo: int
    hi: int
    help: str

    def parse(self, raw: Any) -> int:
        try:
            v = int(str(raw).strip())
        except ValueError:
            raise OptionError(f"give a whole number from {self.lo} to {self.hi} (got {raw!r})") from None
        if not self.lo <= v <= self.hi:
            raise OptionError(f"give a whole number from {self.lo} to {self.hi} (got {raw!r})")
        return v


@dataclass(frozen=True)
class Switch:
    help: str

    def parse(self, raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        v = str(raw).strip().lower()
        if v in ("on", "yes", "true", "1"):
            return True
        if v in ("off", "no", "false", "0"):
            return False
        raise OptionError(f"say on or off (got {raw!r})")


@dataclass(frozen=True)
class Colour:
    help: str

    def parse(self, raw: Any) -> str:
        v = str(raw).strip()
        if v.lower() in ("auto", ""):
            return "auto"
        try:
            r, g, b = colour(v)
        except ValueError as e:
            raise OptionError(str(e)) from None
        return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(frozen=True)
class Text:
    help: str

    def parse(self, raw: Any) -> str:
        return str(raw)


@dataclass(frozen=True)
class Size:
    help: str

    def parse(self, raw: Any) -> str:
        v = str(raw).strip().lower()
        if v == "auto":
            return v
        try:
            w, h = (int(x) for x in v.split("x"))
        except ValueError:
            raise OptionError(f"give auto or a size like 1080x1920 (got {raw!r})") from None
        if not (64 <= w <= 4096 and 64 <= h <= 4096) or w % 2 or h % 2:
            raise OptionError(f"give even sides between 64 and 4096, like 1080x1920 (got {raw!r})")
        return v


def size_of(value: str):
    return None if value == "auto" else tuple(int(x) for x in value.split("x"))


def parse_all(spec: dict, defaults: dict, given: dict) -> dict:
    """The options to render with: the defaults, overridden by what was given, every value checked."""
    unknown = [k for k in given if k not in spec]
    if unknown:
        raise OptionError(f"no option called {', '.join(unknown)}; this effect takes {', '.join(spec)}")
    out = {}
    for name, kind in spec.items():
        raw = given.get(name, defaults[name])
        try:
            out[name] = kind.parse(raw)
        except OptionError as e:
            raise OptionError(f"{name}: {e}") from None
    return out
