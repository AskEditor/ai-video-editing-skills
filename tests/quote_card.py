"""Checks for quote-card's finder: a line said as a quote is found from where it starts to the end of
its sentence, and talk that only mentions saying something is not taken for one.

    .venv/bin/python tests/quote_card.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fx.core.moments import TakenBack  # noqa: E402
from fx.core.words import Word  # noqa: E402
from fx.effects.quote_card import said_quotes  # noqa: E402


def words(text: str) -> list[Word]:
    return [Word(w, 0.3 * i, 0.3 * i + 0.25) for i, w in enumerate(text.split())]


CASES = [
    ("My client told me that this edit saved her whole week. Then we moved on.", "This edit saved her whole week."),
    ("And the post says, nobody watches past the first three seconds. Which is wild.",
     "Nobody watches past the first three seconds."),
    ("She was like, you have to post every single day. I disagree.", "You have to post every single day."),
    ("The headline reads, creators are the new studios. Big claim.", "Creators are the new studios."),
    ("And I quote, the best camera is the one you have. End of story.", "The best camera is the one you have."),
    ("So I went to the shop and bought a light.", None),
    ("I said yes.", None),  # under three words: a reply, not a quote
]


def main() -> int:
    bad = 0
    for text, want in CASES:
        got = said_quotes(words(text), TakenBack())
        line = " ".join(got[0].text) if got else None
        ok = line == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {text[:60]!r} -> {line!r}")
    print("all passed" if not bad else f"{bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
