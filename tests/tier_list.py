"""The tier list's reading of the words, checked on a transcript written for the test (no clip needed).

    .venv/bin/python tests/tier_list.py

`tests/fixtures/tier-list-words.json` ranks ten apps in the ways people say it ("Notion is S tier",
"Figma goes in A", "I'd put Slack in B tier", "Google Docs, C tier", "S tier: Cursor and ChatGPT",
"Zoom would be a solid D", "I'd give Canva a B", "Excel's D tier", "Linear is S-tier", one ranked
twice), among sentences that must never be taken for a ranking ("this is a tier list", "I made a
video", "see you later", "that's an S tier move", "put it in a drawer"). Each ranking must be found
in the right row, and land on the word that makes it known; nothing else may be found. It also
checks rows with names of their own, items given without times, and the short form of the items.
Exits 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core.words import Word  # noqa: E402
from fx.effects import tier_list as T  # noqa: E402

FIXTURE = PACK / "tests/fixtures/tier-list-words.json"

# (card, row, the word it lands on)
EXPECTED = [
    ("Notion", "S", "S"),
    ("Figma", "A", "A."),
    ("Slack", "B", "B"),
    ("Google Docs", "C", "C"),
    ("Cursor", "S", "Cursor"),
    ("ChatGPT", "S", "ChatGPT."),
    ("Zoom", "D", "D."),
    ("Canva", "B", "B."),
    ("Excel", "D", "D"),
    ("Trello", "B", "B"),  # ranked twice: the last one stands
    ("Linear", "S", "S-tier."),
]


def words_of(items: list[dict]) -> list[Word]:
    return [Word(i["w"], float(i["s"]), float(i["e"])) for i in items]


def main() -> int:
    data = json.loads(FIXTURE.read_text())
    words = words_of(data["words"])
    failures = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    tiers = T.parse_tiers(T.DEFAULT_TIERS, "solid")
    found = T.find_rankings(words, tiers)
    got = [(f.text, tiers[f.tier].label, words[f.tier_word].text if f.at == words[f.tier_word].start else
            words[f.words[0]].text) for f in found]
    print("found:", "; ".join(f"{f.text} -> {tiers[f.tier].label} at {f.at:.2f} s ({f.phrase!r})" for f in found))
    check(sorted(got) == sorted(EXPECTED), f"the {len(EXPECTED)} rankings, each in its row and on its word"
          + ("" if sorted(got) == sorted(EXPECTED) else f": missing {sorted(set(EXPECTED) - set(got))}, "
             f"extra {sorted(set(got) - set(EXPECTED))}"))
    check([f.at for f in found] == sorted(f.at for f in found), "in the order they are said")
    for f in found:
        start = words[f.words[0]].start
        check(f.at >= start and f.at >= words[f.tier_word].start,
              f"{f.text} lands once both it and its row are said ({f.at:.2f} s)")

    # The README's own examples, names in lower case as a transcriber writes them mid-sentence, among
    # three that must not count ("Anyway, B tier", "A tier is solid", "S tier is basically everything").
    coffee = T.find_rankings(words_of(data["coffee"]["words"]), tiers)
    got = sorted((f.text, tiers[f.tier].label) for f in coffee)
    want = sorted([("Espresso", "S"), ("Flat White", "A"), ("Latte", "B"), ("Iced Coffee", "C"), ("Decaf", "D"),
                   ("Cortado", "S"), ("Instant coffee", "D")])
    check(got == want, f"the README's examples: {got}" + ("" if got == want else f", wanted {want}"))

    # Rows with names of their own.
    custom = data["custom_rows"]
    rows = T.parse_tiers(custom["tiers"], "solid")
    got = [(f.text, rows[f.tier].label) for f in T.find_rankings(words_of(custom["words"]), rows)]
    check(sorted(got) == sorted([("Notion", "GOAT"), ("Slack", "Mid"), ("Zoom", "Skip")]),
          f"rows named GOAT, Mid and Skip: {got}")

    # Items given without times: a ranking heard lands on its row, a name heard lands on the name, a
    # name split by the transcriber is still found, and one never said is placed between the others.
    items, rows = T.parse_items("S: Notion, Chat GPT; A: Figma; B: Google Docs; D: Fax machine", tiers, False,
                                PACK, "solid")
    cards, notes = T.time_items(items, words, found, words[-1].end)
    how = {c.text: (c.how, c.word) for c in cards}
    check(how["Notion"] == ("ranked", "S"), f"a named item lands on its ranking: {how['Notion']}")
    check(how["Chat GPT"][0] == "ranked", f"'Chat GPT' is found as 'ChatGPT': {how['Chat GPT']}")
    check(how["Google Docs"] == ("ranked", "C"), f"a two-word name: {how['Google Docs']}")
    check(how["Fax machine"][0] == "placed" and any("Fax machine" in n for n in notes),
          f"a name never said is placed and the run says so: {how['Fax machine']}")
    check([c.at for c in cards] == sorted(c.at for c in cards), "the cards come in time order")

    # A time given lands on the word that starts nearest it.
    items, _ = T.parse_items("A: Figma@2.45", tiers, False, PACK, "solid")
    cards, _ = T.time_items(items, words, [], words[-1].end)
    nearest = min(words, key=lambda w: abs(w.start - 2.45))
    check(abs(cards[0].at - nearest.start) < 1e-9, f"a given time moves onto the nearest word: {cards[0].at:.2f} s "
          f"('{cards[0].word}')")

    # Rows named by the items themselves, when no rows were asked for.
    items, rows = T.parse_items('{"Must have": ["Notion"], "Nice": ["Figma"], "Skip": ["Zoom"]}', tiers, False,
                                PACK, "solid")
    check([t.label for t in rows] == ["Must have", "Nice", "Skip"], f"rows from the items: {[t.label for t in rows]}")
    try:
        T.parse_items("Z: Notion", tiers, True, PACK, "solid")
        check(False, "a row the board does not have is refused")
    except T.TierError as e:
        check("does not have" in str(e), f"a row the board does not have is refused: {e}")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
