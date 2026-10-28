"""Build the pack's shared files from each effect's own folder.

    python3 tools/build_manifest.py           # write effects.json, README.md and SKILL.md's table
    python3 tools/build_manifest.py --check   # change nothing; fail if any of them is out of date

Every effect keeps its manifest entry in `effects/<id>/effect.json`. From those entries this writes:

- `effects.json`: every entry, sorted by category and then id, each with its `prompt` (the one paste
  the pages offer, made from the entry's `do` and the repository's address in `fx/meta.py`);
- `README.md`: the whole page (the gallery). It is never edited by hand;
- the effects table in `SKILL.md`, between `<!-- effects:start -->` and `<!-- effects:end -->`.
  Everything outside the markers is written by hand and kept as it is;
- the first line of each `effects/<id>/README.md`, the link to the effect's page, but only while
  `fx/meta.py`'s `PAGES_LIVE` is on (and the root README's link to the directory with it). While it is
  off, the line is left out, so nothing points at a page that does not answer yet.

Running it twice changes nothing the second time. It needs only Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
from fx import meta  # noqa: E402  (the repository's address and the pages' switch, kept in one place)

REPO = meta.REPO_URL
START, END = "<!-- effects:start -->", "<!-- effects:end -->"
REQUIRED = {"name": str, "title": str, "line": str, "card": str, "do": str, "category": str, "needs": list, "options": dict,
            "type_this": str, "askeditor": dict}
CARD_MAX = 40  # the card line on askeditor.com/skills: one line under the title, never longer


class Bad(ValueError):
    pass


def entries() -> list[dict]:
    out = []
    for path in sorted((PACK / "effects").glob("*/effect.json")):
        try:
            e = json.loads(path.read_text())
        except ValueError as err:
            raise Bad(f"{path.relative_to(PACK)} is not valid JSON: {err}") from None
        for key, kind in REQUIRED.items():
            if not isinstance(e.get(key), kind):
                raise Bad(f"{path.relative_to(PACK)} needs {key!r} ({kind.__name__})")
        if e["name"] != path.parent.name:
            raise Bad(f"{path.relative_to(PACK)} says its name is {e['name']!r}; it must match its folder")
        card = e["card"].strip()
        if not card or len(e["card"]) > CARD_MAX:
            raise Bad(f"{path.relative_to(PACK)}: card is {len(e['card'])} characters; it must be 1 to {CARD_MAX}")
        if e["title"].lower() in card.lower():
            raise Bad(f"{path.relative_to(PACK)}: card repeats the title {e['title']!r}; say what it does instead")
        do = e["do"].strip()
        if not do or do != e["do"] or do.endswith((":", ".")) or "my video" in do.lower():
            raise Bad(f"{path.relative_to(PACK)}: do is the request that goes before \"my video:\" "
                      f"(\"clean cut\"), with no ending of its own")
        # `pages: false` keeps an effect off askeditor.com/skills (a partial, one on hold, one not yet
        # reviewed); missing means it is ready to be listed.
        if "pages" in e and not isinstance(e["pages"], bool):
            raise Bad(f"{path.relative_to(PACK)}: pages is true or false (false keeps it off askeditor.com/skills)")
        e["prompt"] = meta.prompt(do)  # in effects.json, never in effect.json: it is made from do and REPO
        if not all(isinstance(e["askeditor"].get(k), str) for k in ("what", "why")):
            raise Bad(f"{path.relative_to(PACK)} needs askeditor.what and askeditor.why")
        out.append(e)
    names = [e["name"] for e in out]
    if len(set(names)) != len(names):
        raise Bad("two effects share a name")
    return sorted(out, key=lambda e: (e["category"], e["name"]))


def manifest_text(es: list[dict]) -> str:
    return json.dumps(es, indent=2, ensure_ascii=False) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


GROUPS = [  # the order the README shows the kinds of effect in, and what each is called there
    ("Cut and clean up", ("talking", "cutting", "repurposing")),
    ("Words on screen", ("text",)),
    ("Behind you", ("behind-you",)),
    ("Graphics over the picture", ("graphics",)),
    ("Camera and layout", ("camera", "layout")),
    ("Colour and sound", ("colour", "sound")),
    ("Checks and publishing", ("checks", "publishing")),
]


def _link(address: str) -> str:
    return f"[{address}](https://{address})"


def pages_line(es: list[dict], live: bool) -> str:
    """The root README's line to the directory, under the opening paragraph; empty while the pages are off."""
    return f"All of them, with their films: {_link(meta.PAGES)}\n\n" if live else ""


def effect_readme_text(name: str, current: str, live: bool) -> str:
    """The effect's own README with its first line the link to its page while the pages are live, and
    without it while they are off. The rest of the file is written by hand and kept as it is."""
    lead = "See it and copy the prompt: "
    body = current
    if body.startswith(lead):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.lstrip("\n")
    return f"{lead}{_link(meta.page(name))}\n\n{body}" if live else body


def _row(e: dict) -> str:
    poster = PACK / "effects" / e["name"] / "poster.jpg"
    pic = f'<img src="effects/{e["name"]}/poster.jpg" width="110" alt="">' if poster.exists() else ""
    say = e.get("say") or []
    ask = f"“{_cell(say[0])}”" if say else f"`{_cell(e['type_this'])}`"
    return f"| {pic} | [**{_cell(e['title'])}**](effects/{e['name']}/README.md)<br>{_cell(e['card'])} | {ask} |"


def readme_text(es: list[dict], live: bool = meta.PAGES_LIVE) -> str:
    sections, placed = [], set()
    for title, cats in GROUPS + [("More", ())]:
        group = [e for e in es if (e["category"] in cats) or (not cats and e["name"] not in placed)]
        group = [e for e in group if e["name"] not in placed]
        if not group:
            continue
        placed |= {e["name"] for e in group}
        rows = "\n".join(_row(e) for e in group)
        sections.append(f"### {title}\n\n| | Effect | Just say |\n|---|---|---|\n{rows}")
    body = "\n\n".join(sections)
    first = meta.prompt("clean cut")
    preview = (PACK / "assets" / "preview.gif").exists()
    hero = (f'<p align="center"><img src="assets/preview.gif" width="600" alt="Twelve of the effects, playing"></p>\n\n'
            if preview else "")
    before_after = (f'<p align="center"><img src="assets/clean-cut.gif" width="480" '
                    f'alt="Clean Cut: the raw take, then the pauses and retakes gone and captions on"></p>\n\n'
                    if (PACK / "assets" / "clean-cut.gif").exists() else "")
    return f"""# AI video editing skills

**{len(es)} free video edits for your AI agent.** Film yourself talking, drop the video into Claude
Code or Codex, and say what you want. It cuts the pauses, captions every word, puts a giant word
behind your head, drops in your B-roll, grades the colour and lays your music under your voice.
No editing app, no account, nothing to pay. It all runs on your own computer, so your video is
never uploaded.

{pages_line(es, live)}{hero}<!-- This page is written by tools/build_manifest.py from effects/*/effect.json. Edit those, not this. -->

## Start in one paste

Open Claude Code (the desktop app or the terminal), paste this, and drop your video in after it:

```
{first}
```

Claude installs the skills and makes the edit. After that first time, just ask:

{before_after}| You say | You get |
|---|---|
| "cut the pauses and retakes" | the dead air and false starts gone, captions on |
| "put a giant word behind me when I say city" | the word behind your head, landing as you say it |
| "turn the lights down, with my name in neon" | the room goes dark and you stay lit |
| "cut my b-roll in when I mention each place" | your own footage on the words that name it |
| "add a tier list as I rank these" | each card drops into its row the moment you rank it |
| "lay my song under this" | the music dips while you talk and ends with the video |

You never need to know a command: say it the way you would say it to an editor, with the timing
in plain words ("when I say 'watch'"). Claude finds the exact second. If you like commands, every
effect also has one, `/video-editing <effect> <video>`, listed in its own page.

### Or install it yourself

For Claude Code:

```
git clone {REPO} ~/.claude/skills/video-editing
```

For Codex (the ChatGPT desktop app's agent):

```
git clone {REPO} ~/.codex/skills/video-editing
```

Then start a new chat. The first edit sets itself up in its own folder, a few minutes and about
600 MB of free tools and models. It needs ffmpeg and Python 3.10, 3.11 or 3.12, and says exactly
how to get either if one is missing. Tested on macOS; Linux and Windows are untested so far.

## What it does well

- **It listens.** A free speech model on your computer hears every word and when it is said, so
  captions, titles and graphics land on the word. It skips the word you took back ("the town, or
  the city": it uses *city*) and the talk about the recording ("give me a second").
- **It keeps your face clear.** Words, cards and graphics stay off your face and inside the part of
  the screen Reels, Shorts and TikTok never cover. Some effects put things behind you instead.
- **It checks before it renders.** Every effect shows Claude a few frames first, so a problem is
  caught before the whole video is made. `qc-check` checks the finished file too.
- **Your sound is looked after.** Your voice is kept, and sounds and music sit under it at the
  loudness the apps use.

## All {len(es)} effects

{body}

## What it does not do

- It edits videos of someone talking, in English, with one person in the picture at a time.
- It draws one thing over the picture per video. Run effects one after another: cuts first, then
  colour or anything behind you, then one effect over the picture, then sound.
- It makes no new pictures, voices or music. The music it lays is yours.
- Every result is a finished MP4. Changing it means asking again and rendering again.

## Want to change it without rendering again?

[AskEditor](https://askeditor.com) keeps the same edit open on a canvas: move a caption, retype a
word, swap a picture, with no new prompt and no new render. It also makes what this pack leaves
out: voices, music, generated scenes, and ready-made templates. See the
[templates in AskEditor](https://askeditor.com/templates).

## Licence

The code is MIT licensed. The fonts are under the SIL Open Font License, and the models each keep
their own licence; [NOTICE.md](NOTICE.md) lists them all.
"""


def skill_table(es: list[dict]) -> str:
    rows = ["| Effect | What it does | People say | Needs |", "|---|---|---|---|"]
    for e in es:
        say = "; ".join(f"\u201c{s}\u201d" for s in e.get("say", [])) or "\u2014"
        needs = ", ".join(e["needs"]) or "\u2014"
        rows.append(f"| `{e['name']}` | {_cell(e['line'])} | {_cell(say)} | {needs} |")
    return "\n".join([START, *rows, END])


def skill_text(es: list[dict], current: str) -> str:
    if START not in current or END not in current:
        raise Bad(f"SKILL.md has no {START} … {END} block for the table")
    a, b = current.index(START), current.index(END) + len(END)
    return current[:a] + skill_table(es) + current[b:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/build_manifest.py")
    ap.add_argument("--check", action="store_true", help="fail if a generated file is out of date; write nothing")
    args = ap.parse_args(argv)
    try:
        es = entries()
        skill_path = PACK / "SKILL.md"
        want = {
            PACK / "effects.json": manifest_text(es),
            PACK / "README.md": readme_text(es),
            skill_path: skill_text(es, skill_path.read_text()),
        }
        for e in es:
            path = PACK / "effects" / e["name"] / "README.md"
            if path.exists():
                want[path] = effect_readme_text(e["name"], path.read_text(), meta.PAGES_LIVE)
    except Bad as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stale = [p for p, text in want.items() if not p.exists() or p.read_text() != text]
    if args.check:
        if stale:
            print("out of date: " + ", ".join(str(p.relative_to(PACK)) for p in stale)
                  + ". Run python3 tools/build_manifest.py", file=sys.stderr)
            return 1
        print(f"up to date ({len(es)} effects)")
        return 0
    for p in stale:
        p.write_text(want[p])
    print(f"{len(es)} effects; " + (f"wrote {', '.join(str(p.relative_to(PACK)) for p in stale)}" if stale else "nothing changed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
