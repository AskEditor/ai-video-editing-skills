"""Check that no sentence of ours was copied from another skill pack.

    python3 tests/nocopy.py owner/repo [owner/repo …] [--refs DIR]

For each reference repository it fetches the text of its README and SKILL.md files with `gh api`
(the text only: nothing is cloned, installed or run) into `<refs>/<owner>__<repo>/`, and then
compares our own words against them: SKILL.md, every README, every effect.json's text, and the
help line of every option a person can read.

Both sides are lowercased, stripped of punctuation and split into runs of eight words
(shingles). Code is left out first on both sides: fenced code blocks, `inline code` and command
lines, so a shared `git clone` line or option name is not a finding. Any run of eight or more words
that appears in both fails the check, and is printed with where it is in our files.
"""

from __future__ import annotations

import os
import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
# The reference texts compared against are not in this repository: FX_LAB/refs, or tests/lab/refs.
REFS = Path(os.environ.get("FX_LAB") or Path(__file__).resolve().parent / "lab") / "refs"
if __name__ == "__main__" and not REFS.exists():
    print("skipped: these checks need the test clips, which are not in this repository. "
          "Set FX_LAB to their folder, or put them at tests/lab.")
    raise SystemExit(0)
N = 8
COMMAND = re.compile(r"^\s*(\$ |>\s|git |cd |python|pip |brew |bash |sudo |winget |npm |npx |/video-editing|\.venv|ffmpeg |claude )")


def prose(text: str) -> list[str]:
    """The words a person reads: code blocks, inline code, command lines, HTML and link targets removed."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)  # keep a link's words, drop where it points
    text = re.sub(r"https?://\S+", " ", text)
    lines = [ln for ln in text.splitlines() if not COMMAND.match(ln)]
    words = []
    for ln in lines:
        ln = ln.lower().replace("’", "'")
        ln = re.sub(r"[^a-z0-9\s]", " ", ln)
        words.extend(ln.split())
    return words


def shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + N]) for i in range(len(words) - N + 1)}


def ours() -> dict[str, list[str]]:
    """Every piece of our text a person reads, by where it lives."""
    out: dict[str, list[str]] = {}
    for p in [PACK / "SKILL.md", PACK / "README.md", *sorted((PACK / "effects").glob("*/README.md"))]:
        if p.exists():
            out[str(p.relative_to(PACK))] = prose(p.read_text())
    sys.path.insert(0, str(PACK))
    for p in sorted((PACK / "effects").glob("*/effect.json")):
        e = json.loads(p.read_text())
        bits = [e.get("title", ""), e.get("line", ""), *e.get("say", []), e.get("askeditor", {}).get("what", ""),
                e.get("askeditor", {}).get("why", "")]
        out[str(p.relative_to(PACK))] = prose("\n".join(bits))
        try:
            module = importlib.import_module("fx.effects." + e["name"].replace("-", "_"))
            helps = [getattr(kind, "help", "") for kind in module.OPTIONS.values()]
            out[f"fx/effects/{e['name'].replace('-', '_')}.py (option help)"] = prose("\n".join(helps))
        except Exception:  # an effect still being built: its manifest text is still checked
            pass
    return out


def fetch(repo: str, refs: Path) -> list[Path]:
    """The reference's README and SKILL.md files as text, saved under refs/<owner>__<repo>/."""
    owner, name = repo.split("/", 1)
    dest = refs / f"{owner}__{name}"
    dest.mkdir(parents=True, exist_ok=True)
    meta = subprocess.run(["gh", "api", f"repos/{repo}"], capture_output=True, text=True)
    if meta.returncode != 0:
        raise SystemExit(f"could not read {repo} with gh api: {meta.stderr.strip()}")
    branch = json.loads(meta.stdout).get("default_branch", "main")
    tree = subprocess.run(["gh", "api", f"repos/{repo}/git/trees/{branch}?recursive=1"], capture_output=True, text=True)
    if tree.returncode != 0:
        raise SystemExit(f"could not list {repo}: {tree.stderr.strip()}")
    paths = [t["path"] for t in json.loads(tree.stdout).get("tree", [])
             if t.get("type") == "blob" and Path(t["path"]).name.lower() in ("readme.md", "skill.md")]
    saved = []
    for path in paths:
        got = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw", f"repos/{repo}/contents/{path}?ref={branch}"],
                             capture_output=True, text=True)
        if got.returncode == 0:
            target = dest / path.replace("/", "__")
            target.write_text(got.stdout)
            saved.append(target)
    (dest / "FETCHED.txt").write_text(f"{repo} at {branch}: " + ", ".join(paths) + "\n")
    return saved


def runs(ours_words: list[str], theirs: set[tuple[str, ...]]) -> list[str]:
    """Our shared stretches, each joined into one run of eight or more words."""
    hits = [i for i in range(len(ours_words) - N + 1) if tuple(ours_words[i:i + N]) in theirs]
    out, start, prev = [], None, None
    for i in hits + [None]:
        if i is not None and prev is not None and i == prev + 1:
            prev = i
            continue
        if start is not None:
            out.append(" ".join(ours_words[start:prev + N]))
        start = prev = i
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--refs", default=str(REFS))
    args = ap.parse_args(argv)
    mine = ours()
    total_words = sum(len(w) for w in mine.values())
    failed = False
    for repo in args.repos:
        files = fetch(repo, Path(args.refs))
        theirs: set[tuple[str, ...]] = set()
        their_words = 0
        for f in files:
            w = prose(f.read_text())
            their_words += len(w)
            theirs |= shingles(w)
        found = {where: runs(words, theirs) for where, words in mine.items()}
        found = {k: v for k, v in found.items() if v}
        print(f"{repo}: {len(files)} file(s), {their_words} words of theirs against {total_words} of ours "
              f"in {len(mine)} places: " + ("no shared run of 8 words" if not found else "SHARED RUNS FOUND"))
        for where, rs in found.items():
            for r in rs:
                print(f"  {where}: \"{r}\"")
        failed |= bool(found)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
