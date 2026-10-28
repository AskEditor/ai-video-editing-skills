"""Where the pack lives, in one place: the repository's address and the pages that show each effect.

`tools/build_manifest.py` (the manifest, the READMEs) and `fx/run.py` (the line after a render) both
read these, so a rename of the repository is a change to `REPO` alone. Standard library only, and
importing it loads nothing else from `fx`.
"""

from __future__ import annotations

REPO = "github.com/AskEditor/ai-video-editing-skills"
REPO_URL = f"https://{REPO}"

PAGES = "askeditor.com/skills"  # the directory; each effect has its own page at PAGES/<id>
# Off until askeditor.com/skills/<id> answers: while off, no README and no render points at the pages.
PAGES_LIVE = False


def prompt(do: str) -> str:
    """The one paste a page offers: install the pack, then do one effect (`do` is the effect's
    request in the person's words, as its effect.json gives it: "clean cut")."""
    # It names the folder, so the agent installs the pack where Claude Code loads skills from.
    return f"Install the free video editing skills from {REPO} into ~/.claude/skills/video-editing, then {do} my video:"


def page(name: str) -> str:
    """The effect's own page (it answers only once PAGES_LIVE is on)."""
    return f"{PAGES}/{name}"
