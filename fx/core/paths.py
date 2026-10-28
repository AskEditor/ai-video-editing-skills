"""Where the pack keeps things: its own folder, the models, the fonts, and the caches next to a clip."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PACK = Path(__file__).resolve().parents[2]
MODELS = PACK / "models"
FONTS = PACK / "fonts"
CACHE = PACK / ".cache"


class NeedSetup(RuntimeError):
    """Something setup.sh provides is missing."""


def beside(clip: str | os.PathLike, suffix: str) -> Path:
    """The cache file for `suffix` next to the clip (`talk.mp4` -> `talk.words.json`).

    When the clip's folder cannot be written, the file goes in the pack's own `.cache/` instead,
    under a folder named after the clip's full path.
    """
    clip = Path(clip).resolve()
    target = clip.with_name(f"{clip.stem}.{suffix}")
    if os.access(clip.parent, os.W_OK):
        return target
    key = hashlib.sha1(str(clip).encode()).hexdigest()[:12]
    folder = CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    return folder / target.name


def source_signature(clip: str | os.PathLike) -> dict:
    """What a cache records about its clip, so a changed clip is never read with a stale cache."""
    st = Path(clip).stat()
    return {"name": Path(clip).name, "bytes": st.st_size}
