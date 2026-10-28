"""post-copy: the words a person posts with the video, written from what is said in it.

Nothing is rendered. The effect writes `<video>.post.md`, and the agent running the pack is the
writer:

1. **What is said, in its parts.** The transcript split where the talk moves on (the same parts as
   `chapter-cards`, from `fx/core/parts.py`), each with its times and a title of its key words.
2. **A brief.** The video's length and shape; the line that makes the point, chosen by the
   core's picker the way `headline` chooses its line, and the strongest line of each part; the key words, the names and the
   numbers heard (`fx/core/amounts.py`); and the rules for the copy: the speaker's own claims only.
3. **The limits.** Each platform's limits as its own help pages gave them, with the page and the
   day they were read. A limit nobody publishes is never guessed, and never checked.
4. **YouTube's chapter lines**, exactly as YouTube takes them: `00:00` first, at least three, in
   order, each part 10 seconds or longer. A video that cannot hold three such parts gets none, and
   the file says why.
5. **A block to fill for each piece of copy**: a title and a caption for Reels, TikTok, Shorts,
   YouTube and LinkedIn, the hashtags, and a short post. Each sits between markers.

Then `check=on` reads the filled file and checks every block against those limits, counting the
way each platform counts (TikTok in UTF-16 units, YouTube's description in characters and in bytes,
X with its weights), and says what is wrong. A filled file is never written over.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from ..core import parts as partsmod
from ..core.words import unsure_stretches
from . import options as O

NAME = "post-copy"
WRITES = ".post.md"
READ_ON = "28 October 2026"  # the day the limits below were read from each platform's own pages


@dataclass(frozen=True)
class MinPart:
    """auto, or the shortest a chapter may be in seconds (YouTube takes 10 or more)."""
    help: str

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().lower().rstrip("s")
        if v in ("auto", ""):
            return "auto"
        try:
            x = float(v)
        except ValueError:
            raise O.OptionError(f"say auto or give seconds from 10 to 600 (got {raw!r})") from None
        if not 10 <= x <= 600:
            raise O.OptionError(f"give seconds from 10 to 600: YouTube takes no chapter shorter than 10 (got {raw!r})")
        return x


OPTIONS = {
    "titles": O.Text("auto: each part named by its key words (the chapter lines' titles) · a number: that many parts · "
                     "or your own titles between | marks, each may begin with its time. Give chapter-cards the same "
                     "titles and the chapters match the cards"),
    "min_part": MinPart("the shortest a chapter may be: auto (10 s, YouTube's least, or a tenth of a long video) or "
                        "seconds, from 10"),
    "platforms": O.Text("all, or the ones to write for, from: reels, tiktok, shorts, youtube, linkedin"),
    "check": O.Switch("off: write the file for the copy · on: check the copy already written into it"),
}
PLATFORMS = ("reels", "tiktok", "shorts", "youtube", "linkedin")

# The matrix renders videos; this effect writes a file, so it has no runs there. Its counting and
# checking are tested in tests/post_copy.py, and its sample files are made by running it.
MATRIX: list = []

# ---- the limits, as each platform's own pages gave them on READ_ON ----------------------------------------

SOURCES = {
    "instagram-hashtags": ("Instagram Help Center, “Use hashtags on Instagram”",
                           "https://help.instagram.com/351460621611097"),
    "instagram-caption": ("Instagram Platform docs, “IG User Media” (updated 28 September 2026)",
                          "https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media"),
    "tiktok": ("TikTok for Developers, Content Posting API, “Direct Post”",
               "https://developers.tiktok.com/doc/content-posting-api-reference-direct-post"),
    "youtube-details": ("YouTube Help, “Edit video settings”", "https://support.google.com/youtube/answer/57404"),
    "youtube-api": ("YouTube Data API, “Videos”", "https://developers.google.com/youtube/v3/docs/videos"),
    "youtube-hashtags": ("YouTube Help, “Find playlists & videos using hashtags”",
                         "https://support.google.com/youtube/answer/6390658"),
    "youtube-chapters": ("YouTube Help, “Video Chapters”", "https://support.google.com/youtube/answer/9884579"),
    "shorts": ("YouTube Help, “Upload YouTube Shorts”", "https://support.google.com/youtube/answer/12779649"),
    "linkedin": ("LinkedIn Help, “Post and share updates”", "https://www.linkedin.com/help/linkedin/answer/a528176"),
    "x": ("X Developer Platform, “Counting characters”, and the configuration it links to",
          "https://docs.x.com/fundamentals/counting-characters"),
}

INSTAGRAM_CAPTION = 2200  # characters; the same page allows 20 @ tags
INSTAGRAM_MENTIONS = 20
INSTAGRAM_HASHTAGS = 5  # the Help Center: "up to 5"; the developer page still says 30, so the app's 5 is kept
TIKTOK_CAPTION = 2200  # UTF-16 units, for a caption posted through TikTok's own posting interface
YOUTUBE_TITLE = 100  # characters, no < or >
YOUTUBE_DESCRIPTION = 5000  # characters in YouTube Studio; 5000 bytes through the YouTube API
YOUTUBE_HASHTAGS = 60  # more than this and YouTube ignores every hashtag on the video
YOUTUBE_SHOWN = 3  # up to three hashtags show by the title
LINKEDIN_POST = 3000  # characters
X_POST = 280  # weighted: a link counts 23, an emoji 2, CJK and most scripts 2
X_LIGHT = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))  # code points that weigh 1
X_LINK = 23

# Each block the agent fills: (its key between the markers, its heading, what to write in it)
BLOCKS = [
    ("reels.title", "Reels: title", "Reels has no title field: this is the line on the cover and the caption's first "
     "line, the one people see before the rest is folded away. A few words."),
    ("reels.caption", "Reels: caption", "Starts with the title above. Then the point of the video in two or three "
     f"short lines, in the speaker's words. Hashtags last, {INSTAGRAM_HASHTAGS} at most."),
    ("tiktok.title", "TikTok: title", "TikTok has no title field for a video: this is the cover's words and the "
     "caption's first line. A few words."),
    ("tiktok.caption", "TikTok: caption", "Starts with the title above, then a line or two and the hashtags."),
    ("shorts.title", "Shorts: title", f"At most {YOUTUBE_TITLE} characters, no < or >. What a viewer gets from it, in "
     "plain words."),
    ("shorts.description", "Shorts: description", "Two or three lines; up to three hashtags at the end (they can show by "
     "the title)."),
    ("youtube.title", "YouTube: title", f"At most {YOUTUBE_TITLE} characters, no < or >. It may be longer and more "
     "searchable than the Shorts title."),
    ("youtube.description", "YouTube: description", "A short paragraph on what the video covers, then the chapter lines "
     "below exactly as they are (when there are any), then up to three hashtags."),
    ("linkedin.title", "LinkedIn: title", "LinkedIn has no title field: this is the post's first line, the one people "
     "see before it is folded away."),
    ("linkedin.post", "LinkedIn: post", "Starts with the title above. A few short paragraphs: what was done or learned "
     "and why it matters, in the first person if the speaker talks that way. Two or three hashtags at most, or none."),
    ("hashtags", "Hashtags", "The hashtags for this video, most specific first, one word each with no spaces or $ or % "
     "(#chickenpatty). Each platform's caption above takes the ones it has room for."),
    ("short.post", "A short post", f"To share the video anywhere (X, Threads, a newsletter): one to three sentences, "
     f"{X_POST} characters at most as X counts them (a link counts {X_LINK})."),
]


# ---- counting the way each platform counts ----------------------------------------------------------------

_LINK = re.compile(r"(?:https?://|www\.)\S+"
                   r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|ai|co|app|dev|me|tv|ly|gg|so)\b(?:/\S*)?", re.I)
_HASHTAG = re.compile(r"(?<![\w#])#(\S+)")
_MENTION = re.compile(r"(?<![\w@])@[\w.]+")


def utf16(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _emoji(cp: int) -> bool:
    return (0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF or 0x2B00 <= cp <= 0x2BFF or 0x1F1E6 <= cp <= 0x1F1FF
            or cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x3030, 0x303D, 0x3297, 0x3299))


def x_length(text: str) -> int:
    """A post's length as X counts it: a link 23, an emoji (whatever it is joined from) 2, a code point
    in X's light ranges 1, any other 2."""
    total = 0
    pos = 0
    for m in _LINK.finditer(text):
        total += _x_plain(text[pos:m.start()]) + X_LINK
        pos = m.end()
    return total + _x_plain(text[pos:])


def _x_plain(text: str) -> int:
    cps = [ord(c) for c in text]
    total = i = 0
    while i < len(cps):
        cp = cps[i]
        keycap = chr(cp) in "0123456789#*" and i + 1 < len(cps) and cps[i + 1] in (0xFE0F, 0x20E3)
        if _emoji(cp) or keycap:
            total += 2  # one emoji, however it is built
            i += 1
            if 0x1F1E6 <= cp <= 0x1F1FF and i < len(cps) and 0x1F1E6 <= cps[i] <= 0x1F1FF:
                i += 1  # a flag is two letters
            while i < len(cps):
                if 0x1F3FB <= cps[i] <= 0x1F3FF or cps[i] in (0xFE0F, 0x20E3):
                    i += 1  # a skin tone, the emoji style mark or a keycap belongs to it
                elif cps[i] == 0x200D and i + 1 < len(cps):
                    i += 2  # joined to the next emoji: still the one emoji
                else:
                    break
            continue
        total += 1 if any(a <= cp <= b for a, b in X_LIGHT) else 2
        i += 1
    return total


def hashtags(text: str) -> list[str]:
    return ["#" + h.rstrip(".,;:!?)’”\"'") for h in _HASHTAG.findall(text)]


# ---- the file -------------------------------------------------------------------------------------------------

_BLOCK = re.compile(r"<!-- ([a-z.]+) -->\n(.*?)\n?<!-- /\1 -->", re.S)
FILL = "FILL:"


def _clock(t: float) -> str:
    return partsmod.stamp(t)


def _quote(text: str, width: int = 100) -> str:
    out, line = [], ""
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            out.append("> " + line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        out.append("> " + line)
    return "\n".join(out) or "> (nothing heard)"


def _names(words) -> list[str]:
    """Words said with a capital in the middle of a sentence, or as an acronym: names, places,
    products. The transcriber spells these worst, so the brief lists them to check."""
    out: dict[str, int] = {}
    for i, w in enumerate(words):
        t = w.clean.strip().strip("\"'“”‘’.,;:!?()[]")
        if len(t) < 2 or t.lower() in ("i", "i'm", "i've", "i'll", "i'd"):
            continue
        acronym = t.isupper() and t.isalpha()
        mid = i > 0 and not words[i - 1].ends_sentence and t[:1].isupper()
        inner = any(c.isupper() for c in t[1:])
        if acronym or mid or inner:
            # join two capitalised words said together ("Hua Hin")
            if i > 0 and out and words[i - 1].clean.strip()[:1].isupper() and not words[i - 1].ends_clause:
                prev = words[i - 1].clean.strip().strip(".,;:!?")
                key = f"{prev} {t}"
                if prev in out:
                    out[key] = out.pop(prev)
                    continue
            out[t] = out.get(t, 0) + 1
    return list(out)[:12]


def _line_of(words, span: Optional[tuple[int, int]]) -> str:
    if span is None:
        return ""
    a, b = span
    return " ".join(w.text for w in words[a:b + 1]).strip()


def _thin(words, span: tuple[int, int]) -> bool:
    """A line with less than two words of subject in it ("First of all,"): a signpost, not a point."""
    return sum(1 for w in words[span[0]:span[1] + 1] if partsmod.content(w)) < 2


def ranked_lines(words, quiet) -> list[tuple[int, int]]:
    """Every line the core's picker weighs (the one the headline effect sets), strongest first,
    leaving out a line that is only a signpost ("First of all,") and anything the speaker took back."""
    from ..core import moments

    ranked: list = []
    moments.pick_line(list(words), quiet, ranked=ranked)
    return [span for _, span in ranked if not _thin(words, span)]


def in_its_sentence(words, span: Optional[tuple[int, int]], most: int = 28) -> str:
    """The sentence a line was said in, with the line itself in bold."""
    if span is None:
        return "none found"
    a, b = span
    lo = a
    while lo > 0 and not words[lo - 1].ends_sentence and a - lo < most // 2:
        lo -= 1
    hi = b
    while hi + 1 < len(words) and not words[hi].ends_sentence and hi - b < most // 2:
        hi += 1
    before = " ".join(w.text for w in words[lo:a])
    line = " ".join(w.text for w in words[a:b + 1])
    after = " ".join(w.text for w in words[b + 1:hi + 1])
    text = " ".join(x for x in (before, f"**{line}**", after) if x)
    cut_before = "\u2026 " if lo > 0 and not words[lo - 1].ends_sentence else ""
    cut_after = " \u2026" if not words[hi].ends_sentence else ""
    return f"\u201c{cut_before}{text}{cut_after}\u201d"


class PostCopy:
    def __init__(self, clip, opts: dict):
        self.clip = clip
        self.opts = opts
        self.duration = clip.info.frames / clip.info.rate
        wanted = [p.strip().lower() for p in re.split(r"[,\s]+", opts["platforms"]) if p.strip()]
        if not wanted or wanted == ["all"]:
            wanted = list(PLATFORMS)
        bad = [p for p in wanted if p not in PLATFORMS]
        if bad:
            raise O.OptionError(f"platforms: choose from {', '.join(PLATFORMS)} (got {', '.join(bad)})")
        self.platforms = [p for p in PLATFORMS if p in wanted]

    # -- the parts, as chapters
    def parts(self):
        words, quiet = self.clip.words, self.clip.quiet
        count, given = partsmod.parse_titles(self.opts["titles"])
        floor = (max(partsmod.YOUTUBE_SHORTEST, partsmod.auto_min_part(self.duration))
                 if self.opts["min_part"] == "auto" else float(self.opts["min_part"]))
        found = partsmod.find(words, quiet, self.duration, min_part=floor, count=count, titles=given,
                              first_min=partsmod.YOUTUBE_SHORTEST)
        return found, floor

    def blocks_wanted(self) -> list[tuple[str, str, str]]:
        keep = []
        for key, head, what in BLOCKS:
            plat = key.split(".")[0]
            if plat in PLATFORMS and plat not in self.platforms:
                continue
            keep.append((key, head, what))
        return keep

    # -- writing the file
    def write(self, out: Path) -> dict:
        if out.exists():
            old = out.read_text()
            mine = {key for key, _, _ in BLOCKS}  # the pieces the agent writes; the chapter lines are ours
            filled = [k for k, body in _BLOCK.findall(old) if k in mine and body.strip() and FILL not in body]
            if filled:
                raise ValueError(f"{out.name} already has copy written in it ({', '.join(filled[:4])}); it is never "
                                 f"written over. Check it with --opt check=on, or move it away to start again")
        clip = self.clip
        words = clip.words
        found, floor = self.parts()
        lines = partsmod.chapter_lines(found, self.duration)
        problems = partsmod.check_chapters(lines, self.duration)
        info = clip.info
        shape = "vertical" if info.height > info.width else ("square" if info.height == info.width else "landscape")
        lines_ranked = ranked_lines(words, clip.quiet)
        best = lines_ranked[0] if lines_ranked else None
        from ..core import amounts, moments

        back = moments.taken_back(words, clip.quiet)
        said = [a for a in amounts.spoken(words) if not back.holds(words, a.first, a.last)]
        # every number heard, as heard (never one taken back, in a first attempt or an aside):
        # amounts, and years or lengths of time marked as such
        heard = [a.said + ("" if a.kind == "amount" else f" ({a.kind})") for a in said if not a.weak and a.said]
        nums = ", ".join(dict.fromkeys(heard)) or "none"
        # each part's most telling words, never one that another already holds ("train" in "train station")
        keys: list[str] = []
        for p in found:
            for k in p.keys[:4]:
                mine = {partsmod._stem(x.lower()) for x in k.split()}
                if any(mine <= {partsmod._stem(x.lower()) for x in o.split()} for o in keys):
                    continue
                keys = [o for o in keys if not {partsmod._stem(x.lower()) for x in o.split()} <= mine] + [k]
        out_lines = [
            f"# Post copy: {Path(clip.path).name}",
            "",
            f"Written by the video-editing pack's `post-copy` on {date.today():%d %B %Y} from `{Path(clip.path).name}` "
            f"({_clock(self.duration)}, {info.width}x{info.height}). Nothing in it has been posted anywhere.",
            "",
            "**For the agent writing the copy:** read the two sections below, then replace every `FILL:` line in "
            "“The copy” with copy of your own, keeping the `<!-- … -->` markers around each piece. Then "
            "check it, and fix whatever the check names:",
            "",
            "```",
            f".venv/bin/python -m fx.run post-copy \"{clip.path}\" --opt check=on",
            "```",
            "",
            "## What is said, in its parts",
            "",
        ]
        for p in found:
            out_lines += [f"### {p.index + 1} · {_clock(p.start)}–{_clock(p.end)} · {p.title}", "",
                          _quote(p.said(words)), ""]
        per_part = []
        for p in found:
            span = next((sp for sp in lines_ranked if p.first <= sp[0] and sp[1] <= p.last), None)
            if span:
                per_part.append(f"{p.index + 1}: \u201c{_line_of(words, span)}\u201d")
        out_lines += [
            "## The brief",
            "",
            f"- **The video:** {_clock(self.duration)} long, {shape} ({info.width}x{info.height}), one speaker, English.",
            "- **The line that makes the point** (picked as the headline effect picks its line): "
            + in_its_sentence(words, best) + (f" at {_clock(words[best[0]].start)}" if best else "") + ".",
            "- **The strongest line of each part:** " + ("; ".join(per_part) or "none found") + ".",
            "- **Key words:** " + (", ".join(keys[:10]) or "none") + ".",
            "- **Names heard** (the transcriber spells names worst: check each one before it goes in a title): "
            + (", ".join(_names(words)) or "none") + ".",
            f"- **Numbers said:** {nums}.",
            "- **How to write it:**",
            "  - Only what the speaker says or shows. No claim, number, price, result or promise of your own, and "
            "nothing the video does not deliver.",
            "  - In their words and their tone: their phrases are better than marketing ones.",
            "  - Each title and first line says what a viewer gets, plainly, so it is worth stopping for.",
            "  - Hashtags only for what the video is really about, most specific first.",
        ]
        if clip.unsure:
            out_lines.append("- **Heard but not used** (a second listen could not be sure of these, so they are left "
                             "out of everything above): " + "; ".join(f"“{u['w']}” at {u['s']:.1f} s"
                                                                   for u in unsure_stretches(clip.unsure)) + ".")
        out_lines += ["", f"## The limits, read on {READ_ON}", "",
                      "From each platform's own pages. Nothing here is guessed: a limit no platform publishes (how much "
                      "of a caption shows before it is folded away, say) is not listed and not checked.", "",
                      "| Where | Title | Caption or description | Hashtags | Read from |", "|---|---|---|---|---|"]
        rows = {
            "reels": f"| Instagram Reels | none (the caption's first line) | {INSTAGRAM_CAPTION:,} characters, "
                     f"{INSTAGRAM_MENTIONS} @ mentions at most | {INSTAGRAM_HASHTAGS} at most | "
                     f"{_src('instagram-caption')}; {_src('instagram-hashtags')} (the developer page still says 30 "
                     f"hashtags; the app's own help says 5, so 5 is kept) |",
            "tiktok": f"| TikTok | none (the caption's first line) | {TIKTOK_CAPTION:,} characters, counted in UTF-16 "
                      f"units (most emoji count 2) | not limited on the page | {_src('tiktok')} (the posting "
                      f"interface; the app may allow more) |",
            "shorts": f"| YouTube Shorts | {YOUTUBE_TITLE} characters, no < or > | {YOUTUBE_DESCRIPTION:,} characters, "
                      f"no < or > | more than {YOUTUBE_HASHTAGS} and all are ignored; up to {YOUTUBE_SHOWN} show by the "
                      f"title | {_src('youtube-details')}; {_src('youtube-hashtags')}; {_src('shorts')} (a Short runs "
                      f"up to 3 minutes) |",
            "youtube": f"| YouTube | {YOUTUBE_TITLE} characters, no < or > | {YOUTUBE_DESCRIPTION:,} characters (and "
                       f"{YOUTUBE_DESCRIPTION:,} bytes through the API), no < or > | as Shorts | "
                       f"{_src('youtube-details')}; {_src('youtube-api')}; {_src('youtube-hashtags')} |",
            "linkedin": f"| LinkedIn | none (the post's first line) | {LINKEDIN_POST:,} characters | not limited on "
                        f"the page | {_src('linkedin')} |",
        }
        out_lines += [rows[p] for p in self.platforms]
        out_lines += [f"| A short post (X, Threads, a newsletter) | | {X_POST} as X counts them: a link "
                      f"{X_LINK}, an emoji 2, most non-Latin letters 2 | | {_src('x')} |", ""]
        out_lines += ["## YouTube chapters", ""]
        if "youtube" in self.platforms:
            if problems:
                out_lines += [f"None: YouTube needs at least three chapters of 10 seconds or more, starting at 00:00 "
                              f"({_src('youtube-chapters')}), and this video holds {len(found)} part"
                              f"{'' if len(found) == 1 else 's'} of that length where the talk moves on. Leave "
                              "chapters out of the description.", ""]
            else:
                out_lines += [f"These follow YouTube's rules ({_src('youtube-chapters')}: the first at 00:00, at least "
                              "three, in order, each 10 seconds or more). Put them in the description exactly as they "
                              "are; a title may be reworded, but keep each time.", "",
                              "<!-- youtube.chapters -->", *lines, "<!-- /youtube.chapters -->", ""]
        else:
            out_lines += ["YouTube was not asked for.", ""]
        out_lines += ["## The copy", ""]
        for key, head, what in self.blocks_wanted():
            out_lines += [f"### {head}", "", f"<!-- {key} -->", f"{FILL} {what}", f"<!-- /{key} -->", ""]
        out.write_text("\n".join(out_lines).rstrip() + "\n")
        return {"post": str(out), "chapters": (f"{len(lines)} lines for YouTube" if not problems
                                                  else "none: " + "; ".join(problems)),
                "next": f"write the copy into each FILL: line of {out.name}, then run it again with --opt check=on"}

    # -- checking the copy
    def check(self, out: Path) -> dict:
        if not out.exists():
            raise ValueError(f"there is no {out.name} to check yet; run without check=on first to write it")
        text = out.read_text()
        found = dict(_BLOCK.findall(text))
        rows, failed = [], False

        def say(ok: bool, key: str, msg: str, soft: bool = False) -> None:
            nonlocal failed
            mark = "ok  " if ok else ("note" if soft else "FAIL")
            rows.append(f"{mark} {key}: {msg}")
            if not ok and not soft:
                failed = True

        chapters = [ln for ln in found.get("youtube.chapters", "").splitlines() if ln.strip()]
        for key, head, _ in self.blocks_wanted():
            body = found.get(key)
            if body is None:
                say(False, key, "the block is missing (keep the markers around each piece)")
                continue
            body = body.strip()
            if not body or FILL in body:
                say(False, key, "not written yet")
                continue
            n = len(body)
            if key == "reels.caption":
                tags, ats = hashtags(body), _MENTION.findall(body)
                ok = n <= INSTAGRAM_CAPTION and len(tags) <= INSTAGRAM_HASHTAGS and len(ats) <= INSTAGRAM_MENTIONS
                say(ok, key, f"{n:,} of {INSTAGRAM_CAPTION:,} characters, {len(tags)} of {INSTAGRAM_HASHTAGS} "
                             f"hashtags, {len(ats)} of {INSTAGRAM_MENTIONS} mentions")
                self._first_line(say, key, body, found.get("reels.title", ""))
                self._tags_ok(say, key, tags)
            elif key == "tiktok.caption":
                u = utf16(body)
                say(u <= TIKTOK_CAPTION, key, f"{u:,} of {TIKTOK_CAPTION:,} UTF-16 units")
                self._first_line(say, key, body, found.get("tiktok.title", ""))
                self._tags_ok(say, key, hashtags(body))
            elif key in ("shorts.title", "youtube.title"):
                say(n <= YOUTUBE_TITLE and not re.search(r"[<>]", body), key,
                    f"{n} of {YOUTUBE_TITLE} characters" + (", but < or > is not allowed" if re.search(r"[<>]", body)
                                                             else ""))
                if "\n" in body:
                    say(False, key, "a title is one line")
            elif key in ("shorts.description", "youtube.description"):
                b = len(body.encode("utf-8"))
                title = found.get(key.split(".")[0] + ".title", "")
                tags = hashtags(body + " " + title)
                ok = n <= YOUTUBE_DESCRIPTION and b <= YOUTUBE_DESCRIPTION and not re.search(r"[<>]", body)
                say(ok, key, f"{n:,} of {YOUTUBE_DESCRIPTION:,} characters ({b:,} bytes)"
                             + (", but < or > is not allowed" if re.search(r"[<>]", body) else ""))
                say(len(tags) <= YOUTUBE_HASHTAGS, key, f"{len(tags)} hashtags (YouTube ignores them all past "
                                                        f"{YOUTUBE_HASHTAGS}; up to {YOUTUBE_SHOWN} show by the title)")
                self._tags_ok(say, key, tags)
                if key == "youtube.description" and chapters:
                    have = [ln.strip() for ln in body.splitlines()]
                    at = [have.index(c.strip()) if c.strip() in have else -1 for c in chapters]
                    whole = all(i >= 0 for i in at) and at == sorted(at)
                    say(whole, key, "the chapter lines are in it, in order" if whole else
                        "the chapter lines must be in it exactly as they are, in order, one a line")
            elif key == "linkedin.post":
                say(n <= LINKEDIN_POST, key, f"{n:,} of {LINKEDIN_POST:,} characters")
                self._first_line(say, key, body, found.get("linkedin.title", ""))
            elif key == "short.post":
                x = x_length(body)
                say(x <= X_POST, key, f"{x} of {X_POST} as X counts them")
            elif key == "hashtags":
                tags = hashtags(body)
                say(bool(tags), key, f"{len(tags)} hashtags" if tags else "no hashtags in it")
                self._tags_ok(say, key, tags)
            elif key.endswith(".title"):
                say("\n" not in body, key, f"{n} characters" if "\n" not in body else "a title is one line")
        if chapters:
            probs = partsmod.check_chapters(chapters, self.duration)
            say(not probs, "youtube.chapters", "they pass YouTube's rules" if not probs else "; ".join(probs))
        return {"checked": str(out), "results": "\n  " + "\n  ".join(rows),
                "verdict": "every piece is within its limits" if not failed else "fix the FAIL lines and check again",
                "failed": failed}

    @staticmethod
    def _first_line(say, key: str, body: str, title: str) -> None:
        title = title.strip()
        if not title or FILL in title:
            return
        first = body.strip().splitlines()[0].strip()
        say(first.startswith(title.rstrip(".!?:")) or title.startswith(first), key,
            "starts with its title" if first.startswith(title.rstrip(".!?:")) or title.startswith(first) else
            "should start with the title above: this platform has no title field, so the first line is the title",
            soft=True)

    @staticmethod
    def _tags_ok(say, key: str, tags: list[str]) -> None:
        bad = [t for t in tags if not re.fullmatch(r"#[^\W]+", t)]
        if bad:
            say(False, key, "a hashtag is one word of letters, numbers or _, with no spaces or $ or %: "
                            + ", ".join(bad[:5]))


def _src(key: str) -> str:
    name, url = SOURCES[key]
    return f"[{name}]({url})"


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job

    info = clip.info
    pc = PostCopy(clip, opts)
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, None, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    checks: dict = {}
    if not opts["check"]:
        found, floor = pc.parts()
        checks["parts"] = " · ".join(f"{p.index + 1} {partsmod.stamp(p.start)} {p.title}" for p in found)
        checks["chapters_at_least"] = f"{floor:g} s each"
    job = Job(NAME, clip, tl, fr, None, {"effect": NAME, "clip": clip.path, "options": opts}, checks=checks)
    job.writes = pc.check if opts["check"] else pc.write
    return job
