"""The rules of pictures-above-head, checked on transcripts and heads written for the test (no clip).

    .venv/bin/python tests/pictures_above_head.py

A short talk, timed word by word, is matched against pictures named for what they show, one listed
in a tags file, two with only a camera's number for a name, and a logo. Every picture must land on
the first frame of its word; a picture whose name is never said is left out with its reason; the
camera-numbered ones take the strongest things named, in folder order; a logo lands only where its
brand's whole name is said, never on one of its words alone, and never on a thing named. Pictures
named while others are up share the screen, and leave together. Placed round heads drawn for the
test, every tile stays inside the safe zone and clear of the face, and no picture is ever squashed.
Exits 1 on any failure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import match  # noqa: E402
from fx.core.layout import Rect  # noqa: E402
from fx.core.words import Word  # noqa: E402
from fx.effects import pictures_above_head as P  # noqa: E402

TALK = ("Hi, it's Sam. I flew into the city, then took the train to the coast. My hotel was tiny, but the "
        "sea was calm. I edit everything in AskEditor now, so just ask for it. The harbour and the market "
        "were the best part of the trip.")


def timed(text: str) -> tuple[list[Word], list[tuple[float, float]], float]:
    """Words a third of a second each, a real pause after each sentence."""
    words, quiet, t = [], [(0.0, 0.6)], 0.6
    for raw in text.split():
        length = 0.18 + 0.035 * len(raw.strip(".,"))
        words.append(Word(raw, round(t, 3), round(t + length, 3)))
        t += length
        if raw.endswith("."):
            quiet.append((round(t, 3), round(t + 0.45, 3)))
            t += 0.45
    quiet.append((round(t, 3), round(t + 0.8, 3)))
    return words, quiet, round(t + 0.8, 3)


def picture(name: str, kind: str = "photo", shape: tuple[int, int] = (400, 300), tags: tuple = (),
            logo: bool = False) -> P.Picture:
    h, w = shape
    rgb = np.zeros((h, w, 3), np.uint8)
    p = P.Picture(f"/pictures/{name}", kind, rgb, None, w, h, tags=tags, logo=logo)
    p.names = match.brand_words(name) if logo else match.name_words(name)
    return p


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(("ok    " if ok else "FAIL  ") + what)
        if not ok:
            failures.append(what)

    # ---- names: a logo's brand, and the names a phone or an app gives a file
    check(match.brand_words("AskEditor.png") == ("ask", "editor"), "a logo's brand from a joined-up name")
    check(match.brand_words("notion-logo-dark.svg") == ("notion",), "what describes a logo file is not its brand")
    check(match.brand_words("3M.png") == ("3m",), "a short brand with a figure is kept")
    for n in ("Screenshot 2026-10-28 at 10.12.31.png", "IMG_1234.PNG", "WhatsApp Image 2026-10-01 at 09.10.png",
              "CleanShot 2026-10-01 at 09.10.png"):
        check(match.name_words(n) == () and match.brand_words(n) == (), f"“{n}” names nothing")
    words, quiet, seconds = timed(TALK)
    said = match.said(words)
    check([said[i].text for i in match.said_whole(said, ("ask", "editor"))] == ["AskEditor"],
          "a brand lands on its whole name, never on “ask” alone")
    joined = [Word("Ask", 0.0, 0.2), Word("Editor's", 0.2, 0.5), Word("great.", 0.5, 0.8)]
    check(match.said_whole(joined, ("askeditor",)) == [0], "a brand said as two words, with an 's, is its name")

    # ---- which picture on which word
    pool = [picture("city.jpg"), picture("IMG_0001.JPG"), picture("IMG_0002.JPG"),
            picture("lighthouse.jpg"),  # never said: left out, saying why
            picture("DSC_9.JPG", tags=(("sea",),)),  # tagged
            picture("AskEditor.png", "graphic", (300, 300), logo=True)]
    shots, left, notes = P.decide(words, quiet, pool, {}, "keywords", seconds, 30.0)
    by_item = {s.item: s for s in shots}
    check(0 in by_item and by_item[0].word.lower() == "city", "the picture named for the city lands on “city”")
    check(4 in by_item and by_item[4].word.lower() == "sea", "the picture tagged sea lands on “sea”")
    check(5 in by_item and by_item[5].word == "AskEditor", "the logo lands on its brand's whole name")
    check(3 not in by_item and "never said" in left.get(3, ""), "a picture whose name is never said is left out")
    numbered = sorted((s for s in shots if s.item in (1, 2)), key=lambda s: s.start)
    check(len(numbered) == 2 and [s.item for s in numbered] == [1, 2] and all(s.why == "keyword" for s in numbered),
          "camera-numbered pictures take the strongest things named, in folder order")
    check(all(s.word.lower() not in ("sam", "best", "tiny", "calm") for s in shots),
          "a name, a describing word or a counting word never carries a picture")
    starts = {round(w.start, 3) for w in words}
    check(all(round(s.start, 3) in starts and s.frame == math.ceil(s.start * 30 - 1e-6) for s in shots),
          "each picture lands on the first frame of its word")
    gaps = [b.start - a.start for a, b in zip(shots, shots[1:])]
    check(all(g >= P.APART - 1e-6 for g in gaps), "no two pictures land closer than a quarter of a second")
    only_logo = [picture("AskEditor.png", "graphic", (300, 300), logo=True)]
    ask_only, left2, _ = P.decide(timed("Then just ask for it. An editor that edits for you.")[0],
                                  timed("Then just ask for it.")[1], only_logo, {}, "keywords", 6.0, 30.0)
    check(not ask_only and "never said" in left2.get(0, ""), "a logo is never laid on “ask” or on a thing named")
    mapped_words, mq, ms = timed("Then just ask for it. An editor that edits for you.")
    mapped, _, _ = P.decide(mapped_words, mq, only_logo, {"an editor": 0}, "keywords", ms, 30.0)
    check(len(mapped) == 1 and mapped[0].word.lower() == "editor" and mapped[0].why == "map",
          "map puts a logo on the words the person names")
    try:
        P.parse_map("town town.jpg")
        check(False, "a map pair without = is refused")
    except ValueError as e:
        check("town=town.jpg" in str(e), "a map pair without = is refused, saying how to write one")

    # ---- together, and gone together
    groups = P.group_shots(shots, "auto", "auto", words, quiet, seconds)
    check(all(len({s.leave for s in g.shots}) == 1 for g in groups), "the pictures of a group leave together")
    check(all(a.leave <= b.shots[0].land + 1e-6 for a, b in zip(groups, groups[1:])),
          "a group has gone by the time the next lands")
    fixed = P.group_shots(shots, "auto", "2.5", words, quiet, seconds)
    check(all(abs(g.leave - min(seconds, g.shots[-1].land + 2.5)) < 1e-6 or
              any(abs(g.leave - n.shots[0].land) < 1e-6 for n in fixed) for g in fixed),
          "hold=2.5: the pictures stay 2.5 s after the last lands, unless the next comes first")
    rotated = P.group_shots(shots, "rotate", "auto", words, quiet, seconds)
    check(all(len(g.shots) == 1 for g in rotated), "rotate shows one at a time")
    two = [P.Shot(0, 1.0, "a", "map", 3, frame=30, land=1.0), P.Shot(1, 6.0, "b", "map", 3, frame=180, land=6.0)]
    bridged = P.group_shots(two, "auto", "4.4", [], [], 9.0)
    check(len(bridged) == 2 and abs(bridged[0].leave - 6.0) < 1e-6,
          "pictures due to leave just before the next lands stay until it does")

    # ---- where: round heads drawn for the test, never over the face, inside the safe zone
    W, H = 1080, 1920

    def room_with(head: Rect, face: Rect) -> P.Room:
        return P.Room(W, H, [head] * 100, [face] * 100)

    wide_open = room_with(Rect(330, 820, 750, 1500), Rect(330, 950, 750, 1500))
    close_up = room_with(Rect(150, 380, 930, 1380), Rect(150, 520, 930, 1380))
    zone = wide_open.zone
    for name, room in (("room above the head", wide_open), ("a close-up", close_up)):
        head = room.heads[0]
        for n in (2, 3, 4):
            got = P.fit_arc(room, n, 0.8, head, P.ARC_CAP[n])
            if got:
                tiles = got[1]
                check(all(t.box(P.OVERSHOOT).inside(zone, 0.5) and not t.box(P.OVERSHOOT).intersects(head) for t in tiles),
                      f"{name}: an arc of {n} stays inside the safe zone and clear of the head")
            elif room is wide_open:
                check(False, f"{name}: an arc of {n} fits")
        row = P.fit_row(room, 3, 0.8, head, P.CAP["row"])
        if row:
            check(all(not t.box(P.OVERSHOOT).intersects(head) and t.box(P.OVERSHOOT).inside(zone, 0.5) for t in row[1]),
                  f"{name}: a row under the chin keeps clear of the face")
        single = P.fit_single(room, 1.0, head, P.CAP["rotate"])
        check(single is not None and not single[1].box(P.OVERSHOOT).intersects(head),
              f"{name}: one picture finds a place clear of the face")
        cl = P.fit_cluster(room, [1.0, 0.8, 1.25], head, P.CAP["cluster"])
        if cl:
            ts = cl[1]
            share = max((P.shared_area(a, b) / min(a.w * a.h, b.w * b.h) for i, a in enumerate(ts) for b in ts[i + 1:]),
                        default=0.0)
            check(share <= 0.17 + 1e-6 and all(not t.box(P.OVERSHOOT).intersects(head) for t in ts),
                  f"{name}: a cluster overlaps itself a little at most ({share:.0%}), never the face")
    arc = P.fit_arc(wide_open, 3, 0.8, wide_open.heads[0], P.ARC_CAP[3])
    check(arc is not None and arc[1][1].cy < arc[1][0].cy and abs(arc[1][0].angle) > 1,
          "an arc: the middle tile highest, the ends lower and turned out")
    check(arc is not None and min(math.sqrt(t.w * t.h) for t in arc[1]) >= P.READ * W,
          "an arc of three over a head with room above it reads on a phone")
    full = room_with(Rect(60, 150, 1020, 1800), Rect(60, 200, 1020, 1800))
    check(P.fit_single(full, 1.0, full.heads[0], P.CAP["rotate"]) is None, "a face that fills the frame leaves no room")
    a, b = P.Tile(0, 0, 100, 100), P.Tile(50, 0, 100, 100)
    check(abs(P.shared_area(a, b) - 5000) < 1e-6 and P.shared_area(a, P.Tile(150, 0, 100, 100)) == 0.0,
          "tiles are compared by their real shapes")

    # ---- framed, never squashed
    photo = picture("street.jpg", shape=(1200, 900))
    for aspect in (0.8, 1.0, 1.33):
        x, y, w, h = P.cover_window(photo, aspect)
        check(abs(w / h - aspect) < 1e-6 and x >= 0 and y >= 0 and x + w <= 900 + 1e-6 and y + h <= 1200 + 1e-6,
              f"a photo is cropped to the tile's shape ({aspect:g}), never squashed")
    screen = picture("dash.png", "screen", (900, 1440))
    screen.busy, screen.text_px = (0.4, 0.3), 15.0
    whole, part = P.screen_windows(screen, 1.25, 300, 1080)
    check(abs(whole[2] / whole[3] - 1.25) < 1e-6 and abs(part[2] / part[3] - 1.25) < 1e-6 and part[2] < whole[2],
          "a screenshot too small to read lands whole, then its readable part, both in the tile's shape")
    logo = picture("AskEditor.png", "graphic", (200, 500), logo=True)
    logo.rgb[60:140, 100:400] = 200
    logo.ground = (0, 0, 0)  # its own ground, as loading a logo finds it
    img, plate = P.graphic_image(logo, 300, 300)
    ink = np.argwhere(np.abs(img.astype(int) - np.array(plate)).max(axis=2) > 10)
    iw, ih = int(np.ptp(ink[:, 1])) + 1, int(np.ptp(ink[:, 0])) + 1
    check(abs((iw / ih) / (300 / 80) - 1) < 0.06, "a logo is shown whole, at its own shape (to a pixel at its edges)")

    # ---- the pop: there on its first frame, gone when it leaves, never past its planned box
    s = P.Shot(0, 2.0, "city", "name", 2, frame=60, land=2.0, leave=4.0)
    s.tile = P.Tile(540, 500, 300, 360)
    first = P.state(s, 2.0)
    check(first is not None and first[1] >= 0.7, "a picture is plainly there on the first frame of its word")
    check(P.state(s, 1.99) is None and P.state(s, 4.0) is None, "nothing before its word, nothing after it leaves")
    planned = s.tile.box(P.OVERSHOOT)
    inside = all(s.tile.box(st[0], dy=st[2]).inside(planned, 0.5)
                 for t in np.arange(2.0, 4.0, 1 / 60) if (st := P.state(s, float(t))) is not None)
    check(inside, "as it pops in and leaves, a picture never passes the box it was placed by")

    print(f"\n{len(shots)} pictures: " + "; ".join(f"{s.start:.2f} “{s.word}” → {pool[s.item].label} ({s.why})"
                                                 for s in shots))
    if failures:
        print(f"\n{len(failures)} failed")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
