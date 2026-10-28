# New backdrop

You are cut out of the room you filmed in and set in a new one: a colour, your own pictures that
change as you name each idea, four pictures round you, or your screenshots floating in depth
behind your head. The room's light falls on you a little, you cast a soft shadow on it, and your
edge has no rim of the old room, so you look like you are in it rather than pasted on it.

```
/video-editing backdrop my-talk.mp4
/video-editing backdrop my-talk.mp4 color=sage
/video-editing backdrop my-talk.mp4 pictures=~/trip-photos
/video-editing backdrop my-talk.mp4 pictures="~/a.jpg, ~/b.jpg, ~/c.jpg" at="town, city, hotel"
/video-editing backdrop my-talk.mp4 mode=grid pictures=~/four-photos
/video-editing backdrop my-talk.mp4 mode=screens pictures=~/screenshots
```

Or just ask: "put me on a plain green background", "change the background to a picture of each
place I mention", "put my app's screenshots behind me", "a grid of my photos around me".

## The four rooms

| `mode` | What is behind you |
|---|---|
| `room` | One colour, lit softly from behind your head and deepening toward the edges (`gradient=flat` for one even colour, or a second colour for a run from top to bottom). `pattern=grid-paper` or `dots` adds a fine pattern. Several colours (`color="sky, blush, butter"`) change the room on each idea. |
| `pictures` | Each picture fills the room and gives way to the next on the next idea, drifting and pushing in slowly while it is up. `layout=card` shows each one instead as a framed print on the wall behind you. A clip works too: it plays, muted. |
| `grid` | Four pictures, two by two, with you in front of the middle. Each lands on its idea, so the grid fills one idea at a time; a fifth picture replaces the first. |
| `screens` | Screenshots floating behind your head at three depths, turned a little toward the middle, drifting a little with you as a real room would. One picture gets one large screen, two get one either side. |

`mode=auto` (the default) is a colour room, or `pictures` when you give pictures.

## When the pictures change

Each picture lands on the next idea you name. An idea is the thing a phrase names, found in your
words: "the lively town", "production quality", "the marketing campaign". A doing word is not an
idea, and neither is a word said again. A phrase lands on its first word, so the picture is there
while the idea is heard, and two changes are never closer than a second. With more pictures than
ideas, the extra pictures land in the middle of the longest stretches between them.

To choose yourself, give one moment per picture with `at`: words you say (`at="town, city,
hotel"`), or seconds (`at="0, 2.4, 6"`). The run prints each picture with the second it lands and
the words it lands on.

## How you sit in the room

- **The edge.** The cut-out is firmed up round its middle, so there is no faint haze of the old
  room past your edge and the new room never shows through the inside of your edge. A pixel on
  the edge is part you and part old room: only the old room's share is swapped for the new one.
  The edge is steadied against the frames either side wherever the picture did not change, so it
  holds still on a still shot and keeps up when you move.
- **Two looks.** Two ways of finding a person look at every frame, and only what both agree is
  you is kept. Each takes things for you that the other does not (in the tests, a vase against
  the hair, a window panel beside a face on a train), and both find your hands, your microphone
  and your hair. `cutout=quick` keeps one look: it finds you in about half the time, with more of
  the room stuck to you.
- **Only you.** Somebody walking past behind you, or a stray patch the cut-out took for you, is
  left out, as long as it is not touching you.
- **A covered lens.** When a hand covers the lens, nobody can be cut out and the frame holds almost
  no detail: those frames show as filmed, rather than an empty room where you were.
- **Light.** The room's light wraps a little way round your edge (`rim`), more from a bright room
  than a dark one. From about 0.6 it reads as a lit edge.
- **Shadow.** You cast a soft shadow on a colour room, on cards and on screens, falling away from
  the brighter side of your face (`shadow`). Over a full-frame picture of somewhere far away there
  is no wall to fall on, so the shadow only darkens the room close round you.
- **Colour.** Your colours lean toward the room's light (`match`), changing as the pictures
  change. Skin is guarded: the lean is capped, and the run prints how far your skin moved.
- **Grain.** A colour room is given the same fine grain as your clip, so it is no cleaner than
  you are, and it does not band.

The room colour, when you do not choose one, is the one that stands out most from the colours at
your own edge (your hair, your clothes), so your outline always reads.

## Options

| Option | Default | What it does |
|---|---|---|
| `mode` | `auto` | `room`, `pictures`, `grid` or `screens` |
| `pictures` | | a folder of pictures, or files separated by commas: photos, screenshots, or clips (they play muted) |
| `at` | `auto` | when each picture lands: `auto` (on each idea), words you say, or seconds, one per picture |
| `color` | `auto` | the room's colour: a name (`lavender`, `sage`, `sand`, `sky`, `blush`, `mint`, `butter`, `cream`, `coral`, `forest`, `navy`, `plum`, `slate`, `charcoal`, or `white`, `black` and the like) or a hex colour; several, separated by commas, change the room on each idea |
| `gradient` | `soft` | `soft` (lit from behind your head), `flat`, or a second colour for a run from top to bottom |
| `pattern` | `none` | `grid-paper` or `dots` on the room |
| `layout` | `auto` | pictures: `full` or `card`; screens: `trio`, `pair` or `single`; the grid has one layout |
| `transition` | `fade` | how one picture gives way to the next: `fade`, `cut` (on the word's own frame) or `slide` |
| `move` | `on` | `off` holds everything behind you still |
| `blur` | `auto` | photos a little soft, as a camera sees a room behind you; screens stay sharp; `off`, or 0 to 1 |
| `rim` | `auto` | the room's light round your edge: `off`, or 0 to 1 |
| `shadow` | `auto` | your shadow on the room: `off`, or 0 to 1 |
| `match` | `0.35` | how far your colours lean toward the room's light, from 0 to 1 |
| `cutout` | `careful` | `careful` keeps only what two looks agree is you; `quick` uses one, faster and with more stray pieces |
| `frame` | `keep` | `9:16` reframes a wider video round your face |
| `fix` | | misheard words, as `wrong=right; …`, so `at` can find them |

The grid's tiles, the card and the screens are placed from where you are over the whole clip: each
stays at least 40% in view if it can, then the larger size wins. You stand in front of their inner
corners on purpose; that is what puts them behind you.

## What it checks

`--still auto` draws the first frame and the moments just after the first changes, exactly as the
video will show them. Every run also prints:

- the room's colour and why it was chosen;
- every change, with the second it lands and the words it lands on;
- `frames_over_face`: frames where the new room would show inside your face because the cut-out
  missed part of it (it should be 0), and the most of the face it showed through on any frame
  checked;
- how much of each tile, card or screen stays in view;
- the rim, shadow and match, and how far your skin's colour moved.

The plan it writes (`<video>-backdrop.plan.json`) holds every decision: the mode and layout, the
room's colours, each picture with its moment, and the light. Change a moment or a picture there and
run again with `--plan` to get exactly that.

## Honest limits

- **The cut-out is only as good as what it can see.** With two looks, things in the room that one
  of them takes for part of you are left out (in the tests, a vase against the hair). What both
  take for you stays with you: a ceiling sprinkler touching the hair on some frames, the chair you
  sit in in pieces for a moment, and on a train a bag on the rack touching the top of a cap.
  Something apart from you is dropped.
- **A busy room that moves** (filmed on a train) is the hardest case: two looks cleared most of it
  (a window panel, a strap), but stray edges can still come and go next to your head. Film against
  something plainer for this effect.
- **Glasses:** what was behind you, seen through the lens, stays in the lens.
- **A close shot leaves little room.** Pictures in the grid, the card and the screens are mostly
  behind your head when your face fills the frame; the run says how much of each stays in view.
- **Fine hair against a plain wall comes out well; wisps finer than a few pixels do not.** The
  edge is firm, so the faintest strands are lost rather than haloed.
- **One person.** Two people standing together come out as one shape; a person beside you is cut
  out with you.
- **A covered lens** (a hand over it) shows as filmed for those frames, not in the new room.
- **Pictures are yours to give.** Nothing is fetched. A HEIC photo needs saving as JPEG first.
- **Speed.** Every pixel of every frame is redrawn, and the first run finds you twice: about a
  minute and a half for ten seconds of 1080×1920 on a recent laptop. `cutout=quick` is faster.

## In AskEditor

To swap a picture, move a moment or change the room without rendering again, open the video in
[AskEditor](https://askeditor.com) and ask "Cut me out and put my pictures behind me, one for each
idea": the cut-out and each picture stay separate layers on the canvas.
