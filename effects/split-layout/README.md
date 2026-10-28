# Split layout

You share the frame with what you are talking about: your own B-roll, a picture, a screen
recording or a line of text. You sit in a rounded panel at the bottom with your content above, in
one half of a split screen, beside it, or small in a corner over it, and the layout holds through a
section of your film before it changes, on your clip's own cuts or between sentences.

```
/video-editing split-layout my-talk.mp4 content=~/Movies/b-roll/
/video-editing split-layout my-talk.mp4 layout=split content="walk.mp4 | view.jpg | text: Three *simple* steps"
/video-editing split-layout my-talk.mp4 layout="panel, split, pip" switch=cuts full=between captions=seam
```

Or just ask: "put me in a panel at the bottom with my B-roll on top", "split screen me and my
footage", "put my screen recording above me", "switch the layout on every cut".

## What it does

1. **Cuts your film into sections.** The layout can change where your clip already cuts from one
   shot to the next (found from the picture: a hand over the lens is not a cut), or between
   sentences, in sections of about four seconds and never under two and a half, or at the seconds
   you give. Each item of `content` takes the sections in turn; with more sections than items an
   item holds across several, and a clip carries on where it left off.
2. **Keeps your face in your panel.** Your panel shows a window of your picture that always holds
   your forehead and your chin, and your hair when there is room, with your face across the middle
   and a little above its centre. The window follows you like a calm camera operator: still while
   you stay near, easing after you when you move, and jumping with your clip at its own cuts. It
   never runs past the edge of your picture, so there are never black bars. When your shot is so
   close that your whole head will not fit, your panel is made taller until it does, and when your
   panel is at the bottom it is made taller until your face sits above the bottom fifth of the
   frame, where the app's caption and buttons go. The run says when it did either.
3. **Fills the content panel well.** A clip of yours fills its panel, cropped about its middle,
   unless that would throw away most of it (a wide clip in a tall panel): then it is shown whole over
   a blurred copy of itself. Black bars in it are taken off first. A clip shorter than its time on
   screen plays again through a soft dissolve rather than freezing. A picture fills its panel the
   same way and moves slowly closer at one steady rate. A screen recording or a screenshot is never
   cropped or moved: it is scaled once, sharply, and shown whole as a window on a card, with its
   panel shaped to it and kept clear of the app's bar at the top. It is told apart from footage by
   its flat colours; `kind` in the list overrides that.
4. **Sets a line of text on a card** in heavy type, as large as the card allows, with one word on a
   coloured mark (the one you put between asterisks, or the most specific one). A line you give
   rises in as its section starts. With no content at all, each section shows the strongest line
   you say in it: on the card from the start, faint, each word darkening as you say it.
5. **Changes on the moment.** A section's layout starts to move on its first frame (the cut, or the
   first word of the sentence) and settles in under half a second. `slide` pushes the content in
   from its edge while your panel moves to its place, as if it made room for it; `scale` grows and
   shrinks the panels into place; `cut` changes on the frame. A soft rush of air sits under each
   move, well under your voice.
6. **Lets the pack's other effects in.** `over` lays an effect over the whole frame, and it keeps
   clear of your face where the layout has put it, not where it was in your clip. `inside` lays
   one inside the content panel, keeping to the part of that panel the app leaves clear: a chart
   over your B-roll, a headline on a card. `captions` puts the pack's captions on the seam between
   you and your content, in your panel under your face, or anywhere clear of your face.

Your sound is kept as it is; your content's own sound is not used.

## The layouts

| `layout` | What you see | Where you sit (`corner`) |
|---|---|---|
| `panel` (default) | two rounded panels on a plain ground, equal gutters all round | at the bottom (`top` puts you above) |
| `split` | two halves edge to edge | on top in a tall film (`bottom`), on the right in a wide one (`left`) |
| `side` | two rounded panels side by side; made for wide films | on the right (`left`) |
| `pip` | your content fills the frame, you sit in a small rounded panel over it | a lower corner clear of the app's buttons, or any corner |

Between sections you can fill the frame alone (`full`): at the start only (the default, so your
opening line is yours), between every section, at the start and the end, or never. Several layouts
(`layout="panel, split, pip"`) take turns, one a section.

## What it can show

| You give | It shows |
|---|---|
| a folder | every video and picture in it, in name order |
| files separated by `\|` | each in turn; `walk.mp4@4.2-9` gives it a time |
| `text: Your line` | the line on a card; `*word*` marks the word |
| a `.json` list | items as `{"path": …, "from": …, "to": …}` or `{"text": …}`, where `from` and `to` are seconds or words you say, plus `at` (where in the clip to start), `fit` (`fill` or `whole`), `kind` (`video`, `picture`, `screen`) and `layout` |
| nothing | the strongest line you say in each section, on a card |

## Options

| Option | Default | What it does |
|---|---|---|
| `layout` | `panel` | `panel`, `split`, `side`, `pip`, or several separated by commas, in turn. |
| `content` | (empty) | What shares the frame with you (see above). Empty: your own lines on cards. |
| `switch` | `auto` | When the layout changes: `auto` (your clip's own cuts, else between sentences), `cuts`, `words`, `times` (the times in `content`), or seconds such as `4.2, 9, 15`. |
| `full` | `first` | When you fill the frame alone: `first`, `between`, `ends` or `none`. |
| `ratio` | `0` | Your share of the frame, such as `0.55`. `0` lets the layout decide (about half; a third for `pip`) and grow it for your shot. |
| `corner` | `auto` | Where you sit: `top`, `bottom`, `left`, `right`, or a corner for `pip`. |
| `transition` | `slide` | `slide`, `scale` or `cut`. |
| `style` | `dark` | The ground round the panels: `dark`, `light`, or `blur` (your own picture, blurred and dimmed). |
| `border` | `off` | `on` draws a thin line round each panel. |
| `captions` | `off` | `seam`, `panel` or `frame` (see above). |
| `inside` | (empty) | Effects of the pack inside the content panel, separated by `\|`, each with its own options: `inside="chart values=2K,197K"`. |
| `over` | (empty) | Effects of the pack over the whole frame: `over="headline style=serif"`. |
| `sound` | `on` | `off` keeps your clip's sound exactly, with nothing laid under the moves. |
| `frame` | `keep` | Your clip's own shape, or `9:16` or `16:9`. Your panel is cut from your full picture, so a wide clip makes a sharp vertical panel. |
| `size` | `auto` | The output size; never enlarged unless you ask. |
| `fix` | (empty) | Misheard words, as `wrong=right`, so `from="…"` can find them. |

Every run writes a plan beside the video (`<video>-split-layout.plan.json`): each section's times,
layout, your share of the frame and what it shows. Change a section's layout or its times there and
run again with `--plan` to get exactly that.

## Honest limits

- **A very close shot fills its panel.** When your head takes up half the height of your picture,
  your panel shows little more than your head, and at the bottom of the frame your panel takes up
  to two thirds of it so your face stays above the app's caption. On the closest test clip (C) even
  that is not quite enough for a few seconds, and the run says so: `corner=top` or `layout=split`
  keep your face higher.
- **A wide screen recording in a tall panel is small.** It is sharp but its words may be hard to
  read on a phone; crop it to the part that matters first, or use `layout=side` on a wide film.
- **Effects that repaint the whole picture** (a colour look, a before and after) cannot go inside a
  layout: run them on your clip first, then split layout on the result.
- **Captions in your panel** need room under your face; on a close shot they sit on the seam
  instead, and the run says so. A tier list fits better over the whole frame than inside a panel,
  where it comes out small, and inside a layout its finished board does not hold at the end.
- **Cuts that change almost nothing** (you barely moved between two takes) are not found: on the
  test clip with ten joins it finds nine. Give `switch` the seconds for the one it misses.
- **It follows one face.** A second person in your clip is not kept in your panel.
- **English only** for switching between sentences and for spoken lines, like the rest of the pack.
- **Telling screens from footage** reads how flat the colours are: a very plain shot (a white wall)
  could be taken for a screen and a busy screen for footage. `kind` in a `.json` list settles it.
- **A picture whose edges are truly black** (a night sky) can have them taken for bars and cropped.

## In AskEditor

There the same layout is your clip and your content placed on the timeline, so you can drag a
panel's edge, swap a clip or move the moment the layout changes on the canvas, without rendering
again.
