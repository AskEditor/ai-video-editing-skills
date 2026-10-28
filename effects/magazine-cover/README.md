# Magazine cover

Your video opens as a magazine cover with you as the cover star. A big title sits behind your head,
so your head is in front of it as on a real cover, and short lines in your own words sit round you.
It holds for a few seconds, then lifts away and the video carries on.

```
/video-editing magazine-cover ~/Movies/my-take.mp4
```

Or just ask: "make me a magazine cover", "put me on a cover with my name as the title", "a classic
cover look for the first seconds".

## What goes on the cover

- **The title** (the masthead) is the thing your clip is about, taken from what you say: a short
  word you name, such as a topic or your own name, rather than a verb or a filler, and never a word
  the transcription was unsure of. When nothing you say is strong enough, the run asks you for one.
  Give your own with `title`.
- **The lines** are your strongest short phrases, as you said them. A number you say ("we now have
  2.5 million views") makes the biggest line, with the figure set large. The small lines start with
  a bold lead-in ("AN EDITOR", then "that edits for you"). Give your own with `lines`.
- **An issue line and a thin rule** sit above the title: an issue number and the month you filmed,
  which are decoration and claim nothing.

Nothing is made up. Every line is words you said or words you typed. A real magazine's name is
never used as a title, even when you ask for one.

## How it is placed

- **The title** is as wide as the part of the screen the apps leave clear. It sits low enough for
  your head to cover a little of its bottom edge, and never so low that a letter stops reading as
  itself (a hidden foot turns an E into an F). How much your head covers is measured on every frame
  the cover is on screen. When your head moves a lot, the title sits just above it.
- **The lines** go only where nothing of you covers them, with some air left round you. The big
  line goes wherever it can be largest (often above the title), and the small ones go down the side
  of your head. A close shot has less room, so it gets fewer lines, and the run tells you which ones
  fit.
- **Everything is printed on the wall behind you.** You stand in front of the whole cover, so no
  letter ever crosses your face, and your hair and shoulders keep their own edge.
- **The colours** are light or dark, whichever stands out from the wall behind each part. A busy
  wall gets a soft shade round the letters. The big line's key word takes one accent colour.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `bold` | `bold`: a tall condensed title with heavy lines. `classic`: a serif title, serif lines and a warm film look with grain while the cover is up |
| `title` | `auto` | your own title, one or two words |
| `lines` | `auto` | your own lines, the big one first, separated by `\|`. A `*word*` in stars takes the accent in the big line, and ends the bold lead-in in a small one. `off` leaves the title alone |
| `when` | `auto` | `auto` covers the opening. `whole` keeps it for the whole clip. A time such as `4.5`, or words you say such as `"the marketing campaign"`, brings the cover back later as a second hook |
| `color` | `auto` | the title's colour: light or dark, picked for you, or a name (`red`) or a hex colour |
| `issue` | `auto` | the issue line: an issue number and the month you filmed. Write your own (`"Issue 12 · Spring"`) or `off` |
| `frame` | `keep` | `9:16` reframes a wider video round your face |
| `fix` | | misheard words, written as `wrong=right; …` (a name above all, since it may become the title) |

## What it checks

`--still auto` draws four moments exactly as the video will show them. Every run also prints:

- the title, why it was chosen and the next best words, and each line with where it went;
- the frames where any letter would show over your face (should be 0) and outside the clear part of
  the screen (also 0);
- how much of the title your head covers, at most and for the most covered letter, and how much of
  any line is covered (0);
- the contrast each part reaches against the wall behind it;
- which lines did not fit, and why.

The plan it writes (`<video>.plan.json`) holds every decision: the words, sizes, places, colours
and timing. Edit any of them and run again with `--plan` to render exactly that.

## Honest limits

- It needs a person in the shot. On a tight close-up, most of the frame is you: the title, the
  issue line and perhaps one line fit, and the run says so.
- When your head moves a lot, the title sits just above it, so its letters stay readable.
- Words picked from speech are a careful guess. Read what the run printed, and give `title` or
  `lines` when you want others. English speech only.
- Names are often misheard. Check the title, and pass `fix` (`"Jon=John"`) before you render.
- One person is the cover star. Someone walking behind you counts as the wall.
- The cover is fixed to the screen while you move. The cut-out can lag a frame behind very fast
  movement at the edge of your hair.

## Keep editing it

The render is a finished video file. To retype the title or the lines, move them or change when the
cover appears without rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a
Magazine hook".
