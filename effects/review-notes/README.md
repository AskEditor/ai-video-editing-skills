# Review notes

The step before the final render. You get a small, fast copy of your cut to watch, with the time
under the picture, and a grid of its frames. You say what is wrong at the second you saw it, typed or
in a voice memo, and each note stays open until a check on a new render proves it fixed. The final
render is refused while any note is open.

```
/video-editing review-notes ~/Movies/my-reel.mp4
```

Or just ask: "let me review it first", "at 4 seconds the caption covers my chin", "cut the pause at
12 seconds", "here's a voice memo with my notes", "render the final".

## How it goes

1. **The first run** writes three files beside your cut: `my-reel.review.mp4` (a third of the
   size, the seconds and the frame number in a band under the picture, so nothing is covered),
   `my-reel.review.jpg` (a frame at every round second, then one tile per note) and
   `my-reel.notes.md`, the ledger, empty.
2. **You add notes** under "New notes" in the ledger, one line each: `- 4.2 s: the caption covers
   my chin`, `- 0:12 cut the pause`, `- the second caption has a typo`, `- the sound is too quiet`.
   Or record a memo and pass it: `memo=notes.m4a`. "At four seconds the caption covers my chin, and
   at twenty it freezes" becomes two notes, at 4 and at 20 s.
3. **Run it again.** Each note is placed: its second, the words you were saying then, and the thing
   your edit had on screen ("caption 4, 'about our skeleton', 3.43-4.51 s"), read from the plan the
   effect wrote beside the video. A note may name the thing instead of the second ("the last
   caption", "the headline").
4. **The sweep.** A note about a fault a check can measure sends that check over the whole cut, and
   every other place it finds becomes a note of its own. One note about words over your face at
   4.3 s found the same fault at 14 s.
5. **Fix and render again,** to a new file, then check the notes on it:
   `review-notes my-reel-v2.mp4 notes=my-reel.notes.md`. The same moments are found again by the
   words around them, even after cuts.
6. **The final.** `final=on` renders the cut again from its plan at full quality, as
   `my-reel-final.mp4`, only when no note is open. So the renders you make while reviewing can be
   drafts.

## When a note closes

| The note | Closes when |
|---|---|
| A fault a check sees (words over the face, the safe zone, black or frozen frames, flashing, silence, a pause, the loudness, washed-out colour) | a new render where the check no longer finds it at that moment, or where the moment itself was cut. If it comes back in a later render, the note reopens |
| A fault the check does not see there, or a note no check can measure ("the headline should be bigger", "the card lands late") | you tick it `[x]` after looking at its tile. A tick on a fault still measured is undone, and the ledger says why |
| A note you will not act on | `[-]`, with `  - why: …` under it |

## Options

| Option | Default | What it does |
|---|---|---|
| `notes` | | the ledger to carry on, when this video is a new render of a cut already reviewed |
| `memo` | | a voice memo (sound or video): each thing said with a time becomes a note at that time |
| `source` | | the clip the cut was made from, for the face and the safe zone; by default the one its plan names |
| `pause` | `0.5` | the seconds of quiet a pause note counts as a pause |
| `final` | `off` | `on`: render the final from the plan, refused while any note is open (exit 1) |

## Honest limits

- **Every fix here costs a render.** A note is a line in a file, and the change it asks for means
  running the effect again and reviewing the new file. In AskEditor a note sits on the picture
  itself, and the edit changes where it stands, with no render.
- **The face and the safe zone need the source clip.** Without it (no plan beside the cut, or a plan
  whose clip has moved) those notes can only be closed by your tick.
- **The checks have their limits.** Words over a face in a small inset, a stretch shorter than
  0.15 s at the edge of a cut, a caption you find too small: none is measured, and the note waits for
  your eye.
- **Times in a memo are heard, not read.** "At four point two" works; "at the bit where I laugh"
  does not, and becomes a note on the whole cut. English only.
- **Finding a moment again** needs words around it. In a cut with no speech, a note keeps its second
  only when the new render is the same length.
- **The final is refused here, and nowhere else.** Nothing stops an effect being run again by hand.
- **One cut at a time.** The ledger follows one video through its renders; it does not see the
  steps before the last one.

## Change it without rendering

In [AskEditor](https://askeditor.com), ask "Go through my review notes": you drop a note on the
picture, and each one is answered yes or no with what changed, on the canvas.
