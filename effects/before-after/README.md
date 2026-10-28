# Before and after

One divider wipes across your video while it plays: the clip as you shot it on one side, and on the
other a colour look or your finished edit, with BEFORE and AFTER labels that never sit on your face.

```
/video-editing before-after ~/Movies/raw-take.mp4 after=~/Movies/my-edit.mp4
```

Or just ask: "before and after of the grade", "wipe from my raw take to the edit", "a slider that
swipes across me in black and white".

## What goes on the after side

- **A look:** `after=moody` (the default), `black-and-white`, `warm`, `cool`, `pop` or `film`, the
  same six as the `grade` effect, with their own vignette and grain. Skin stays natural in all six,
  and the run prints the skin check. For a look at another strength, run `grade` first and pass its
  file as the after.
- **Your edit:** `after=` the path of another video of the same take, such as the version with
  captions and music. It is played at your clip's frame rate and fills the frame without stretching.
  The run listens to both and lines them up by their sound, so an export that starts a few frames
  late is put back in step; it says what it found. An edit that cuts out pauses no longer lines up
  with the take after its first cut, so pass a version that keeps the take's timing.

## The wipe

| Mode | What it does |
|---|---|
| `auto` (the default) | `hold` for a look, `sweep` for your edit: held still, a split would cut the edit's own captions in half |
| `hold` | the divider comes in and stops halfway; the frame stays split |
| `sweep` | the divider comes in and swings back and forth across you, slowing at each end like a hand on a slider |
| `reveal` | the divider goes all the way, and the after takes the whole frame |

The divider is a thin white line with a soft dark edge either side, so it reads on a white wall as
well as a dark room, and it moves smoothly rather than a pixel at a time.

## The labels

BEFORE sits on the clip as shot and AFTER on the after, so the divider uncovers and covers each one
with its own picture. They are kept:

- clear of your head over every frame of the video, not only the first;
- inside the part of the screen the apps never cover, away from its edges;
- off your edit's own captions and titles: the run finds where the edit differs sharply from the
  take and keeps the labels away from there.

They start at the top of the frame; when your head or your captions are there, they move to the
foot, or to the first line down the frame where both fit. When nothing fits they get a little
smaller, and if nothing fits at all the video goes without them and the run says so, rather than
putting a label over your face.

## Options

| Option | Default | What it does |
|---|---|---|
| `after` | `moody` | a look, or the path of a video |
| `mode` | `auto` | `auto`, `hold`, `sweep` or `reveal` |
| `direction` | `left` | which way the divider travels: `left` (the after comes in from the right, so BEFORE is on the left once split), `right`, `up` or `down` |
| `at` | `1` | when the wipe starts: seconds, or words you say (`at="here it is"`) |
| `duration` | `auto` | how long the divider takes to cross: `auto` (1 second; 2 seconds a swing for `sweep`) or seconds |
| `labels` | `on` | `on`, `off`, or your own two words with a slash between them: `labels=RAW/EDIT` |
| `fix` | | misheard words as `wrong=right` (only matters when `at` is words) |

The sound is your clip's own.

## What it checks

`--still auto` draws four moments of the wipe exactly as the video will show them. Every run counts
the frames where a label touches your face's box and where one leaves the safe area (both should be
0), and prints how the two videos were lined up or how the look treated your skin.

## Honest limits

- The edit must keep the take's timing (same start, nothing cut out); a lag of a few frames is fixed,
  a recut is not.
- An edit that zooms or reframes the picture changes too much of it for the labels to find its
  captions; they then avoid only your face.
- One divider at a time, straight; no light-leak or angled wipe yet.
- With a close-up that fills the frame and captions across the rest, the labels may be left out.

## Keep editing it

The render is a flat video file. To move the divider, retime the wipe or change the look without
rendering again, open it in [AskEditor](https://askeditor.com) and ask "Show a before and after of
my edit".
