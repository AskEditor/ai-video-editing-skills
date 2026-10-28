# Glitch

For a split second the picture breaks like a bad signal: the colours pull apart, slices of the frame
shove sideways, blocks turn to mush and lines of static run across. Then it is whole again. It lands
on the word that matters, or on a cut. Add a bar and "Uninstalling doubt…" fills up first and breaks
with the frame.

```
/video-editing glitch ~/Movies/my-video.mp4
```

Or just ask: "glitch the frame on the word that matters", "a glitch on every cut", "add an
uninstalling bar that glitches out".

## Where and when

- **On its own** it picks the strongest word you say, the same way every effect in the pack picks
  one: never a word you took back, a filler or an aside. A long video gets about one glitch every
  twelve seconds, at least six seconds apart, three at most.
- `at="production"` puts it on words you say, `at=4.2` on a second, `at=cuts` on each change of
  shot in the clip. Several go with `;` between them.
- The worst frame is the first frame of the word, or the first frame of the new shot. The break
  builds over a few frames before it and dies away after it.

It is short on purpose: 0.24 s (`light`), 0.36 s (`medium`) or 0.52 s (`heavy`), never more than
0.6 s, so it never sits on your voice for long. The frames before and after are untouched, and over
your face the break is a third as strong as elsewhere, so your face still reads during it.

## The bar

| `bar` | What you see |
|---|---|
| `off` (default) | The break alone. |
| `uninstall` | A frosted panel above your head: "Uninstalling *your word*…", a red bar filling in uneven steps to 100% on the break. It is torn with the frame and is gone when the frame comes back. |
| `progress` | The same panel with "Loading *your word*…" and a blue bar. After the break it says "Done" for a moment, then goes. |

The panel goes above your head, else under your chin, inside the part of the screen the apps leave
clear, and never over your face. The percentage is set in a mono face so it never wobbles as it counts.

## Options

| Option | Default | What it does |
|---|---|---|
| `at` | `auto` | the strongest word; or words you say, a time in seconds, or `cuts` |
| `strength` | `medium` | `light`, `medium` or `heavy`: how long and how broken |
| `bar` | `off` | `off`, `uninstall` or `progress` |
| `sound` | `on` | a digital crackle under your voice on the break (and a soft tick as the bar appears); `off` keeps your sound exactly |

## What it checks

`--still auto` draws the frames around each break exactly as the video will. Every run prints each
glitch with its word, its worst frame, how long it lasts and how much of that is over speech, the
clean frames either side, and two counts that should be 0: frames with the bar over your face, and
frames with the bar outside the safe area. The plan file lists every glitch; move one there and pass
it back with `--plan`.

## Honest limits

- English only for finding the word; a time or `cuts` works in any language.
- The whole frame breaks, including the parts the app's buttons cover: that is the look, but it
  means a check for things in the bottom fifth will notice it for those few frames.
- `cuts` finds changes of shot with a person in both; a cut between two empty shots is not seen.
- The bar's words are made from the word glitched on ("Uninstalling production…"); a long word is
  shortened, and the words cannot be typed yet.
- One person's face is protected.

## Keep editing it

The render is a finished video file. To move the glitch to another word or change its words without
rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a Glitch hook on my key word".
