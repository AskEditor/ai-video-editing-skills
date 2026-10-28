# Reframe

Turns a landscape or square video into a vertical one (or 4:5, square or 16:9) with your face kept
in the frame, the way a calm camera operator would hold it. Or makes every shape at once.

```
/video-editing reframe ~/Movies/my-podcast.mp4
/video-editing reframe ~/Movies/my-take.mp4 format=all
```

Or just ask: "make this vertical", "crop this for Reels", "make a square and a 4:5 version",
"follow whoever is talking".

## How the frame moves

- **It holds still** while your face stays near where it is. A nod, a lean or a gesture moves
  nothing: the frame waits until your face has really gone somewhere else.
- **Then it glides**, starting and stopping gently, and it starts a little before your head
  arrives rather than chasing it. Two moves the same way close together become one longer move.
- **Your face never leaves the frame.** If you lean out faster than a glide would follow, the
  frame is pushed along with you, and the check says how many frames that happened in.
- **Where your video already cuts, the frame cuts too**, instead of sliding across from one shot
  to the next.
- **The top of your head stays in**, with a tenth of the frame above it, whenever the new shape is
  shorter than the video (a vertical video made 4:5, or a square made 16:9). When your head is
  simply taller than the frame, it keeps your face whole, lets the top of your hair go, and says so.
- **Black bars filmed or exported into the video** are found and left out, so no frame shows a
  strip of black.

## Two people

When two people are in the picture, it follows whoever is talking: it watches each mouth and
listens to the sound, and the person whose mouth moves with the voice is the speaker. Each change
of speaker becomes a cut to the other person (or a glide, when they sit close), placed in a pause
between words. When both fit in the frame together, both are kept. `speaker=largest` follows the
bigger face instead.

## Every shape at once

`format=all` makes one file for each shape your video is not already: from a landscape video, a
9:16, a 4:5 and a 1:1. The first is the name you gave (or `<your video>-reframe.mp4`), the others
sit beside it as `…-reframe-4x5.mp4` and so on. Each one has its own frame path and its own checks.

## Options

| Option | Default | What it does |
|---|---|---|
| `format` | `9:16` | `9:16`, `4:5`, `1:1`, `16:9`, or `all` |
| `mode` | `track` | `track` follows your face · `center` holds a still crop from the middle · `fill` keeps the whole picture, fitted over a blurred, darkened copy of itself |
| `speaker` | `auto` | with two people: `auto` follows whoever is talking, `largest` the bigger face |
| `size` | `auto` | `auto` never enlarges the picture (a 1080p landscape video gives a 608×1080 vertical); `full` makes each shape its full size: 1080×1920, 1080×1350, 1080×1080 or 1920×1080 |

## What it checks

- **Stills first.** `--still auto` draws moments either side of the first moves and changes of
  speaker, exactly as the file will show them.
- **On every render it prints** the share of frames with your face wholly inside the frame, how
  often the top of your head is cut (when it can be), how still the frame is, how many glides and
  cuts it made, its fastest move, and how many times it turned back while moving (jitter would
  show there; it should be 0).
- **A path sheet** (`<file>.path.jpg`) shows your video with the frame drawn on it at six moments,
  and under it the whole path: where your face was, where the frame could be with your face
  inside it, and where it went.
- Run `qc-check` on the result with `source=` your original video to check the sound, the safe zone
  and the rest before you post.

## Honest limits

- **Your sound is kept as it is.** A quiet recording stays quiet: `qc-check` says so, and
  `music-bed` sets a mix to -14 LUFS.
- **It never enlarges by default**, so a 1080p landscape video makes a 608×1080 vertical one;
  `size=full` enlarges it, and it is softer.
- **A wide shot of a small face** (a stage, a room) stays wide: it reframes, but does not zoom in.
- **Telling who is talking** works from the mouths and the sound together, and it is the weakest
  part. It was tested on three side-by-side pairs made from our own takes, where both
  mouths move all the time and one voice is heard at a time (harder than a real listener, whose
  mouth rests; no real two-person take was there to test on). Two were right, each change of
  speaker found 0.3 s and 1.2 s late; the third, whose turns lasted only four seconds, was missed
  and followed one person throughout. Turns shorter than about six seconds are not told apart.
  The plan file lists the turns: change a turn's `person` and run again with `--plan` to fix one.
  With three or more people it follows the two seen most. When it cannot tell, it says so and
  follows the larger face.
- **A close shot made vertical** leaves no room at the sides: when the head is as wide as the
  frame, a lean keeps the face in but can cut the side of the hair.
- **Faces far from the camera** (smaller than about a twentieth of the frame) may not be found; the
  frame then stays in the middle and says so.

## Keep editing it

The render is a flat file. To change a move, who it follows or the shape without rendering again,
open the video in [AskEditor](https://askeditor.com) and ask "Make this vertical": the camera stays
keyframes on the timeline that you can drag.
