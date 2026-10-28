# Checks before done

Looks over a finished video before you post it and tells you, in plain words, what would let it
down: black or frozen frames, sound that is too quiet, clipped or missing, words over your face,
anything in the part of the screen the app covers, and washed-out colour. It changes nothing.

```
/video-editing qc-check ~/Movies/my-reel.mp4 source=~/Movies/my-take.mp4
```

Or just ask: "check this before I post it", "is this ready for Reels", "is anything over my face".

Give it the video you made it from (`source=`) whenever you have it: that is what lets it measure
your face and the safe zone frame by frame.

## What it checks

| Check | How |
|---|---|
| Shape, size, frame rate, codec, length | from the file: 9:16 for Reels, TikTok and Shorts; under 720 on the short side is soft; a frame rate that changes, or one different from the source's |
| Loudness and true peak | measured as the pack itself sets loudness, against -14 LUFS and -1 dBTP: within 3 LU is right, 3 to 6 LU off is a warning, more fails |
| Clipping | samples that reach full scale |
| Silence | stretches of dead silence; with the source, the voice missing where the source has speech; long dead air in a talking video |
| Black frames | a frame almost wholly black; a black stretch in the middle fails, a fade at the start or end is a warning |
| Frozen frames | a picture that stops changing while the voice goes on, or while the source still moves, fails; a held last frame (an end card) is fine; a freeze the plan beside the video names, or one where the sound stops with the picture, is a freeze for effect and worth knowing |
| Flashing | more than three flashes in a second |
| Washed-out colour | HDR or log footage that was never converted: from the colour tags, and the brightness and colour of the picture against the source read both ways |
| Words over the face | with the source: frame by frame, inside your face's box, anything the video shows that the source does not |
| The safe zone | with the source: anything added outside it (x 120 to 960 on a 1080-wide frame, above the bottom fifth, below the top tenth). Words under the caption or the buttons (the bottom fifth, the right margin) for half a second fail; words under the top bar or in the left margin, and pictures reaching into a margin, are worth knowing |

**How the video is lined up with its source.** In time, by the sound: the same frames, a trim, or
cuts (each found where the sound jumps; the frames either side of a cut are not judged). In space:
the same frame, a crop that moves (as `reframe` makes), or the whole picture fitted inside a bigger
frame. The comparison is of shapes, not colours, so a grade, a black-and-white look or a new
background does not count as something over your face; words, cards and stickers do.

## What you get

- `<video>.qc.md`: the verdict first (PASS, or FAIL and what to fix first), then what is worth
  knowing, then every check with what was measured.
- `<video>.qc.jpg`: the frames that matter (each problem's worst moment first), with the safe zone
  dashed and your face's box in yellow, and anything over your face that is not in the source
  tinted red; under them a strip of the whole video: the picture, the face, the safe zone and the
  sound.
- It exits with 1 when something fails, so a script can stop on it. `strict=on` fails on warnings
  too.

## Options

| Option | Default | What it does |
|---|---|---|
| `source` | | the video it was made from: the face, the safe zone, missing sound and a changed frame rate are checked against it |
| `platform` | `reels` | `reels`, `tiktok`, `shorts` (9:16, -14 LUFS, -1 dBTP), `youtube` (16:9), or `any` (no shape or loudness target) |
| `strict` | `off` | `on`: a warning fails the check too |

## Honest limits

- **Without the source** it cannot tell your words from your picture, so the safe zone is not
  measured, and the face check only notices a face that vanishes for a moment.
- **Where the picture is moved into a new layout** (a split screen, a picture in a picture), it
  cannot line the video up with the source, says so, and skips the face and the safe zone there.
- **Cutaways.** Where the whole frame shows another picture (B-roll, the next clip), there is no
  source to measure against, so the face and the safe zone are not judged there; the report says
  what share of frames that was. Words over a face in a small inset during a cutaway are not seen.
- **Words or pictures?** A line of words is told from a picture by its shape: short and wide, and
  placed out in the margin rather than reaching into it from a bigger picture. A picture with words
  in it (a screenshot behind you) is reported as worth knowing, not as a fault. A thin line along
  the frame (a wipe's divider, a border) is drawn on purpose and is not counted.
- **The safe zone is Instagram's.** TikTok's and Shorts' buttons sit in about the same places.
- **It checks; it does not judge taste.** Whether the words land on the right beat is for you.

## Fix it without rendering again

To fix what it found without rendering again, open the edit in [AskEditor](https://askeditor.com)
and ask "Check my edit before I post it": it watches the edit as it will export, and each fix is a
move on the canvas.
