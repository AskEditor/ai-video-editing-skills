# Retro overlay

Your video as if seen through something older: a camcorder's viewfinder, an editing timeline made
of your own shots and sound, a tape played back, or film with its burns. Everything is drawn in
code, and nothing sits on your face.

```
/video-editing retro-overlay my-talk.mp4
/video-editing retro-overlay my-talk.mp4 date="JUL 14 2026"
/video-editing retro-overlay my-talk.mp4 look=timeline
/video-editing retro-overlay my-talk.mp4 look=vhs captions=on
/video-editing retro-overlay my-talk.mp4 look=film strength=0.6 start="back in the day" end=6
```

## The looks

| `look` | What you see |
|---|---|
| `camcorder` (default) | The viewfinder's four corner marks, a red dot blinking beside REC, a battery, a timecode running with the clip (hours, minutes, seconds, frames) and the date. The picture is lifted and given a small camera's edge. |
| `timeline` | A frosted panel under your chin (above your head when there is no room) showing the clip as an editor does: a ruler, a video track whose clips are your video's own shots, cut where it cuts and filled with its own frames, your sound's own waveform, and a red playhead that moves with the real time. |
| `vhs` | Colour that bleeds sideways and splits at the edges, soft scanlines, a tracking band drifting up the picture, a tape's tone, PLAY and a tape counter. |
| `film` | A warm film tone, grain, a slow gate weave, specks of dust and a passing scratch, the rounded corners of a projected frame, and a film burn flaring in from a corner at the start and the end, with a soft light leak in between. |

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `camcorder` | `camcorder`, `timeline`, `vhs` or `film`. |
| `date` | `auto` | The date on the camcorder and the tape: `auto` (the date your file says it was recorded), a date as you want it shown, or `off`. |
| `strength` | `0.8` | How strong the look is: `0` leaves only the readouts, `1` is the most. |
| `start`, `end` | `0`, `end` | When the look comes on and goes: seconds, or words you say. It eases in and out over a quarter of a second. |
| `captions` | `off` | `on` lays the pack's captions over the look; the readouts and the panel make room for them. |
| `sound` | `on` | A quiet sound of the machine under your voice: a beep as REC starts, a tape deck's whirr, a projector's clatter, or a soft slide as the timeline rises. `off` leaves your sound untouched. |
| `fix` | (empty) | Misheard words, `wrong=right;…`, for `start`, `end` and the captions. |

## What it keeps

- **Your face.** The readouts sit in the top and bottom bands of the safe area and move to the
  other band, or are left out, where your head would reach them. The burns bloom from the edges
  and fade where your face is; dust and scratches land off it.
- **Your skin.** The tape's and the film's colour are made with the grade's own maths, which gives
  skin back most of its own colour.
- **Your sound,** with the machine's sound far under it.

## Honest limits

- **The date is only what is true.** `auto` shows the date the file carries; many exported or
  re-saved files carry none, and then no date is shown (type one with `date=`).
- **The timecode starts at zero** with your clip, as a camera's would when it starts recording.
- **One shot, one clip on the timeline.** A video with no cuts shows one long clip; the timeline
  reads best on a video you have already cut.
- **A close shot leaves little room.** On a face that fills the frame the timeline panel is drawn
  smaller, and a readout may be left out (the run says which).
- **VHS and film soften the picture** on purpose. On a busy or bright frame the effects read more
  quietly; raise `strength` if you want more.
