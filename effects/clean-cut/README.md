# Clean cut

Turns a raw take of you talking into a tight vertical video: the long silences, the retakes and
the false starts come out, captions go on, and the frame becomes 9:16 with your face in it.

```
/video-editing clean-cut ~/Movies/my-take.mp4
```

Or just ask: "cut the pauses out of this", "keep only my last take", "make this a clean reel".

## What it does

- **Shortens long pauses to a breath.** A silence longer than `pause` seconds keeps a little after
  the last word and a little before the next, so you never sound clipped. Shorter pauses are your
  rhythm, and stay.
- **Keeps your last take.** When you start a line and say it again within a few seconds, the
  earlier try comes out, whether you stopped halfway ("so the product is- so the product is really
  fast") or said the whole line twice. A list ("most ads, most people") or a word you double on
  purpose ("very very") is left alone.
- **Takes out um, uh and erm** (`fillers=keep` leaves them in).
- **Cuts the asides**: what you say about the recording rather than to the viewer. A mic check
  ("testing, testing"), asking for a moment ("give me a second"), starting again ("let me start
  again", "take two"), the equipment ("I've got the audio set up", "is this recording?") and talk to
  a crew ("are we rolling?"). Each sentence is judged whole, so "testing our product" or "cut the
  pauses" is never one, and a remark about something your video is about ("this audio works
  better", in a video about editing audio) is the talk, not an aside. Asides gather at the start
  and the end of a take, between pauses and next to each other, and those count too: words that
  could be said to the viewer as easily ("hold on", "cut") are only cut where they sit like an
  aside. `asides=keep` leaves them in.
- **Cuts only in silence, between words.** Each join is blended over 20 ms so nothing clicks.
- **Puts captions on**, in any look from [captions](../captions/README.md), clear of your face.
- **Frames it 9:16.** A wider video keeps its full height and the frame follows your face. A video
  that is already 9:16 is left as it is.

The pauses are found in the sound itself, not in the transcript: speech recognisers report almost
every word as touching the next. Where the transcript cannot explain a stretch of voice (one word
lasting far too long, or speech with no words), that stretch is transcribed again on its own, which
is how a false start the transcriber tidied away still gets found. Heard out of its sentence, a
second of sound is also where a transcriber makes words up, so what that second listen hears is
used only when there is voice under it, the transcriber is not unsure of it, it is not one of the
shapes its inventions take, and it is a word you say nearby. Anything else is kept aside as unsure:
never captioned, and never cut on.

## What it checks

- **Stills first.** `--still auto` shows the first line, a moment right after a cut and the last
  line, drawn exactly as the video will be.
- **The over-cut check.** After the render, the new video is transcribed again and compared word
  by word with the words that were meant to stay. A word lost or chopped at a cut is reported. A
  word heard differently next to a cut that was made in silence is listed separately: no sound of it
  was removed, so it is the transcriber hearing it another way.
- **The cut report** (`<name>.cuts.md` beside the video) lists every cut: where, how long, why, and
  what was said in it. Each aside is a cut of its own, saying what it is ("a mic check, in the
  opening"). The plan (`<name>.plan.json`) holds the same list; delete a cut from it and run again
  with `--plan` to keep that bit.

## Options

| Option | Default | What it does |
|---|---|---|
| `pause` | `0.35` | seconds of silence that count as a pause to shorten |
| `keep_retakes` | `off` | `on` keeps every take |
| `fillers` | `cut` | `keep` leaves um and uh in |
| `asides` | `cut` | `keep` leaves in the talk about the recording (mic checks, "give me a second") |
| `captions` | `on` | `off` for no captions |
| `look` | `stacked` | the caption look: `stacked`, `one-word`, `serif-chip` or `fitted` |
| `color` | `auto` | the caption accent: a name (`yellow`, `green`) or a hex colour |
| `position` | `auto` | `top` or `bottom` to choose; captions never cover the face either way |
| `max_words` | `0` | the most words on screen at once (`0`: the look decides) |
| `frame` | `9:16` | `keep` leaves the video's own shape |
| `size` | `auto` | `auto` never enlarges; `1080x1920` forces that size |
| `fix` | | misheard words, as `wrong=right; …` |

## Honest limits

- English only.
- A retake is found when it repeats your words. Starting the line over in different words ("I
  think- what I mean is") is not a retake it can see.
- An aside is found when its whole sentence is one. Said in the same breath as the talk ("give me
  a second, so the product is…"), or longer than twelve words, it stays; so does one said only in
  the middle of the take with no pause around it, and one whose words could just as well open a
  pitch ("hold on a second", "one second"), unless it sits among other asides.
- Jump cuts show: every cut moves the picture a little. Captions and a moving frame hide most of it.
- When the talking is fast and has no long pauses, there is little to cut; the cut report shows how little.
- One face. With two people talking, the frame follows the larger face.

## Keep editing it

The render is a flat video file. To change a cut or a caption without rendering again, open it in
[AskEditor](https://askeditor.com) and ask "Clean up my talking video": every cut stays on the
timeline to nudge or undo, and the captions stay words you can move and retype.
