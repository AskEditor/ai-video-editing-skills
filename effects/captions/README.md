# Captions

Captions timed to every word you say, in four looks, placed where they never cover your face.

```
/video-editing captions ~/Movies/my-take.mp4 look=serif-chip
```

Or just ask: "add captions", "subtitle this", "one word at a time, in green".

## The four looks

| Look | What it is |
|---|---|
| `stacked` (the default) | two or three lines; the most important word larger, in the accent colour |
| `one-word` | one big word at a time; the important ones in the accent colour |
| `serif-chip` | serif words in a dark rounded box; the word being said lights up |
| `fitted` | one big accent word, the words around it set as wide as it is, as one block |

Words appear as they are said. Each phrase stays a moment after its last word, and never flashes by.

## How it decides

- **Which words go together.** Each sentence is cut where a reader expects: after a comma or a
  real pause in your voice first, then before a word that starts a new clause ("and", "because",
  "which", "once"), then before one that starts a phrase ("to", "with", "the"). A pause you make
  while finding the next word ("the… product") is not a cut. A line never ends on a word that leans
  into the next ("is", "of", "the", "and", "my") in the middle of a sentence, never splits what is
  said as one ("right now", "kind of", a first and last name, a number and its unit), and lines
  stay about the same length. One word is alone on screen only when you said it alone, or it is
  the point ("Distribution | is a real big issue").
- **Which word is the key word.** The most specific word in the phrase: a long word that is not
  grammar or filler, a number, a name, an acronym. When nothing is specific enough, no word is
  highlighted rather than a vague one. The words around it stay at least about 60% of its height.
- **Where it goes.** Below your face at chest level, or in the lower third, or above your head when
  your face is low in the frame; always inside the part of the screen the apps never cover (between
  x = 120 and 960 on a 1080-wide video, and above the bottom fifth). The place is checked against
  where your head is over every frame the phrase is on screen, so turning or leaning never brings
  your face under the words. The video gets one steady place; a phrase moves only when it has to.
- **What it leaves out.** Um, uh and erm are heard but never shown.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `stacked` | `stacked`, `one-word`, `serif-chip` or `fitted` |
| `color` | `auto` | the accent: a name (`yellow`, `green`, `pink`) or a hex colour such as `#ffd400` |
| `position` | `auto` | `top` or `bottom` to choose; the face is avoided either way |
| `max_words` | `0` | the most words on screen at once, never passed (`0`: the look decides, and may show one word more to keep a phrase whole) |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `size` | `auto` | `auto` never enlarges; `1080x1920` forces that size |
| `fix` | | misheard words, as `wrong=right; …` (names are the usual ones) |

## What it checks

`--still auto` draws four moments exactly as the video will show them. Every run also counts the
frames where a caption touches the face's box (it should be 0) and where one leaves the safe area
(also 0), and prints both.

## Honest limits

- English only. Names and brand words are often misheard; read the words first and pass `fix`.
- Only words the transcriber is sure of are shown. A stretch it hears a second time, on its own,
  can come back with words nobody said; those are kept aside as unsure and never shown.
- One face is followed. Someone else walking through the frame is ignored.
- When the face fills almost the whole frame, the band beside it is short: a phrase that would
  need squeezing is set flatter instead (on one line, or two even lines). Only when even that does
  not fit do the captions get smaller, and if nothing fits they sit in the margin and the run says so.

## Keep editing it

The render is a flat video file. To restyle, move or retype a caption without rendering again,
open it in [AskEditor](https://askeditor.com) and ask "Add captions to my video".
