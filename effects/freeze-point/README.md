# Freeze and point

The picture stops on a moment, and a hand-drawn arrow and a label point at you: "that's me". A
record scratch lands on the freeze, the arrow draws itself in one stroke from the label to just
outside your face, the label writes itself on in marker, and after a beat the clip carries on from
exactly where it stopped. Instead of the arrow, a loop can be drawn round your head, or the held
frame can turn into a comic panel with the label in a caption box.

```
/video-editing freeze-point ~/Movies/my-take.mp4
/video-editing freeze-point ~/Movies/my-take.mp4 label="me, three years ago" at="this is me"
/video-editing freeze-point ~/Movies/my-take.mp4 mark=circle label="the editor" sound=hit
/video-editing freeze-point ~/Movies/my-take.mp4 mark=panel label="me, three hooks in"
/video-editing freeze-point ~/Movies/my-take.mp4 sound=voice hold=1.2
```

Or just ask: "freeze the frame and point at me", "yep that's me freeze frame", "freeze it and
circle my head", "freeze into a comic panel".

## What happens to the sound (and the length)

This is a choice, so it is yours, and the default is the usual one:

- **`sound=scratch`** (the default): your voice stops with the picture, a record scratch lands on
  the freeze, the room's own sound holds under the still picture (looped from the quietest stretch
  of your clip where nobody speaks, at the room's level, so the hold never plays as the sound
  dropping out), and the clip carries on from exactly where it stopped. Nothing
  you say is lost and picture and sound stay together, so **the video gets longer by the hold**
  (1.6 seconds by default). `sound=hit` is the same with a heavy hit; `sound=none` with the room alone.
- **`sound=voice`** keeps the video's length: your voice carries on under the still picture, and
  when the hold ends the picture jumps ahead to meet it. Your lips are still while your words go
  on, which reads as a deliberate pause-and-talk-over; use it when the video's length must not
  change (music under it, a fixed slot). Near the end of a clip the hold then lasts only to the
  end, since nothing comes after it.

The freeze always lands where your voice is quiet, never inside a word, so no word is cut in half
when the voice stops.

## How it decides

- **The moment.** Just after the word that makes the point of your opening line (the same word
  `words-behind` would choose), or after the words you name with `at`, or at a time (a time that
  falls inside a word moves to that word's end). Inside the pause after the word, it stops on the
  sharpest frame of your face, so the frame held is not a blur.
- **The label** goes in the free space beside your head on the held frame: on nobody, clear of
  your face and head, inside the part of the screen the apps leave clear, as large as reads well.
  For an arrow it goes where the arrow from it to you is a good length and crosses as little of
  you as it can; for a loop, beside the loop. It is written in marker, by hand, left to right.
- **The arrow** is one stroke: the shaft bows slightly the way a hand's would, away from your face,
  and the head is flicked on at the end of the same stroke. Its tip lands just outside your face,
  your forehead counted as face, so it points at you and never touches you: it may land on your
  hair or beside your head.
- **The loop** goes round your head from the upper right, once and a bit more, its end passing
  its start a little wide, as a quick circle does. It never crosses your face.
- **The comic panel** inks the held frame (flat tones, black lines on its edges, halftone dots in
  its shadows), shrinks it into a black frame on a page, tilts it a touch, and pops a caption box
  with your label into the corner furthest from your face.
- **The colour** of the marks, unless you give one, is the marker colour (yellow, white or red)
  that stands out most from the picture under them; a soft dark shadow under every mark keeps it
  readable on a bright wall too.
- **The look of the hold.** A white flash as the picture stops, the corners a touch darker while
  it holds, and the marks fading just before it moves again, so nothing is left floating over a
  moving picture.

## Options

| Option | Default | What it does |
|---|---|---|
| `at` | `auto` | when the picture stops: `auto` (just after the word that makes the point), `start`, a time such as `2.4`, or words you say (it stops as they end) |
| `hold` | `1.6` | how many seconds the picture holds still, from `0.5` to `6` |
| `label` | `auto` | the words that point at you, up to seven (`auto` writes "that's me"); `none` for no label |
| `mark` | `arrow` | `arrow`, `circle` or `panel` |
| `color` | `auto` | the marks' colour: a name (`yellow`, `white`, `red`…) or a hex colour such as `#ffd400` |
| `sound` | `scratch` | `scratch`, `hit` or `none` (your voice stops for the hold, and the video gets longer), or `voice` (your voice carries on and the length is kept) |
| `captions` | `off` | the pack's captions in a look (`stacked`, `one-word`, `serif-chip`, `fitted`); they wait while the picture holds, or with `sound=voice` carry on, the label and arrow kept clear of them |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `fix` | | misheard words, as `wrong=right; …` |

## What it checks

`--still auto` draws four moments of the hold (the flash, the marks drawing, all drawn, just before
it moves again) exactly as the video will show them. Every run also prints:

- the moment, why it was chosen, the frame held, and the words around it;
- what the sound does and how much longer the video is (or that its length is kept);
- the arrow's length and how far its tip lands from your face;
- the label, where it went, and the colour and why;
- `frames_over_face` (it should be 0): any frame where a mark, the label or a caption would touch
  your face; and `frames_outside_safe_zone`, for the label (also 0).

The plan it writes (`<video>-freeze-point.plan.json`) holds every decision. Change the moment
there and run again with `--plan` to freeze somewhere else, with the same look.

## Honest limits

- **It needs your face on the held frame:** a moment where you look away or leave the shot is
  refused, and says so.
- **A close shot leaves little free space**, so the label is smaller and the arrow shorter there.
- **Words run together** sometimes leave no pause after the word: the freeze then lands on the
  quietest instant between the two words, which can clip a breath, never a word.
- **The longer video** (with the voice stopped) no longer lines up with anything already timed to
  the original, such as music or an edit of it: add music after, or use `sound=voice`.
- **The marks stay where they are drawn**: the picture is still while they show, so they fit
  exactly, but they are not for a moving picture.
- **Nobody has listened to the scratch yet** (it is made in code and measured, not yet heard).

## Keep editing it

The render is a flat video file. To move the moment, retype the label or change the look without
rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a Freeze intro hook".
