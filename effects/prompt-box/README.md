# Prompt box

A prompt types itself over your clip, and the edit answers it on the same beat: "make it moody"
is sent and the colour turns, "captions on" is sent and the captions start, "big title" is sent
and the title slams in. Several prompts in a row build the edit up in front of the viewer.

```
/video-editing prompt-box ~/Movies/my-take.mp4 text="make it moody | captions on | big title"
```

Or just ask: "show me typing make it moody, and grade it as it's sent", "a prompt box that turns
on the captions", "type each edit on screen as it happens".

## The answer is another effect

Each prompt is answered by another effect of this pack, with its own options. Name it with
`answer`, in the same order as the prompts:

```
text="make it moody | captions on | big title" answer="grade look=moody | captions | headline style=serif"
```

Left as `auto`, the answer is read from the prompt's own words: "make it black and white" is the
grade's black-and-white look, "captions on" is captions, "big title" is a headline, "words behind
me" is a word behind your head, "add a follow button" is the follow card. The run prints what it
chose and why. Any effect of the pack can answer, including ones added later, as long as it keeps
the clip's length and shape (so `clean-cut` cannot: run it first, then the prompt box on its
result).

## When it is sent

`when` is the moment each prompt is sent, which is the moment its answer lands:

- `auto` (the default): the start of a spoken line, with time before it to type the prompt. Several
  prompts are spread across the clip, each box gone before the next one types. Near its share of the
  clip, each takes the start of the strongest line (a sentence's start before a clause's), never a
  line you take back or a first attempt you say again.
- seconds: `when=4.2`.
- words you say: `when="moody"` sends it as you say "moody", from the word times (the transcriber
  marks words a little late; the pack corrects for that and checks it against the sound).

The answer is told that moment through its own options, and every run measures the first frame
the answer changes. The check line says `first change on the send frame` when the two agree; when an
answer waits for a moment of its own (a headline that lands on the next words you say), the run
says how late it lands.

`until` ends an answer: `auto` keeps it for as long as the answer itself lasts (a look or captions to
the end, a headline for its line), so the edit builds up; `next` ends it as the next prompt is
sent; or give seconds or words.

## The box

Our own composer bar, never a copy of an app's: a placeholder and a blinking caret, then the prompt
typed at a person's pace (quick runs inside a word, a longer reach between words, a beat before
sending). The send button lights once there is text, is pressed, and turns to a tick. The bar
pulses as it is sent and a ripple leaves it; when the answer changes the whole picture (a grade),
a soft flash lands with it.

| Style | What it looks like |
|---|---|
| `light` (the default) | a white bar with dark type |
| `dark` | a near-black bar with white type |
| `glass` | frosted glass over the picture, smoked just enough that the white type reads on what shows through |

## Where it goes

Above the head when there is room, else below the chin, always inside the area the app's own
buttons leave free, and clear of your whole head (hair included) over every frame it is on screen.
Every box in a clip keeps the same place where it can, so it reads as one composer.

It also keeps clear of the other graphics. When there is no spot clear of them, an earlier graphic
(captions from an earlier prompt, say) steps aside while the box is up and comes back after. When a
prompt's own answer lands right where the box sits, the box dissolves under it as it lands, so the
answer is seen whole. A graphic that lands where an earlier one stands takes its place while it
shows.

## Options

| Option | Default | What it does |
|---|---|---|
| `text` | `make it moody` | what is typed; several prompts separated by `\|` |
| `answer` | `auto` | the effect that answers each prompt, with its options; `auto` reads it from the prompt |
| `when` | `auto` | when each prompt is sent: `auto`, seconds, or words you say |
| `until` | `auto` | when each answer ends: `auto`, `end`, `next`, seconds, or words you say |
| `style` | `light` | `light`, `dark` or `glass` |
| `position` | `auto` | `auto`, `top` or `bottom` (never over the face, whichever you pick) |
| `speed` | `1` | how fast it types: `slow`, `normal`, `fast`, or a number from 0.5 to 2.5 |
| `sound` | `on` | soft key ticks and a send tick, well under your voice; `off` keeps your sound exactly |
| `beat` | `auto` | what ties the send to the answer: `auto`, `ripple`, `flash`, `both` or `off` |
| `accent` | `auto` | the send button and the ripple: `auto` (yellow), a colour name, or a hex colour |
| `fix` | | misheard words as `wrong=right`, for `when` and `until` words; passed on to the answers |

`|` inside quotes stays part of an answer's own option: `answer='headline text="Edit less. | Post
more."'`.

## What it checks

`--still auto` draws a few frames exactly as the video will show them. Every run prints, for each
prompt, when it was sent and why, what answered it, and where the answer's first change landed;
the frames where the box or its ripple would touch your head (0) or leave the safe area (0); which
graphics stepped aside and for how long; and the level of the key ticks and the send against your
voice, measured on the sound laid in alone.

## Honest limits

- One line a prompt, about 34 letters. A long prompt types faster rather than taking longer.
- An answer that changes the clip's length or shape cannot be laid in: cut and frame first.
- `auto` takes the strongest line near each prompt's share of the clip, judged by its words: it
  cannot tell a remark about the recording ("give me a second") from the point, and where no strong
  line is near (the opening of D) it takes the nearest start. For a headline, say which words with
  `when` (or give the headline its `text`).
- Where the band above or below the head is crowded, earlier captions disappear for a second or two
  while the box is up.
- A face that fills the frame leaves no room: the box goes into the margin, smaller, and the run
  says so.
- The look answers (`grade`, `before-after`) draw from the clip as shot, so a second look replaces
  the first rather than going on top of it.
- Word times come from an English transcriber.

## Keep editing it

The render is a flat video file. To retype a prompt, move the moment it is sent or swap its answer,
open it in [AskEditor](https://askeditor.com) and ask "Type my ask on screen in a prompt box, and
make the edit as it is sent"; every prompt and every answer stays editable on the canvas there,
without rendering again.
