# Checklist

Your steps appear beside your head as you say them, and each one is ticked off as you finish it: a
tick that draws itself, a spinner that turns into a tick, switches flipping on in a settings card, a
line struck through, or a bar that fills a step at a time. The list never covers your face.

```
/video-editing checklist ~/Movies/my-steps.mp4
```

Or just ask: "tick off the steps as I say them", "add a to-do list that checks off as I talk",
"flip a switch on for each thing I mention".

## Two ways to fill it

**Say it.** With nothing else given, the steps are found in what you say:

| You say | The steps |
|---|---|
| "First of all, I went to the train station… I then boarded the train… and then eventually I got off the train… I eventually went to my hotel." | Went to the train station · Boarded the train · Got off the train · Went to my hotel |
| "Step one, plug in the mic. Done. Step two, open the app. Check." | Plug in the mic · Open the app |
| "Drop your clip in, then type add captions, then pick a look." | Drop your clip in · Type add captions · Pick a look |
| "It picks your best line, cuts you out, and pops you right out of the frame." | Picks your best line · Cuts you out · Pops you right out of the frame |

The first row of that table is our own take from a hotel room, found with no help. Each
step is the words after "first", "then", "next" or "step two", without what only led into them ("I",
"you want to", "of course"). A "then" after an "if" is never a step.

**Or list them**, separated by `;` (or by commas when there is no `;`):

```
/video-editing checklist my-video.mp4 items="Plug in the mic; Open the app@3.1; Press record"
```

Each step appears where it is said. `@3.1` puts it on the word said nearest that second. A `.json`
file can also give a step the second it is ticked: `[{"text": "Plug in the mic", "done": 2.4}, …]`.

## When it appears, when it is ticked

- **Appears** on the first frame of its word, the word's start corrected for the transcriber's
  lateness; its short entrance runs in the frames before, so that frame shows it still and whole.
  The card grows a row at a time; the rows above never move.
- **Ticked** where you say it is done ("Done.", "Check.", "and that's it"), otherwise where the next
  step starts (its "then", "next" or "step three"), and the last one where its sentence ends. The
  tick starts on that frame with its sound, and never on the frame the next step lands: when the two
  would meet, the tick comes a beat earlier.

After a render the file itself is measured: the frame each row stops moving on, and where each sound
lands against its frame.

## The looks

| `look` | What you see |
|---|---|
| `tick` (default) | A ring before each step; when it is done the ring fills and a tick draws itself, and the step dims a little so the next one stands out. |
| `spinner` | A spinner turns beside the step you are on, then closes into a filled tick when it is done. |
| `switches` | A settings card of our own: every row is there from the start, the one you are on lights up, and its switch slides on (its knob carries a small tick) when it is done. |
| `strike` | Each step is struck through with a line drawn across it, and dims. |
| `progress` | Ticks, with a bar across the top that jumps a step's share on each tick and stands still in between, with a count beside it (2/4), never a made-up percentage. |

`style=glass` puts the card on frosted glass with your video blurred behind it, and `style=paper` on a
light card with a serif face.

## Where it sits

Where your head is not: above it, beside it, or under your chin, inside the part of the screen the
apps leave clear, as large as the room allows. A card that grows is checked frame by frame, so a
head that rises early, while the card holds one row, does not make every row small. If your hair
rises into the card for a moment, you stay in front of it; your face is never behind or under it.

A step too long for its row is made shorter by whole words, never cut inside a word, and the run says
which steps it shortened.

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | | the steps; empty finds them in your words |
| `look` | `tick` | `tick`, `spinner`, `switches`, `strike` or `progress` |
| `title` | | a heading across the top, such as "Before I post" |
| `position` | `auto` | `top`, `side` or `below` to choose; your face is never covered either way |
| `style` | `solid` | `solid` (a dark card), `glass` (frosted) or `paper` (a light card) |
| `accent` | `auto` | the colour of a done step, by name or hex (a green by default) |
| `hold` | `1.0` | seconds the finished list stays on screen at the end (`0` for none) |
| `leave` | `0` | seconds the list stays after its last tick before it goes; `0` keeps it to the end |
| `sound` | `on` | a soft tick as each step is done, a soft bell on the last, a click for a switch; `off` keeps your sound exactly |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `size` | `auto` | `auto` never enlarges; `1080x1920` forces that size |
| `fix` | | misheard words, as `wrong=right; …` |

## What it checks

`--still auto` draws three moments exactly as the video will show them. Every run prints each step
with the word it appears on and the second it is ticked and why, and three counts that should be 0:
frames with anything over your face (or over your head with nobody laid back over it), and frames
where anything leaves the safe area. The plan file lists every step with its times; change one there
and pass it back with `--plan`.

## Honest limits

- English only. Read the words first and pass `fix` for misheard names.
- Steps said with no markers ("we also…", "and there's…") are not found; give them with `items`.
- A step's words are the clause as said, so they read like speech ("Went to the train station"), not
  like an instruction. Give your own words with `items` for that.
- On a tight close-up the card goes under the chin and is small; the run says when its words are
  below an easy reading size.
- One person is followed.

## Keep editing it

The render is a finished video file. To retype a step or move its tick without rendering again, open
it in [AskEditor](https://askeditor.com) and ask "Add a Progress props graphic on the steps I list".
