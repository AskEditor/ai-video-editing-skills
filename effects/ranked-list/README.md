# Ranked list

Items pop into a numbered list the moment you say each one: a top-five board that fills slot by
slot, a countdown that ends on number one, a rapid run, or numbered tiles in an arc above your head.
The list never covers your face.

```
/video-editing ranked-list ~/Movies/my-top-five.mp4
```

Or just ask: "make a top 5 list as I say them", "count down to number one", "put the apps I name in
a numbered list".

## Two ways to fill it

**Say it.** With nothing else given, the list is found in what you say, the way people say lists:

| You say | The list |
|---|---|
| "Number five, Canva. Number four is CapCut… and number one is going to be AskEditor." | 5 Canva … 1 AskEditor |
| "Step one, find a window. Step two is to clip the mic on your collar." | 1 Find a window · 2 Clip the mic on your collar |
| "First, good lighting. Second, a decent mic. Lastly, an editor." | 1 Good lighting · 2 Decent mic · 3 Editor |
| "The first thing you need is a good mic. The second thing is light." | 1 Good mic · 2 Light |
| "Drop your clip in, then type add captions, then pick a look." | 1 Drop your clip in · 2 Type add captions · 3 Pick a look |
| "You need three apps: Notion, Figma and Canva." | 1 Notion · 2 Figma · 3 Canva |

The words that only lead into an item are left out ("is going to be", "you want", "we have"), so each
row reads as the thing itself. When you say a line twice, the take said last is kept. A conditional
"if you want it, then ask", "the first time", "give me a second" and "one, two, three, test" are
never taken for a list.

**Or list them**, separated by `;` (or by commas when there is no `;`):

```
/video-editing ranked-list my-video.mp4 items="Canva; CapCut; Premiere Pro@12.4; AskEditor"
```

Each item lands where its name is said. `@12.4` puts it on the word said nearest that second, and
`5: Canva` gives it its number. An item never said is placed between the ones around it, and the run
tells you which, so you can give it a time. A `.json` file takes the same: `["Canva", {"text":
"CapCut", "at": 4.2}]`.

## When each item lands

On the first frame of its word. The transcriber marks words a little late; the pack corrects for
that against the sound, then the row's short entrance runs in the frames before, so the frame where
your voice begins the word already shows the row still and whole. Its sound lands on that same frame.
After a render the file itself is measured: on which frame each row stops moving, and where each
sound lands.

## The looks

| `look` | What you see |
|---|---|
| `board` (default) | Numbered rows. The empty slots show their numbers from the start, the next one to fill a little brighter, and each fills as you say it. |
| `countdown` | The same board counting down, in the order you say things: the first item said is the highest number, and number one comes last, in gold, with a soft bell. Numbers you say always win. |
| `rapid` | A quick run of pills, the newest one in the colour until the next arrives. When more come than fit, the run rolls up a row at a time. |
| `arc` | Numbered tiles of one size on an arch above your head, each popping out from your head onto its place. A long name takes two lines. |

`style=glass` puts the board on frosted glass with your video blurred behind it, and `style=paper`
on a light card.

## Where it sits

Where your head is not: above it, beside it, or under your chin, inside the part of the screen the
apps leave clear. The head is followed over every frame the list is up, hair included, and the list
is made as large as that room allows, above your head when that is nearly as roomy. If your hair
rises into it for a moment, you stay in front of it; your face is never behind or under it.

A name too long for its row is made shorter by whole words: first without words like "just", "the"
or "your", then cut between two words with an ellipsis, never inside a word. The run lists every row
it shortened, and the plan keeps the words as you said them.

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | | what the list holds; empty finds it in your words |
| `look` | `board` | `board`, `countdown`, `rapid` or `arc` |
| `title` | | a heading across the top, such as "My top 5 apps" |
| `position` | `auto` | `top`, `side` or `below` to choose; your face is never covered either way |
| `style` | `solid` | `solid` (a dark card), `glass` (frosted) or `paper` (a light card) |
| `accent` | `auto` | the colour of the numbers, by name or hex |
| `hold` | `1.0` | seconds the finished list stays on screen at the end (`0` for none) |
| `leave` | `0` | seconds the list stays after its last item before it goes; `0` keeps it to the end |
| `sound` | `on` | a soft pop as each item lands, well under your voice; `off` keeps your sound exactly |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `size` | `auto` | `auto` never enlarges; `1080x1920` forces that size |
| `fix` | | misheard words, as `wrong=right; …` |

## What it checks

`--still auto` draws three moments exactly as the video will show them. Every run prints each row
with its number, the word it landed on and its frame, how many rows are still on the first frame of
their word, and three counts that should be 0: frames with anything over your face (or over your head
with nobody laid back over it), and frames where anything leaves the safe area. The plan file lists
every row with its time; change a time or the words there and pass it back with `--plan`.

## Honest limits

- English only. Names of apps and brands are often misheard: read the words first and pass `fix`.
- A list said with no markers ("my favourite is…", "and also…") is not found; give the items instead.
- Items more than 25 seconds apart are taken as two lists; the longer one is shown.
- On a tight close-up there is little room off the face. On our 720×1280 close-up take, five rows
  under the chin have words 15 pixels high, and the run says so; three rows read at 28 pixels. A shot
  with room above your head gives a board two or three times larger.
- The arc needs room above your head. Where there is none, it says so and uses the board.
- Rows carry words only; pictures and logos are not drawn yet.
- One person is followed.

## Keep editing it

The render is a finished video file. To retype a row, change its number or move when it lands without
rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a Numbered list graphic on
the things I list".
