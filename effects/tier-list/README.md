# Tier list

A tier board (S, A, B, C, D) sits above or beside you, and each card drops into its row the moment
you rank it. At the end the finished board holds on screen, so people can pause and screenshot it.

```
/video-editing tier-list ~/Movies/my-ranking.mp4
```

Or just ask: "make a tier list as I rank these", "put an S to D board on this", "drop each one in
its tier when I say it".

## Two ways to fill it

**Say it.** With nothing else given, the rankings are found in what you say, in the ways people
actually say them:

| You say | The card |
|---|---|
| "espresso is S tier", "espresso's S-tier" | Espresso in S |
| "flat white goes in A", "the latte would be a solid B" | Flat White in A, Latte in B |
| "I'd put iced coffee in C tier", "I'd give decaf a D" | Iced Coffee in C, Decaf in D |
| "S tier: espresso and cortado" | both in S |
| "instant coffee, D tier" | Instant coffee in D |

A thing ranked twice ("A tier... no, actually, B tier") keeps the last one you said. A lone letter
only counts as a row when it ends the phrase or comes before "tier", so "this is a tier list" or "I
made a video" never put a card on the board.

**Or list them.** The rows, then the names, with a time in seconds if you want one:

```
/video-editing tier-list my-ranking.mp4 items="S: espresso, cortado; A: flat white@4.2; D: decaf"
```

A `.json` file does the same and can give a card a picture (a logo with a see-through background
is shown whole; a photo fills its square):

```json
{"S": [{"text": "Espresso", "image": "pics/espresso.png"}, "Cortado"],
 "A": [{"text": "Flat white", "at": 4.2}],
 "D": ["Decaf"]}
```

## When each card lands

On the word that names it, using the word times corrected for the transcriber's habit of marking
words late. A card given a time lands on the word that starts nearest it. A card given no time
lands where its name is said; when you say both the name and its row ("espresso is S tier") it
lands on the row, the moment the ranking is known. A name never said is placed between the cards
around it, and the run tells you which ones, so you can give them a time.

## Where the board goes

Where your head is not: above it, beside it, or under your chin, always inside the part of the
screen the apps leave clear. The board is made as large as that room allows, and the roomiest
place wins, above your head when it is close. The top of your head is read from the cut-out of
you, hair included, so short hair leaves more room than a guess from your face would.

If your head rises into the board for a moment, you stay in front of it: the board passes behind
your hair. Your face itself is never behind or under it, in any frame.

The final layout is worked out before the first card arrives. Cards fill their row from the left in
the order they land, a row only wraps to a second line when it has to, and nothing already on the
board moves when a new card arrives.

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | | what goes in each row (see above); empty finds the rankings in your words |
| `tiers` | `S,A,B,C,D` | the rows, top first; any names work (`GOAT,Good,Mid,Skip`), up to eight, and a row can take its own colour (`S=#ff5c7a`) |
| `title` | | a heading across the top of the board |
| `position` | `auto` | `top`, `side` or `below` to choose; your face is never covered either way |
| `style` | `solid` | `solid` (a dark board, white cards), `glass` (frosted, your video blurred behind it, darker over a bright wall) or `paper` (a light board with index cards) |
| `drop` | `fall` | how a card arrives: `fall` onto the board, `slide` along its row, or `pop` in where it goes |
| `hold` | `1.5` | seconds the finished board stays at the end, on the last frame (`0` for none) |
| `sound` | `on` | a soft tap as each card lands, well under your voice; `off` keeps your sound exactly |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `size` | `auto` | `auto` never enlarges; `1080x1920` forces that size |
| `fix` | | misheard names, as `wrong=right; …` |

## What it checks

`--still auto` draws three moments exactly as the video will show them: the empty board, halfway,
and the finished board. Every run also prints where the board went and how large its words are,
which word each card landed on, and three counts that should be 0: frames where anything is drawn
over your face (or over your head without you in front of it), and frames where anything leaves the
safe area. The plan file it writes lists every card with its time; change a time or a row there and
pass it back with `--plan`.

## Honest limits

- English only. Names of apps and brands are often misheard; read the words first and pass `fix`.
- Rankings said in other ways ("my favourite is…", "number one is…") are not found; list those items
  instead.
- On a tight close-up there is little room off the face, so the board is small (on our 720×1280 test
  clip it sits under the chin with words 20 pixels high), and the run says when its words are below
  an easy reading size. A shot with room above your head gives a board two or three times larger.
- On a square or wide video with you in the middle, the board goes beside you and can be narrow.
- One person is followed. Pictures are fitted into squares; a very wide logo is shown small.

## Keep editing it

The render is a flat video file. To drag a card to another row, retype a name or recolour a row
without rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a tier list
that fills in as I rank things".
