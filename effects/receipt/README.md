# Receipt

Your list prints out like a till receipt above your head: a new line each time you say an item, a
price beside it when you say one, and a total at the bottom. The paper hangs from a printer's dark
mouth, grows downward a line at a time, and never covers your face.

```
/video-editing receipt ~/Movies/my-list.mp4
```

Or just ask: "print my list like a receipt", "a receipt of what I spent, with the total", "make the
things I list print out like a bill".

## Two ways to fill it

**Say it.** With nothing else given, the list is found in what you say: "first … then … and then",
"step two", or a run with a last "and" ("a coffee for four dollars, a bagel for three dollars, and a
new microphone for forty-five dollars"). Each line prints on the first frame of its word.

**Or list them**, with `;` between them; a price after a name and a time with `@` are both optional:

```
/video-editing receipt my-video.mp4 items="Coffee $4; Bagel $3@2.5; New mic $45"
```

## Prices and the total

- A price said inside an item or just after it ("a coffee for four dollars") prints on the frame
  its first word is said, with a dotted leader to it, and leaves the item's own words ("Coffee").
- `total=auto` prints a total when you say one ("that comes to fifty-two dollars"), on that word.
  Otherwise, once two or more prices are said in one currency, it prints their sum half a second
  after the last. `total=off` never prints one; `total="$52"` prints what you type.

## Where it sits, and when there is no room

Above your head when there is room (else beside it, else under your chin), inside the part of the
screen the apps leave clear, checked on every frame, as large as the room allows. When the whole list
would only fit at a size that is hard to read, the paper stops growing and the lines scroll up under
the printer's mouth instead, the newest always on show (an 8-line list on a clip with room for 5).

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | | the lines; empty finds them in your words |
| `title` | | the heading printed at the top (`RECEIPT` when empty) |
| `total` | `auto` | `auto`, `off`, or the total to print |
| `position` | `auto` | `top`, `side` or `below` to choose; your face is never covered either way |
| `sound` | `on` | a small printer's chatter as each line prints (a longer one for the total); `off` keeps your sound exactly |

## What it checks

`--still auto` draws three moments exactly as the video will. Every run prints each line with the
word it prints on, each price and the total with when and why, whether the paper scrolls, and three
counts that should be 0: frames with the paper over your face, frames over your head with nobody laid
back over it, and frames outside the safe area. The plan lists every line, price and time; change one
and pass it back with `--plan`.

## Honest limits

- English only. Prices are found only in dollars, pounds and euros, or as plain figures.
- Lines said with no markers ("we also…") are not found; give them with `items`.
- A line's words are the clause as said; give your own with `items` for shorter ones.
- On a tight close-up with no room above the head the paper goes under the chin and is small.
- No talking test clip says prices yet: they are proven on a computer voice over a grey frame and
  on prices typed over a real take.

## Keep editing it

The render is a finished video file. To retype a line or a price without rendering again, open it in
[AskEditor](https://askeditor.com) and ask "Add a Fact card graphic with the things I list".
