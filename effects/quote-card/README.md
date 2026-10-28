# Quote card

A quote, a post or a clipping pops in as you read it out, and each word lights up on the frame you
say it. It sits above your head or below your chin, never over your face.

```
/video-editing quote-card my-talk.mp4
/video-editing quote-card my-talk.mp4 text="nobody watches past the first three seconds"
/video-editing quote-card my-talk.mp4 look=clipping source="The Daily Ledger, 2024"
/video-editing quote-card my-talk.mp4 text="Ship it today" when=12.5 name="Sam Lee @samlee"
```

## Which words

- **You say a quote:** it is found by itself. A line after "he said", "my client told me",
  "the post says", "the headline reads", "she was like" or "and I quote", up to the end of that
  sentence, becomes the card. Up to three in a video.
- **You give it:** `text="…"` is found where you say it and lit as you say it (several with `|`).
  The card prints your words, so a word the transcript got wrong is printed right.
- **You show it without reading it:** add `when=12.5`, and the card lands at that second with all
  its words lit.
- **Nothing is quoted:** your strongest line stands as a pull quote, and the run says so. A word
  you took back, a first attempt or an aside ("give me a second") is never used.

## The looks

| `look` | What it is |
|---|---|
| `post` | A pane of frosted glass: a round avatar, the name when you give one, the words going from dim to white as you say them (the key word in the accent), and three plain marks under them. Without a name, two plain bars stand where it would be. |
| `clipping` | A cutting of newsprint, torn top and bottom and laid a little crooked, with a highlighter that runs along the words as you say them. The source is printed at the top only when you give it; otherwise a plain double rule. |
| `auto` (default) | A clipping when you say it is from an article, a paper, a headline or a report, or give a `source`; else a post. |

Both are looks of our own: no app's name, logo, colours or layout.

## Where and when

- The card comes in during the beat before the first word, so it is still and whole when you say
  it, and stays `hold` seconds after the last (1.2 by default).
- It goes above your head or below your chin, wherever it can be largest, inside the area the
  app's own buttons and caption leave free, shadow included, and clear of your head on every frame
  it is up. In a tight band it drops the marks under the words, then the name, then gets smaller.
- `captions=stacked` (or another look) adds the pack's captions; they step aside while the card is
  up, so the same words are never on screen twice.
- A soft pop as a post lands, paper laid down for a clipping, under your voice; `sound=off` keeps
  your sound exact.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `auto` | `post`, `clipping` or `auto`. |
| `text` | (empty) | The quote as you say it; several with `\|`. |
| `name` | (empty) | Who said or posted it, on the post (`Sam Lee @samlee`). |
| `source` | (empty) | Where the clipping is from. |
| `when` | `auto` | A time in seconds, for a text you do not read out. |
| `hold` | `1.2` | Seconds the card stays after its last word. |
| `position` | `auto` | `top` or `bottom`. Never over the face. |
| `accent` | `auto` | The key word on a post, the highlighter on a clipping. |
| `captions` | `off` | A caption look, hidden while a card is up. |
| `sound` | `on` | `off` keeps your sound untouched. |
| `fix` | (empty) | Misheard words, `wrong=right; …`. |

## Honest limits

- **Nothing is made up.** No name, handle, source, date or count appears unless you give it.
- **A quote is found by its lead-in.** A quote said with no "he said" before it is not found;
  give it with `text=`. A pull quote is the strongest line, which is not always the one you'd pick.
- **A close shot leaves little room.** On a face that fills the frame the card sits over your chest
  and comes out smaller; a long quote there is small to read. Keep it short.
- **One face.** Someone else walking through the frame is not avoided.
- **Up to 22 words, 3 cards.** Past that it is a transcript, not a quote.

## In AskEditor

The same idea is the Quote marks and Press clipping graphics there, where the words, the look and
the moment change on the canvas without rendering again.
