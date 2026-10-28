# Call to action

The ending that asks for something: comment a word, follow, or watch a comment being posted. It
shows the tap being made, so the viewer only has to do the same, and it never covers your face.

```
/video-editing cta-card my-talk.mp4
/video-editing cta-card my-talk.mp4 word=GUIDE
/video-editing cta-card my-talk.mp4 kind=comment-box word=GUIDE when="comment guide"
/video-editing cta-card my-talk.mp4 kind=follow word=@yourname
```

## The kinds

| `kind` | What happens |
|---|---|
| `comment` | A dark card: COMMENT, then your word typing itself out in big letters, then a pill saying what it gets them (`line`, "for the link" by default). |
| `follow` | A button pops in. A pointer comes in from the edge, presses it, and it turns from Follow to Following with a tick and a ring. With `word` set to your handle, the handle sits beside it. |
| `subscribe` | The same button, saying Subscribe and Subscribed. |
| `comment-box` | A comment bar: your word types in, the send button lights up, it is pressed, and the comment appears posted above the bar. |
| `auto` (default) | `comment` when you give a word, `follow` when you do not. |

Each is a plain look of its own. None carries any app's name, logo, colours or layout, so it reads
the same wherever the video is posted.

## Where and when

- **When:** the last three seconds by default (`when=end`), long enough to read twice and short
  enough not to sit on the video. `when` also takes a time (`when=12.5`), `start`, or words you
  say (`when="comment guide"`), so the card lands as you ask.
- **Where:** below your chin, where a thumb would be, or above your head when there is no room
  below (the comment card tries above first). It stays inside the area the app's own buttons and
  caption leave free, and the pointer travels in the card's own band, so neither ever crosses
  your face. Too big for the room, the card is drawn smaller (down to 55%); smaller than that, it
  goes into the margin and the run says so.
- **Sound:** a soft click under your voice on the press, quiet keys while the word types, and a
  small pop when it lands. `sound=off` leaves your sound exactly as it was.

## Options

| Option | Default | What it does |
|---|---|---|
| `kind` | `auto` | `comment`, `follow`, `subscribe`, `comment-box`, or `auto`. |
| `word` | (empty) | The word people comment (GUIDE, EDIT), or your handle for `follow`. |
| `line` | `for the link` | The pill under the comment card's word: what commenting gets them. |
| `when` | `end` | `end` (the last 3 seconds), a time in seconds, `start`, or words you say. |
| `position` | `auto` | `top` or `bottom` to choose the side. Never over the face. |
| `accent` | `auto` (yellow) | The colour of the word, the button and the send button: a name or a hex colour. |
| `sound` | `on` | `off` keeps your sound untouched. |

## Honest limits

- **Nothing is made up.** No handle, count or name appears unless you give it. A comment card or
  comment bar needs your word; without one the run asks for it.
- **Say what is true.** The pill says what you write in `line`. Do not promise a link or a
  message you will not send.
- **One card a run,** for about three seconds. A second ask in one video reads as pleading.
- **A close shot leaves little room.** On a face that fills the frame the card sits over the chest
  and comes out smaller.
- **It follows one face.** Someone else walking through the frame is not avoided.

## In AskEditor

The same asks are graphics there: the follow card and the follow or subscribe button, where the
word, the handle and the moment change on the canvas without rendering again.
