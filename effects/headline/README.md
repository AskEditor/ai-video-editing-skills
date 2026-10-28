# Headline

Big type on the line that makes the point, two to three times the height of a caption. Each word
appears as it is said, in its final place, above or below your head and never over your face.

```
/video-editing headline my-talk.mp4
/video-editing headline my-talk.mp4 when="bigger captions" style=serif
/video-editing headline my-talk.mp4 text="Edit *less*. | Post more." when="improve your production quality"
```

## What it does

1. **Finds the moment.** With nothing set, it reads what you say and picks the line that makes the
   point: a short clause said whole, with the most specific words in it, that can be set large
   where it will sit. Words the transcriber was unsure of are never picked, nor a line you take
   back ("the lively town, or the lively city"), a first attempt you say again, or the end of a
   sentence the clip came in on halfway. The word in the accent colour is the thing the line
   names, the same word words-behind would set. Name the moment
   yourself with words you say (`when="bigger captions"`), a time (`when=3.2`), `start` or `end`.
2. **Sets the words.** By default, the words you say there, as you say them. Or your own words:
   they land on the words said at `when`, or open the video when nothing is named. One to three
   lines, at most six words a line and twelve in all. The lines are broken where they read best:
   lines close in width, a comma at the end of a line rather than "the", and never a short word
   ("of", "in") alone on a line. One word takes the accent colour, the most specific one, unless
   you choose it.
3. **Lays the headline out once, then lets each word in as it is said.** The whole block is set at
   its final size and place first; each word then appears there on its own word time, by the
   style's move. Nothing else moves, and a highlight's box grows with its words, never ahead of
   them. Word times come from your voice, with the transcriber's lateness taken out, so a word
   appears on its sound, not a few frames after it.
4. **Keeps clear of your face.** It goes above your head, the headline's own place, unless the
   room there is much smaller than below your chin, where captions and hands usually are. It keeps
   clear of the whole head, hair included, when that leaves room for large type. On a close shot
   where it does not, it may cross the top of your hair, but never your face from the hairline to
   the chin, and the run says so. The head is followed over every frame the headline is on screen,
   so a turn of the head never brings the words over the face, and a slam that starts large starts
   smaller when its first frames would reach your face. It stays inside the area the app's own
   buttons and caption leave free. When the room is small the type gets smaller; when nothing fits,
   it goes into the margin and the run says so.

Your sound is kept as it is.

## The styles

| `style` | Looks like | Moves by default |
|---|---|---|
| `block` (default) | heavy capitals, white with a dark outline, the key word in yellow | `slam` |
| `serif` | editorial: a serif, the key word in italic and in colour. The letters turn dark on a bright background and light on a dark one | `rise` |
| `highlight` | each line on a white box that hugs it, the key word on a coloured box | `wipe` |
| `sticker` | dark capitals cut out with a thick white edge, each word a little tilted | `slam` |

## The moves

- `slam`: each word drops in from larger and settles with a small bounce.
- `rise`: each word rises out of a slot just under its line.
- `type`: the letters type out as you say them, with a cursor.
- `wipe`: each word is uncovered from the left as it is said; in `highlight` the box is uncovered
  with it.

## Options

| Option | Default | What it does |
|---|---|---|
| `text` | (empty) | What the headline says. Empty: the words said at `when`. `\|` breaks a line; `*word*` picks the word in the accent colour. |
| `when` | `auto` | `auto` (the line that makes the point), words you say, a time in seconds, `start` or `end`. |
| `style` | `block` | `block`, `serif`, `highlight` or `sticker`. |
| `animation` | `auto` | `auto` (the style's own), `slam`, `rise`, `type` or `wipe`. |
| `accent` | `auto` | The key word's colour: a name (`yellow`, `green`, `pink`…) or a hex colour. |
| `size` | `1` | How big the type is: `1.3` bigger, `0.8` smaller. It is made smaller by itself when there is no room. |
| `duration` | `0` | Seconds on screen. `0` keeps it up until the last word is said, plus a beat. |
| `position` | `auto` | `top` or `bottom` to choose the side. It never goes over the face, whichever you pick. |
| `fix` | (empty) | Misheard words, as `wrong=right`, so `when` can find them. |

Every run writes a plan beside the video (`<video>-headline.plan.json`): the lines, their words, when
each word appears and when the headline goes. Change a word, a line break or a time there, and run
again with `--plan` to get exactly that.

## Honest limits

- **One headline a run.** For a second one, run the effect again on the video it made.
- **English only**, like the rest of the pack, and the words are only as right as the
  transcriber: a misheard word needs `fix`, or your own `text`.
- **A face that fills the frame leaves little room.** On a close shot (clip A in the tests) the
  room above the head is a narrow band: a four-word line comes out about one and a half times the
  height of a caption rather than two to three times, and it crosses the top of the hair. A
  shorter line comes out larger. On a very close shot it may go into the margin, and the run says
  so.
- **It follows one face.** Someone walking through the frame is not avoided.
- **Over captions:** it does not know where captions from another run sit. Put captions on first,
  then choose `position` so the two do not meet.
- **The serif** has no outline or box, so on a very busy background it is the hardest of the four
  to read. `block` or `highlight` read on anything.

## In AskEditor

The same idea is a title card there: the line set as it is said, which you can retype, restyle,
move and retime on the canvas without rendering again.
