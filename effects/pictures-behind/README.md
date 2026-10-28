# Pictures behind you

Your own reels, screenshots or headlines appear behind you while you talk, with you cut out in
front of them: dropped onto a loose pile behind your head, pinned up in a wall, or cut one after
another in the same frame on each word you say. Nothing ever covers your face, and the rest of the
room stays exactly as you filmed it.

```
/video-editing pictures-behind my-talk.mp4 pictures=~/my-reels
/video-editing pictures-behind my-talk.mp4 mode=wall pictures=~/my-posts
/video-editing pictures-behind my-talk.mp4 mode=flash pictures=~/press style=print
/video-editing pictures-behind my-talk.mp4 headlines="I quit my job | 30 days of posting | My first client"
/video-editing pictures-behind my-talk.mp4 pictures=~/my-reels at="my hotel"
```

Or just ask: "pile my reels up behind me", "a wall of my posts behind me", "flash my headlines
behind me on each word".

## The three looks

| `mode` | What happens |
|---|---|
| `pile` (default) | Each picture drops onto a pile behind your head, a little turned, settling with a soft shadow on the wall. The newest lies on top. Up to twelve. |
| `wall` | The pictures pin up in a grid on the wall round you, one tile at a time; the first ones take the tiles you leave most in view. Up to twelve. |
| `flash` | One frame sits behind your head. The picture in it cuts to the next on each word, held in exactly the same place and size, like pages flicked on one word. |

## Which picture lands when

- **A picture named for what it shows lands on that word.** `city.jpg` lands as you say "city", a
  `tags.txt` beside your pictures can name more words for each (`room.jpg: hotel, my room`), and
  `map="hotel=room.jpg"` says it outright. A word you took back ("the town, or the city") is never
  used.
- **Every other picture lands in folder order**, one on each word you say from the moment chosen,
  so the pile builds and the flash cuts in time with your voice. The moment is the line that lands
  in your talk unless you give one: `at="my hotel"` (words you say) or `at=4.5` (seconds). Through
  a long pause the pictures keep the same beat rather than stopping.
- `on=keywords` puts every picture on a thing you name instead, `on=order` puts every picture in
  folder order, and `on="2, 4.5, 7"` gives the seconds yourself.
- They all leave together once the sentence ends (`hold=auto`), after `hold=3` seconds, or stay to
  the end (`hold=end`).

The run prints every picture with the second and the word it lands on, and any it left out, with
why.

## Options

| Option | Default | What it does |
|---|---|---|
| `mode` | `pile` | `pile`, `wall` or `flash` |
| `pictures` | | a folder, or files separated by commas (jpg, png, webp) |
| `headlines` | | your own headlines, separated by `|`, each drawn on a paper card; they come after your pictures |
| `map` | | `word=picture` pairs separated by `;` |
| `on` | `auto` | `auto`, `keywords`, `map`, `order`, or seconds |
| `at` | `auto` | where the pictures in folder order start: `auto`, words you say, or seconds |
| `hold` | `auto` | `auto`, seconds, or `end` |
| `size` | `medium` | `small`, `medium` or `large` |
| `style` | `clean` | `clean` (rounded, a thin dark edge) or `print` (a white border) |
| `move` | `on` | the pictures drift a little with the camera, as a real wall would |
| `sound` | `on` | a soft pop as each lands (a light tick on each cut of the flash), well under your voice |
| `cutout` | `careful` | `careful` (two looks at where you are) or `quick` |
| `fix` | | misheard words, `wrong=right` |

## How you stay in front

You are cut out the way the `backdrop` effect cuts you out: two ways of finding a person agree on
every frame, only the speaker is kept, and the edge is firmed and held steady. Only where a picture
lies is your edge worked out again, so no pale rim of the old wall shows on a picture, and every
other pixel of the frame is exactly as filmed. The run measures your face on sixteen frames: how
much of it the cut-out would let a picture through, and how much of it a picture actually shows
over. Both should be 0.

## Honest limits

- **Close shots leave little wall.** On a phone held close, most of the pile sits behind your head
  and shoulders; the run prints how much of each picture stays in view (the pile aims for at least
  40%) and warns when a picture would be mostly hidden. A wall's lower tiles are mostly behind you.
- **Pictures may reach the edges.** A wall fills the room to its sides, and a pile may reach into
  the margins where the app's buttons sit. They are pictures, not words, so nothing you need to
  read sits there; the run counts how often.
- **The cut-out's own mistakes.** Where both ways of finding you are sure a bit of the room is you
  (a lamp against your hair), it stays in front of the pictures.
- **It does not look at your pictures to decide.** A picture lands on a word because its name, its
  tags or your map say so, or in folder order. A camera's number for a name (`IMG_5922.jpg`) says
  nothing, so name your pictures for what they show to have them land on their words.
- **The wall sits on the screen, not on the wall you filmed.** With `move=on` it drifts a little with
  your head, but a camera that walks or pans moves the room behind it and not the pictures.

To change it without rendering again, open it in AskEditor: swap a picture, move it to another
word or change the pile to a wall on the canvas.
