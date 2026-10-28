# Pictures above your head

Your own photos, screenshots and logos pop in above or beside your head, each one on the word that
names it. Say "the town, or the city" and the town appears as you say "town", the city joins it on
"city", and both leave once the sentence is done. Nothing is ever laid over your face.

```
/video-editing pictures-above-head my-talk.mp4 pictures=~/Pictures/trip
/video-editing pictures-above-head my-talk.mp4 pictures=~/Pictures/trip layout=arc
/video-editing pictures-above-head my-talk.mp4 pictures=~/Desktop/proof layout=rotate style=glass
/video-editing pictures-above-head my-talk.mp4 pictures=~/Desktop/proof map="results=IMG_4410.PNG; our app=Screenshot 2026-10-28 at 10.12.31.png"
/video-editing pictures-above-head my-talk.mp4 pictures="~/Pictures/trip, ~/logos" style=print
```

Or just ask: "put pictures above my head as I mention each place", "show the logos of the tools I
name", "add my screenshots as proof", "a row of my photos over my head".

## How a picture finds its word: by name and tag, never by looking

The pack does not look inside your pictures to decide where they go. It goes by the words you give
it, in this order:

1. **`map`**, if you give one: `map="town=IMG_1.jpg; my hotel=room.jpg"` puts each picture on the
   first time you say its word or words.
2. **The picture's file name.** `town.jpg`, `rice-fields.jpg` and `HotelRoom.png` land on "town",
   "rice" or "fields", "hotel" or "room". A plural finds its singular.
3. **A `tags.txt` beside your pictures**, one picture a line, for names that say nothing:

   ```
   IMG_4410.PNG: results, revenue
   Screenshot 2026-10-28 at 10.12.31.png: our app
   ```

4. **Pictures named only by a camera, a phone or a screenshot tool** (`IMG_5922.JPG`,
   `Screenshot 2026-10-28 at 10.12.31.png`, `WhatsApp Image …`) go, in name order, on the strongest
   things you name that no picture has taken: nouns, places and things, found by rules (never
   grammar, filler, a doing word, a word that describes like "tiny" in "it was tiny", your own name
   as you introduce yourself, or a thing said again). That keeps your proof on the beat of what you
   say, but a screenshot of your results can land on "coffee". Name or tag a picture when it should
   match its word.

A picture whose name and tags you never say is left out, and the run lists it with the reason.

**Logos.** A logo lands only where its brand's whole name is said: `AskEditor.png` on "AskEditor"
or "Ask Editor", never on "ask" in "just ask for it". Name each logo file for its brand, or put
logos in a folder with "logo" in its name. A logo is never laid on a thing you name; give it a `map`
when you call the brand something else ("our editor"). **This pack ships no brand marks and never
downloads one:** a logo is only ever a file you give it, shown as you gave it.

**A sheet to look at.** Every run writes a contact sheet of your pictures (its path is on the
`pictures_sheet` line): each one numbered, with what it was taken for (photo, screen, page or logo),
the part of it that is shown, and the word it landed on or why it was left out. You, or Claude
running the pack, can look at it and write a `map` for any picture on the wrong word.

## When they appear, and when they go

- **On the word.** Each picture is there on the first frame of its word, with a quick pop. The
  transcriber marks every word a little late; the pack pulls each start back and checks it against
  the sound, so the picture arrives as the word does.
- **Together, then gone together.** Pictures named while others are up join them, in the order you
  name them. They stay until the sentence that names the last of them has ended (about two to four
  seconds), then leave together. A picture named just after they would have gone takes their place
  without a blank moment between. `hold=3` keeps them three seconds after the last lands;
  `hold=end` keeps them to the end.
- **Never two at once.** Two pictures never land within a quarter of a second of each other, and a
  picture named in the last second of the clip is left out, since nobody could see it.

## Where they go

The head is measured in every frame the pictures are up, so a speaker who moves never brings a
picture over their face. Everything stays inside the part of the frame the apps leave clear (not
under the caption and buttons at the bottom, not behind the bar at the top).

| `layout` | What it looks like |
|---|---|
| `auto` | chosen by the room your shot leaves and how many pictures share a moment |
| `arc` | a row of tiles curving over your head, the ends a little lower and turned out, filling left to right as you name them |
| `cluster` | a loose pile above your head, the first a little larger, each turned a touch, overlapping a little |
| `column` | stacked on the side of you with more room, top to bottom as you name them |
| `rotate` | one at a time, each new picture replacing the last |

- The tiles take the shape the room favours: upright, square or wide, so a short band over your
  head gets wide pictures rather than tiny tall ones.
- They are as large as the room allows. The run says how large the smallest is, and warns when it
  is small on a phone.
- They keep clear of your whole head when they can. When that leaves only small pictures, they may
  come down over the top of your hair, never your face, and the run says so.
- **On a tight close-up** your head and ears fill the width, so nothing readable fits above or
  beside you inside the safe part of the frame. The pictures then sit under your chin, over your
  chest, and the run says so. A column beside you needs a shot where you sit off to one side.

## How each picture is framed

Every picture sits in a tile with rounded corners, a soft shadow and a hairline edge, so it parts
from a white wall or a white shirt. Nothing is ever squashed.

- **A photo** is cropped to the tile's shape round its subject: a face it holds, kept whole with room
  over it, or else where its detail is.
- **A screenshot** lands whole, so it is recognised, then pushes in to its busiest part, where its
  writing is large enough to read on a phone. One that already reads whole stays whole.
- **A page** (a screenshot far taller than wide) rests on its top, then scrolls slowly down to its
  busiest part and rests there.
- **A logo** is shown whole, trimmed to its own edges, on its own ground (a see-through logo on
  white, or on dark when it is light). It is never recoloured, redrawn or replaced.

Three looks (`style`): `clean` (the default), `print` (a white border, like a printed photo) and
`glass` (each picture on a pane of frosted glass that blurs what is behind it).

## Sound

A soft pop as each picture lands, on the same frame, a little higher for each picture of a row,
well under your voice. After the render each pop is found again in the finished file and measured
against its frame. `sound=off` leaves your sound exactly as it was.

## Options

| Option | Default | What it does |
|---|---|---|
| `pictures` | | your pictures, screenshots and logos: a folder, or files separated by commas |
| `map` | | which picture goes on which word: `town=town.jpg; my hotel=room.jpg` or `{"town": "town.jpg"}` |
| `layout` | `auto` | `auto`, `arc`, `cluster`, `column` or `rotate` |
| `on` | `keywords` | `keywords` (names, tags, then the things you name), `map` (names and tags only), or seconds such as `2.5, 6` |
| `hold` | `auto` | how long they stay after the last lands: `auto` (to the end of the sentence), seconds, or `end` |
| `size` | `auto` | `auto` (as large as the room allows), `small`, `medium` or `large` |
| `style` | `clean` | `clean`, `print` or `glass` |
| `sound` | `on` | `on` (a soft pop as each lands) or `off` |
| `fix` | | misheard words as `wrong=right; …` (a name the transcriber got wrong) |

**Change one thing by hand.** Every run writes a plan beside the video (`<video>-pictures-above-head.plan.json`):
each picture's file, its word and second, when it leaves, its tile (`[x, y, width, height, turn]`),
and the part of the picture shown (`window`, and `from` where a screen's push or a page's scroll
starts). Change any of them and run again with `--plan` to render exactly that.

## Honest limits

- **It matches by name, not by sight.** A picture lands on the word its name or tags say; nothing
  checks the picture shows it. The contact sheet is there so you, or Claude, can look.
- **A screenshot's busiest part is a guess** made from where its detail is. When the part that
  matters is somewhere else, set its `window` in the plan.
- **Logos: only yours.** Nothing is fetched. A brand name you say with no file given gets nothing.
  A brand whose name is also an everyday word lands on that word whenever it is said; use `map`.
- **One thing over the picture at a time.** Like every effect of the pack that draws over your
  video, a second one run on this result cannot see the pictures and may lay its own over them.
  Run captions or a headline first only if they keep clear of the room above your head.
- **The pictures are fixed once placed.** They are placed for the moments they are up, so a speaker
  who stands up or walks out of frame can leave a gap or meet a picture that cannot move.
- English speech only, one face followed.

In AskEditor the same look is the Items beside you graphic: every picture stays a layer on the
canvas, so you can swap one, move the row or change the word it lands on without rendering again.
