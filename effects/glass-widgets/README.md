# Glass widgets

A small cluster of frosted glass widgets gathers round your head as you talk. The larger ones tuck
behind your hair and shoulders, the smaller ones float in front, and each one carries something of
yours: a figure you say, your name, the line that makes your point, how far along you are. Nothing on
it is decoration, a badge or anyone's logo.

```
/video-editing glass-widgets my-talk.mp4 name=Sam
/video-editing glass-widgets my-talk.mp4 widgets="number: 197K followers | note: Shot in one take | name: @sam"
/video-editing glass-widgets my-talk.mp4 widgets="stat: 2K,15K,197K views | progress: 72% edited" when="the whole year"
/video-editing glass-widgets my-talk.mp4 name=Sam layout=bento tone=light
```

## The widgets

| Kind | What it shows | Write it |
|---|---|---|
| `number` | A big figure counting up to its value, with a small label over it | `number: 197K followers` |
| `stat` | A small chart: a climbing line, or bars, and the last value | `stat: 2K,15K,197K views`, `stat: bars Mon:1.2K,Tue:3.4K views` |
| `name` | Your name or handle on a pill, your initials on a round chip | `name: Sam Lee`, `name: @sam` |
| `note` | A short line, on one or two rows, with a bar of colour beside it | `note: Shot in one take` |
| `progress` | A ring that fills to a percentage, or steps that light one by one | `progress: 72% edited`, `progress: 3/5 days` |
| `icon` | A simple sign drawn in the pack, with a caption if you give one | `icon: star`, `icon: pin Lisbon` |

The signs are check, star, play, heart, bolt, spark, clock, up, pin, chat, eye, user, mic and
calendar: plain shapes, never a company's mark. Separate widgets with `|`; two to five suit one
head. The same list can be given as JSON, with `kind`, `text`, `value`, `when` and `layer`
(`front` or `back`) for each.

**Left on `auto`**, the widgets come from the clip itself:

- every figure you say ("197 thousand followers") becomes a number that counts up as you say it;
- two or more said together ("from 2K to 197K") become a small chart;
- a percentage becomes a ring;
- `name=` gives the pill, and when you say your name the pill lands on it;
- the line that makes your point becomes a note, with a sign beside it when its words name one (a
  city gets a pin, a video a play button, quality a star).

## When they land

- **On your words.** A widget that is said lands as you say it, timed from where your voice begins
  on the word, not where the transcriber marked it. The rest land one after another on the words
  that follow, never two at once.
- **Your timing.** `when="the whole year"` brings them in across those words, and `when=4.2` starts
  them at that second. A single widget can have its own: `number: 197K followers @ "the whole year"`.
- **Leaving.** They stay `hold` seconds after the last one has landed (2.6 by default), then leave
  together, the first to arrive the last to go. `hold=end` keeps them to the end.

## Where they sit

- **Round your head, measured on you.** The person is measured in every frame the cluster is up,
  with each widget where its drift takes it, so a turn of the head never brings your face under one.
- **Never over your face.** A widget in front keeps clear of your face from the hairline to the
  chin. A widget behind may pass behind your hair, ears and shoulders, never behind the middle of
  your face, and its writing is never hidden by you.
- **Inside the safe area**, clear of the app's own buttons at the bottom and the bar at the top.
- **One size of type.** Every widget is drawn at one scale, the largest the room allows, with the
  first figure or line a little larger as the lead when that costs nothing. When there is not room
  for every widget at a size a phone shows clearly, the widgets you listed come a little smaller,
  then the least important one is left out; the run says which.

| `layout` | What it does |
|---|---|
| `auto`, `around` | Round the head; the larger widgets tucked behind it, the smaller in front |
| `above` | A row above the head, all in front |
| `bento` | Tiles lined up in rows and columns round the head |

## Depth

The widgets behind and the widgets in front follow your head by different amounts as you move, and
each floats very slowly on its own, so the cluster reads as layers in the room rather than stickers
on the screen. The ones behind are a touch softer, and the ones in front cast a deeper shadow.
`depth=0` turns all of it off: every widget in front, still.

## The glass

What is behind each widget is blurred on every frame and bent at its rim, so the room moves through
it. Behind you, the room is continued under you first, so your hair never smears into the glass.
The glass is toned so its type reads at about 4.5 : 1 or better, and the run prints the measure.

| `tone` | What you see |
|---|---|
| `auto` | Dark glass over a bright room, light glass over a dark one |
| `dark` | Smoked glass with white type |
| `light` | Frosted white glass with dark type |
| `tint` | Glass tinted with `accent` |

Once the last widget has landed, one band of light crosses the cluster.

## Sound

A soft glass chime sounds as each widget arrives, each a step higher than the one before, about
18 dB under your voice; a figure ticks softly as it counts. `sound=off` leaves your sound exactly as
it was.

## Options

| Option | Default | What it does |
|---|---|---|
| `widgets` | `auto` | The widgets, `kind: what` separated by `\|`, or JSON; auto takes them from your words. |
| `name` | (empty) | Your name or handle for the pill. |
| `layout` | `auto` | `around`, `above` or `bento`. |
| `depth` | `0.6` | 0 is flat and still; above it the larger widgets tuck behind you, and the higher it is the further the layers drift apart. |
| `tone` | `auto` | `dark`, `light` or `tint`. |
| `accent` | `auto` | The rings, the chips and the tinted glass: a name or a hex colour. |
| `when` | `auto` | On their own words; or words that are said, or a second. |
| `hold` | `auto` | Seconds after the last lands, or `end`. |
| `sound` | `on` | `off` keeps your sound untouched. |
| `fix` | (empty) | Misheard words as `wrong=right`, so a name or a figure is heard right. |

## Honest limits

- **A close shot leaves little room.** When your head fills the width there is room only above it,
  so fewer widgets fit at a readable size; with five the type comes small, and the run says so. On
  a very tight shot they may sit below your chin, over your hands, never over your face.
- **Behind you relies on the cut-out.** Fine hair against a wall of the same colour can lose a few
  strands at the edge, and someone passing close behind you can be taken for part of you.
- **Figures in English, as the transcriber hears them.** Read the words first with
  `python -m fx.run words my-talk.mp4`; correct a misheard figure or name with `fix`, or list the
  widgets yourself.
- **The line chosen as a note is picked by rule.** If it is not the one you would choose, write the
  note yourself.
- **The drift follows your head, not the camera.** A camera that pans a long way does not carry the
  widgets with the room.
- **One cluster a run.** For a second one later in the video, run it again on the result.

## In AskEditor

The same look is a set of graphics there, where each widget can be moved, retyped or retimed on the
canvas without rendering again.
