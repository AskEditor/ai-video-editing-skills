# Colour grade

A colour look in one word, on the whole video or from the moment a word is said, with your skin
kept the colour it really is.

```
/video-editing grade ~/Movies/my-take.mp4 look=film
```

Or just ask: "make it moody", "black and white from when I say hooks", "warmer, but only half as
much".

## The six looks

| Look | What it does |
|---|---|
| `moody` (the default) | darker and quieter: deep shadows leaning teal, colour pulled back, the corners darkened; the face stays lit |
| `black-and-white` | rich black and white with clean blacks, mixed so that skin reads bright rather than muddy |
| `warm` | golden light: whites go cream, highlights gold, the room glows |
| `cool` | crisp blue light: whites go blue-white, shadows a cool blue |
| `pop` | punchier contrast and livelier colour; the dull colours gain the most, so nothing turns neon |
| `film` | soft contrast, faded blacks, gentle highlights, quiet colour and fine grain |

The plain words work too: `cinematic` is `film`, `bw` or `mono` is `black-and-white`, `vivid` is
`pop`, `golden` is `warm`, `blue` is `cool`, `dark` is `moody`.

## Skin stays natural

A grade that warms the room usually turns faces orange; one that cools it turns them grey or
green. Every look here decides how skin-coloured each colour is (by its hue, how much colour it has
and how bright it is) and gives those colours back most of their own hue and richness, while the
rest of the picture takes the look. Darker and lighter skin both fall inside that band. `moody`
also spares the face part of its darkening, so it stays lit while the room goes dark.

Every run measures it: it finds your face on ten frames, reads the skin before and after, and
prints its hue, saturation and lightness with a verdict, such as
`skin: natural: hue 35.9° → 38.6°, saturation ×0.889`. It warns if skin would turn orange, yellow,
pink or grey.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `moody` | one of the six above |
| `intensity` | `1` | how much of the look, from `0` to `1`: `0.5` is half |
| `vignette` | `auto` | darker corners, centred on where you stand: `auto` (the look decides), `off`, or `0` to `1` |
| `grain` | `auto` | fine film grain: `auto` (only `film` has it), `off`, or `0` to `1` |
| `start` | `0` | where the look begins: seconds (`2.5`), or words you say (`start="make it moody"`) |
| `end` | `end` | where it ends: `end`, seconds, or words you say |
| `fix` | | misheard words as `wrong=right` (only matters when `start` or `end` are words) |

A look that starts or ends partway eases in or out over a quarter of a second, beginning on the
word. To start black and white and snap into colour as you say "now", use
`look=black-and-white end="now"`.

## The look as a LUT

Each look is a standard 3D LUT (`.cube`) that this pack works out by code. To use them in another
editor:

```
.venv/bin/python -m fx.effects.grade cubes ~/Desktop/looks
```

The vignette and the grain are drawn frame by frame, so they are not in the `.cube` files.

## What it checks

`--still auto` draws a few frames exactly as the video will show them. Every run also prints the
skin check above, and where the darks, middle and lights of a frame sit before and after, with a
warning if the look pushes much of the picture to pure black or pure white.

## Honest limits

- It recognises skin by its colour, so wood, sand and warm walls are held back from the look a
  little too.
- It grades the whole frame: it cannot leave one person or one object out.
- Each look is made for ordinary video from a phone or a camera. Log or flat footage needs
  converting to normal colour first; the look then goes on top.
- One look at a time. To change it partway, run it twice on different stretches.

## Keep editing it

The render is a flat video file. To change the look, its strength or where it starts, and see it
at once without rendering again, open it in [AskEditor](https://askeditor.com) and ask
"Colour grade my video".
