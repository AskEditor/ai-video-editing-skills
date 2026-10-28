# Spotlight

The lights go down. With a switch you can hear, the room around you sinks into the dark and loses
its colour, and you stay lit, just as you were filmed. It can also hang your word on the wall
behind your head as a neon sign that flickers on like a real tube, type it out on a dot-matrix LED
board, fill the dark with golden rays and sparkles, or let a beam of light fall on you through
haze.

```
/video-editing spotlight ~/Movies/my-take.mp4
/video-editing spotlight ~/Movies/my-take.mp4 look=neon
/video-editing spotlight ~/Movies/my-take.mp4 look=neon word="OPEN LATE" color=blue
/video-editing spotlight ~/Movies/my-take.mp4 look=led at="the marketing campaign"
/video-editing spotlight ~/Movies/my-take.mp4 look=rays until="let's get started"
/video-editing spotlight ~/Movies/my-take.mp4 look=haze dark=0.9 captions=stacked
```

Or just ask: "turn the lights down when I say hooks", "spotlight on me", "put my word in neon
behind my head", "an LED sign that types it out", "golden rays behind me".

## The looks

| `look` | What happens when the lights go |
|---|---|
| `dark` (the default) | The room goes dark and grey around you, nothing else. |
| `neon` | Your word, in glass tubes on the wall just above your head. Each letter strikes on by itself with a few quick flashes, one catches late, then the sign settles to a steady glow with the faintest flutter. Its light falls on the wall around it and catches the edge of your hair and shoulders. |
| `led` | A dark dot-matrix board behind your head, its unlit dots just visible. Your word types itself out letter by letter as you say it, a cursor blinking ahead. |
| `rays` | Soft golden rays from behind your head, turning slowly, and sparkles twinkling in the dark around you. |
| `haze` | A beam of light falls on you from above, haze drifting down it and specks of dust turning in it. |

## How it decides

- **When.** The lights go down on the word that makes the point of your opening line, the same
  word `words-behind` would choose, so the beat lands where it matters. With a sign they go down
  half a second earlier, and the sign strikes on as the word is said. `at` takes another moment:
  your own words (`at="the marketing campaign"`), a time, or `start`. They stay down to the end,
  or come back up, with the switch again, on `until`.
- **The room.** Darkened and drained of most of its colour by `dark`, with a faint, wide pool of
  light left on the wall behind your head and shoulders, so the room still has depth. The pool is
  wide on purpose: light kept tight round your outline reads as a glow stuck to you.
- **You.** Your face and skin are left exactly as filmed: the run measures your skin's hue,
  colour and lightness with the lights down against the clip, and says so. Lower down you fall off
  a little, as you would under a light from above.
- **Your edge.** You are cut out by two separate looks at each frame, and only what both agree is
  you counts. The edge is firmed, held steady wherever the picture does not move, and only the old
  room's share of each edge pixel goes dark, so no rim of the lit room is left round your hair. The
  render measures both: how much brighter the room is just outside your edge than the dark room
  drawn there, and how far your edge moves from frame to frame where the picture is still.
- **The sign.** As large as the room above your head allows (one to three lines, never wider than
  a sign should be), as low on the wall as it can go while your head never hides more than a
  sliver of a letter on any frame: a neon tube hidden halfway reads as a broken one, and an LED
  letter that loses its foot turns into another letter (an L into an I). With no room above your
  head it goes to the top of the frame and says so.
- **The colour.** Unless you give one, the sign's colour is chosen against what is around it: the
  dark room near it and what you wear beside it (a cap, a collar). Neon starts from pink, LED from
  amber, and moves to another colour when that one would clash. Rays are gold and haze is a warm
  white unless you say otherwise. With `look=dark`, `color` tints the pool of light on the wall.
- **The sound.** A heavy switch as the lights go. The neon crackles as each tube strikes on, then
  hums very quietly under your voice; the LED ticks for each letter; the rays arrive with a soft,
  high shimmer. Every sound is made in code by the pack, set against the loudness of your voice,
  and your own sound is kept as it is. `sound=off` leaves only your sound.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `dark` | `dark`, `neon`, `led`, `rays` or `haze` |
| `at` | `auto` | when the lights go down: `auto` (on the word that makes the point; for a sign, just before it), `start`, a time such as `2.4`, or words you say |
| `until` | `end` | when they come back up: `end` (they stay down), a time, or words you say |
| `dark` | `0.8` | how dark the room goes, from `0` (not at all) to `1` (black) |
| `word` | `auto` | the sign's word or short phrase (up to four words, 24 letters); `auto` takes it from what you say |
| `color` | `auto` | the sign's, the rays' or the beam's colour, or the light on the wall: a name (`pink`, `blue`, `yellow`…) or a hex colour such as `#ff3d9a` |
| `sound` | `on` | `off` keeps only your own sound |
| `captions` | `off` | the pack's captions on top: `stacked`, `one-word`, `serif-chip` or `fitted`; they keep clear of your face and of the sign |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `fix` | | misheard words, as `wrong=right; …` |

## What it checks

`--still auto` draws four moments (just before the lights go, as they go, the sign or light
arriving, and later) exactly as the video will show them. Every run also prints:

- when the lights go down and why, and when they come back up;
- how dark the room is, and how much of its colour it keeps;
- the sign's word, where it hangs, how much of it your head hides at worst, its colour and why,
  and its contrast against the dark wall;
- your skin with the lights down against the clip (it should read "as filmed");
- `frames_over_face`: frames where the dark room would show inside your face because the cut-out
  missed part of it (it should be 0), and the most of your face it showed through;
- after the render: the halo just outside your edge, how far your edge moves where the picture is
  still, and every sound laid with its level.

The plan it writes (`<video>-spotlight.plan.json`) holds every decision: when the lights go down
and come up, the look, the sign's lines, size, place, colour and moments. Change one and run again
with `--plan` to render exactly that.

## Honest limits

- **The cut-out sees what it sees.** Something behind you that both looks take for you (a chair
  you sit in, a lamp touching your hair) stays lit with you; something apart from you is dropped
  and goes dark. Hair finer than a few pixels goes dark with the room rather than glowing.
- **Fast movement** can leave the cut-out a frame behind for an instant at the edge of your hair,
  mostly while the lights are still going down.
- **Glasses** keep the lit room's reflection in their lenses.
- **A busy, moving background** (filmed on a train) darkens well, but its edge is the hardest
  case: watch the stills there.
- **A close shot** leaves little room above your head, so a sign is smaller there.
- **The sign stays where it is** on the screen while you move; on a moving camera it does not stick
  to the wall.
- **English speech only** for choosing the word and the moment; give `word` and `at` for anything
  else.
- **One person.** Someone beside you is lit with you; someone behind you goes dark with the room.
- Nobody has listened to the sounds yet (they are made in code and measured, not yet heard).

## Keep editing it

The render is a flat video file. To change the word, its colour or when the lights go down
without rendering again, open it in [AskEditor](https://askeditor.com) and ask "Add a Neon hook".
