# Chart

A frosted glass card beside or above your head. A line climbs, or bars rise, and the big number
counts up with them, landing on each value as you say it. The card never covers your face.

```
/video-editing chart my-talk.mp4
/video-editing chart my-talk.mp4 values=2K,15K,197K label=Followers when="the lively city"
/video-editing chart my-talk.mp4 type=bars values="Jan:1.2K, Feb:3.4K, Mar:8.9K, Apr:21K"
/video-editing chart my-talk.mp4 type=counter values=197K when=4.2
```

## The numbers

- **Said.** Leave `values` alone and the chart takes the numbers you say. "We went from 2K
  followers to 197K" is two points; "$2,000, then $15,000, then $197,000" is three; "1.5 million,
  up from 200,000" is drawn oldest first. Years ("in 2024"), lengths of time ("six months") and
  small numbers said in passing ("two ways to do it") are left alone. The word you say with the
  numbers becomes the label ("followers").
- **Given.** `values=2000,15000,197000`, oldest first, or with their signs as you would write
  them: `2K,15K,197K`, `$2K,$197K`, `12%,3.5%`, `1.5M`. `Jan:2K` names a point, and the names sit
  under the chart.
- **Exact.** Each value lands written the way it was said or typed. "197,000" is shortened to
  "197K" only because the short form is the same number. Nothing is rounded, and no change is
  worked out for you: the arrow says which way it went, not by how much.

## When it lands

- **On your words.** Each value lands on the word that says it, at the moment your voice begins
  on it (the transcriber marks words a little late; the pack corrects for that). The line draws,
  or the bar rises, in the moment before the value is said, then holds while you talk on.
- **Values you give** that are not said land across the words `when` names, the last on its last
  word (`when="come to life"`). With a time in seconds (`when=12.5`) the card comes in at that
  moment and its values land after it; `start` and `end` put it at the first words or near the end.
- The card stays up for a couple of seconds after the last value lands; `duration` sets the
  seconds on screen.

## The types

| `type` | What you see |
|---|---|
| `line` (default) | A line from point to point, with a soft glow and a fill under it. A dot pops on each value as it lands, and the last one pulses. |
| `bars` | A bar rises for each value. The one being counted is in the accent colour, and the earlier ones settle to clear glass. Each value sits on its bar when there is room. |
| `counter` | The number and its label alone, counting up from nought, or from your first value to your last. |

When the last value lands, a small arrow beside the number shows which way it went.

## Where it sits

- **The room it has.** The card goes where there is most room clear of your face: above your
  head, beside it, or below your chin. That room is measured over every frame the card is up, and
  kept inside the part of the frame the app's own buttons and caption leave free.
- **Its shape.** The card takes the shape that fills its room. It is tall, with the chart under
  the number, or wide, with the chart beside it, and as large as the room allows.
- **Choosing the side.** `position=above`, `beside`, `left`, `right` or `below`. Where that side
  has no room, the card goes where there is room, and the run says so.
- **A close shot.** On a close shot with little room above, the card may cross the top of your
  hair, but never your face.

## The glass

What is behind the card is blurred on every frame, so the room moves through it, and is bent a
little at the card's rim. It is darkened just enough that white type reads on it (about 4.5 : 1)
however bright the room is, and it has a lit edge and a soft shadow. `style=solid` gives a plain
dark card instead, for a background too busy to see through.

## Sound

A very soft tick runs under the count, about 20 dB under your voice, with a rounder one as each
value lands. `sound=off` leaves your sound exactly as it was.

## Options

| Option | Default | What it does |
|---|---|---|
| `type` | `line` | `line`, `bars` or `counter`. |
| `values` | `auto` | The numbers you say, or your own, oldest first: `2K,15K,197K`, `$2K,$197K`, `Jan:2K,Feb:15K`. |
| `label` | `auto` | The small line over the number. Auto takes the word you say with the numbers (with values you give, write it: `label=Followers`); `label=""` shows none. |
| `when` | `auto` | Each value as it is said; or words that are said (the build runs across them), a time in seconds (the card comes in then), `start` or `end`. |
| `position` | `auto` | `above`, `beside`, `left`, `right` or `below`. Never over the face. |
| `accent` | `auto` | Green when it rises, red when it falls; a name or a hex colour. |
| `prefix` | `auto` | What stands before the number (`$`, `£`, `€`). |
| `suffix` | `auto` | What stands after it (`%`, `x`). `K`, `M` or `B` also shortens it: `values=2,15,197 suffix=K` shows 197K. |
| `style` | `glass` | `glass` or `solid`. |
| `duration` | `0` | Seconds on screen; 0 keeps it up for a couple of seconds after the last value lands. |
| `sound` | `on` | `off` keeps your sound untouched. |
| `fix` | (empty) | Misheard words as `wrong=right`, so a number the transcriber got wrong can be put right. |

## Honest limits

- **One chart a run.** For a second chart, run it again on the result.
- **Numbers in English, as the transcriber hears them.** Read them first with
  `python -m fx.run words my-talk.mp4`. If a number was misheard, give `values` yourself or
  correct it with `fix`.
- **Up to eight values.** A line or bars need two or more; a single value is shown as a counter.
  Bars start at nought, so for values below nought use `line`.
- **Two values make a shape, not data.** Between two values the line is a smooth rise that gathers
  pace, or a fall that eases off; only its two ends are your numbers. With three or more values it
  passes through each one and never overshoots between them.
- **Good news can fall.** A fall turns the card red, which is wrong for churn or costs: pass
  `accent=green`.
- **It follows one face.** On a wide shot where the face is too small to find, the card goes in
  the band near the top of the frame.

## In AskEditor

The same look is a graphic there, Figure with trend, where the numbers, the label and the moment
change on the canvas without rendering again.
