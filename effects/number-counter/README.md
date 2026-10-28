# Number counter

Every amount you say counts up on screen and is whole on the very frame you start to say it. Or a
3, 2, 1 flip clock counts down above your head. It never covers your face.

```
/video-editing number-counter my-talk.mp4
/video-editing number-counter my-talk.mp4 style=flip
/video-editing number-counter my-talk.mp4 values=197K label=Followers when="the lively city"
/video-editing number-counter my-talk.mp4 style=countdown when="my hotel"
```

## The numbers

- **Said.** Leave `values` alone and every amount you say gets a counter: "2,000", "197K",
  "$3 million", "19.5%", "2.5 million". Years, lengths of time ("six months") and small numbers said
  in passing ("two things") are left alone, and so is a figure you took back or said in an aside.
- **One counter for a run.** Amounts said together ("from 2,000 to 15,000, and today 197,000")
  share one counter, which runs on from one to the next. Said newest first ("3 million, up from
  500,000"), each counts up from nought on its own word.
- **Exact.** Each lands written as you said it: "197,000" stays 197,000 and "197K" stays 197K, with
  the decimals you said. Nothing is rounded.
- **Most talks say no amounts.** Then the run stops and says so. Give the number with `values=`.

## When it lands

- The count runs in the moment before, and the number is whole and still on the first frame of
  its first word, timed from where your voice begins (the transcriber marks words a little late;
  the pack corrects for that).
- **Values you give** land on their own words when you say them; else across the words `when`
  names, the last on its last word; else at the seconds `when` gives (`when=2.1,5.4`), moved onto
  the word starting there; else on the strongest words of your talk, and the run says which.
- Each stays `hold` seconds (1.8), cut shorter only when the next number needs the room.

## The styles

| `style` | What you see |
|---|---|
| `roll` (default) | The figures roll like a car's odometer on frosted glass: the last place spins fastest, a place turns over only as the one after it passes 9, blurred while it spins. Places not yet reached are not drawn, and the number grows from the right, centred. |
| `flip` | Split-flap tiles of our own. Each time a figure changes, its tile's top half folds over the hinge. A run's numbers share one row of tiles, the ones not yet reached blank. |
| `countdown` | One big flip tile counting 3, 2, 1 above your head. Count down aloud ("three, two, one") and each number flips on its word; else the 1 lands on the word or second `when` names, a second after the 2. `from=5` counts 5 to 1. |

## Where it sits

Above your head, beside it or under your chin, wherever there is most room over every frame the
counter is up, inside the part of the frame the app's own buttons and caption leave free, with its
shadow. `position=above`, `beside` or `below` picks a side; where that side has no room the counter
goes where there is room, and the run says so.

## Sound

A soft tick runs under the roll, spaced by how fast it turns, a rounder one where it lands; each
flip is a small clack. All are made in code and sit well under your voice. `sound=off` keeps your
clip's sound exactly.

## Options

| Option | Default | What it does |
|---|---|---|
| `style` | `roll` | `roll`, `flip` or `countdown` |
| `values` | `auto` | the amounts you say, or your own: `197K`, `$2K,$15K`, `12%` |
| `when` | `auto` | where given values (or the countdown's 1) land: words said, or seconds |
| `label` | `auto` | the small line under the number: the words you say with it, empty for none, or your own |
| `from` | `3` | where the countdown starts (2 to 9) |
| `position` | `auto` | `above`, `beside` or `below` |
| `hold` | `1.8` | seconds a number stays after it lands |
| `sound` | `on` | ticks and clacks under your voice |
| `fix` | | misheard words, `wrong=right` |

## Honest limits

- It counts what is said as an amount. A number said as a word the transcriber writes oddly, or
  one split across a pause, may be missed: give it with `values=`.
- The roll counts every place on its way, so a big number's last places spin as a blur until the
  last moment; a small one ("40%") rolls slowly enough to read.
- A run said newest first is not drawn as one rise; that is `chart`'s job.
- The countdown counts at most from 9.
- On a close shot with little room round the head, the counter comes smaller, and the run says so.
