# Hook

The first three seconds of the video, built on the first line you say. Your line goes on screen the
way you would type it into the app (black words on white boxes) on the exact frame you start
speaking. It sits where it never covers your face. Five openers go with it: the text alone, a snap
zoom on the line's strongest word, a rushed-in start that settles, an iris opening out of black from
your face, or a jolt and a white flash. You can also start the video on your strongest line and then
play it from the beginning.

```
/video-editing hook ~/Movies/my-take.mp4
/video-editing hook ~/Movies/my-take.mp4 look=snap-zoom
/video-editing hook ~/Movies/my-take.mp4 open=strongest look=whoosh-in
/video-editing hook ~/Movies/my-take.mp4 look=shake-flash style=dark
/video-editing hook ~/Movies/my-take.mp4 text="I edited this in one sentence"
```

Or just ask: "add a hook", "put my first line on screen", "start with my strongest line", "open
with an iris".

## How it decides

- **The line.** It uses the first whole line you say. A scrap of a sentence the clip came in on
  halfway is skipped. Fillers and words you took back are left out, so "the lively town, or the
  lively city" reads "the lively city". It joins the parts of the same sentence while they come
  soon enough. A long line ends where you join on a second thought ("…just like this" before "and
  improve…"). The line keeps its commas, drops its last full stop, and starts with a capital.
- **The moment.** The text is whole on the frame your first word starts, with no fade-in, and
  settles by a hair as it lands. It stays up to just after the line ends, for about two to three and
  a half seconds.
- **The place.** It goes above your head first, which is where typed text usually sits. If there is
  no room it goes under your chin, then in the lower third, always inside the safe zone. Your head is
  measured over every frame the line is up, as the camera shows it, zoom and jolt included. If the
  video already has captions drawn in by the pack, it keeps clear of them too. If nothing fits, the
  type gets smaller. It never covers your face.
- **The openers.** `snap-zoom` snaps in on your face on the line's strongest word and eases back
  out in the next pause. `whoosh-in` starts rushed in on your face and settles back over about half
  a second, smeared along the move. `iris-open` opens out of black through a circle widening from
  your face. `shake-flash` holds a touch closer from the first frame, then jolts in four hard steps
  (a shake that eases reads as a wobble) with a white flash on the strongest word. Every zoom keeps
  your whole head in, with a tenth of the frame above it.
- **`open=strongest`.** The pack's shared picker finds the strongest line in the whole clip and
  plays it first as a teaser, cut in the quiet either side so no word is clipped. The video then
  starts from the beginning. The video gets longer by the teaser. If your strongest line is already
  your first, nothing is played twice.
- **Sound.** Each opener has its own sound under your voice: a pop, a hit or a whoosh. Or
  `sound=off`.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `text-box` | `text-box`, `snap-zoom`, `whoosh-in`, `iris-open`, `shake-flash` |
| `open` | `first` | `strongest` plays your strongest line first |
| `text` | | your own words instead, or `off` for none |
| `style` | `box` | `box` (black on white), `dark` (white on dark), `plain` (white with a shadow) |
| `sound` | `auto` | the opener's own sound, or `off` |
| `frame` | `9:16` | `keep` the clip's shape, or 9:16 following the face |
| `fix` | | misheard words, `wrong=right` |

## Honest limits

- **The line is what you said, not a rewrite.** A weak first line makes a weak hook. Type a better
  one with `text="…"`, or use `open=strongest`.
- **On a tight shot with no room above the head** (clip A), the line goes under the chin and is set
  smaller to fit.
- **The teaser is cut on word times.** A line that runs straight into the next with no pause can end
  a little abruptly.
- **Silence before your first word stays.** The text lands with your voice, so the first frames can
  be bare. Cut the silence with `clean-cut` first.
- To retype the line, move it, or try another opener without rendering again, open the video in
  AskEditor and ask "Add a hook".
