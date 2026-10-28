# Punch-in

The camera moves closer on the word that matters. By default it is a quick punch that lands on the
first frame of the word, holds to the end of the phrase, and goes back out in the next pause. It
can also be a slow push that arrives on the word, a sideways drift across the line, or a pull-back
that starts close and opens out. The whole head always stays in the picture.

```
/video-editing punch-in ~/Movies/my-take.mp4
/video-editing punch-in ~/Movies/my-take.mp4 at="money, launch"
/video-editing punch-in ~/Movies/my-take.mp4 at=2.4,6 amount=1.3
/video-editing punch-in ~/Movies/my-take.mp4 style=push sound=off
/video-editing punch-in ~/Movies/my-take.mp4 captions=on
```

Or just ask: "punch in on the point", "zoom in when I say money", "slow push in", "pull back at
the start".

## How it decides

- **The words.** With `at=auto`, it takes each spoken line's strongest word, using the same picker
  as the rest of the pack, so a word you took back, a filler or a first attempt is never chosen.
  The strongest go first, at least about two and a half seconds apart, roughly one for every few
  seconds of video. A word where your head leaves no room to move closer is passed over for the
  next one. You can also name the words (`at="money, launch"`) or give seconds (`at=2.4, 6`). A
  second is moved onto the start of the nearest word.
- **The timing.** A punch arrives on the word's first frame. It never lands part-way through a
  word. It goes back out in a pause between words: a smooth return when the pause is long enough,
  or a cut in the gap when it is short. The picture never moves back under the middle of a word.
- **How close.** About 1.2× (1.15× for a drift), aimed at your face where it is rather than at the
  middle of the frame. A tenth of the frame stays above the top of your head and the band under your
  chin stays free for captions. If your head is too big for the amount you asked for, it moves in
  less and says so. If the recording already cut off the top of your head, it leaves that as it was.
- **The blur.** While the camera moves fast it smears along its own movement, as a real shutter
  would. A held frame is sharp.
- **Captions** (`captions=on`, or a look such as `captions=one-word`) are laid over the moving
  picture, not inside it. They keep clear of your face where the zoom puts it, stay inside the safe
  zone, and never grow or slide with the zoom. If your video already has captions burned in, the zoom
  moves them too, and the run tells you so. Run the captions afterwards instead.
- **Sound.** A soft hit on a punch, a riser into a push, a rush of air on a pull, and nothing on a
  drift, all under your voice. Or `sound=off`, or one of the pack's sounds by name.

## Options

| Option | Default | What it does |
|---|---|---|
| `at` | `auto` | the words to land on, or seconds |
| `amount` | `auto` | how much closer, from 1.05 to 1.5 (1.15 to 1.3 reads best) |
| `style` | `punch` | `punch`, `push`, `drift` or `pull` |
| `sound` | `auto` | the style's own sound, `off`, or a sound by name |
| `captions` | `off` | the pack's captions laid over the moving picture |
| `frame` | `9:16` | `keep` the clip's shape, or 9:16 following the face |
| `fix` | | misheard words, `wrong=right` |

## Honest limits

- **It needs room round the head.** On a close-up where the face fills the frame, or where you move
  a lot while a word is held, it moves in less or not at all, and the run says which words it left
  out. Clip D's "city" is one of these.
- **The top of the head is estimated from the face**, not cut out of the picture. A tall hat or a
  big hairstyle can reach higher than that estimate.
- **A clip with no clear words** (music, chanting, a run of "yeah") gives nothing to land on with
  `at=auto`. Give it seconds.
- **The camera does not follow you while it is held.** A held framing is chosen to keep your whole
  head in over every frame of the hold, so a lot of movement makes it wider.
- It decides once and draws once. To change a move without rendering again, open the video in
  AskEditor and ask "Punch in on the point": every move stays a keyframe you can slide or delete.
