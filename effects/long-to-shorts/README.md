# Long video to shorts

Give it a long talking video and it hands back the best few stretches as vertical shorts, best first.
Each one starts on a whole sentence and ends on one, plays between 20 and 60 seconds, has the dead air,
retakes and fillers cut out of it, is made 9:16 following your face, and is captioned. An index beside
them says why each one was picked.

## What to type

```
/video-editing long-to-shorts my-talk.mp4
/video-editing long-to-shorts podcast.mp4 count=5 max=45
/video-editing long-to-shorts my-talk.mp4 rank=story
/video-editing long-to-shorts my-talk.mp4 only=2        # make the second one again
```

You get `my-talk-long-to-shorts.mp4` (the best), `my-talk-long-to-shorts-2.mp4`, `-3.mp4` …, and
`my-talk-long-to-shorts.shorts.md` (with a `.shorts.json` beside it): the ranking, each short's opening and
closing line, its seconds in the original, the reasons, the scores, and the runners-up that were not made.

## How it picks

- **Where a thought starts and stops.** It finds where your talk moves on to a new idea: a long pause,
  a signpost like "first of all" or "now", a change in what you are talking about. A short may only begin
  where a sentence begins and only end where one ends.
- **The hook.** What the first seconds say, heard cold: a short that opens with "and", "but" or
  "as you can see" leans on what came before and scores lower; a question or a clear claim scores higher.
- **The close.** A full stop with a breath after it, or "that's it", beats ending where the next sentence
  carries straight on.
- **One thought.** A short whose edges are where the talk moves on most, and that does not run across a
  bigger change of subject in its middle, scores higher.
- **Its best line, the pace and the length** count a little.

`rank=hook` weighs the opening most, `rank=story` a whole thought from start to close, `rank=pace` the
liveliest stretches. Shorts never overlap.

## Options

| Option | Default | What it does |
|---|---|---|
| `count` | 3 | how many shorts (fewer when the video holds fewer whole thoughts) |
| `min`, `max` | 20, 60 | the shortest and longest, in seconds as the short plays after its cuts |
| `rank` | best | best, hook, story or pace |
| `only` | 0 | 0 makes them all; a number makes only that one |
| `cuts` | clean | clean cuts dead air, retakes, fillers and asides inside each short; none plays it as filmed |
| `captions` | on | captions on each short, never over your face; `look`, `color`, `position`, `max_words` as in `captions` |
| `loudness` | reels | each short at -14 LUFS with peaks under -1 dBTP; keep leaves your own level |
| `frame` | 9:16 | 9:16 following your face (black bars left out, the speaker followed when there are two); keep |
| `size` | auto | the output size |
| `fix` | | misheard words, `wrong=right; …` (they are fixed in the captions and the index) |

Run with `--still` first: it draws a few frames of every short and writes the index, so you can read the
ranking before anything renders.

## Honest limits

- **It reads the words, not the picture.** A moment that is funny because of a face, a gesture or
  something shown on screen is invisible to it, and so is a laugh. Read the index, and use `only=` or a
  plan to make the one you would have picked.
- A speaker who never stops for breath, or a transcript with no full stops, gives it fewer places to
  start and end, so the shorts may be longer than you would cut.
- It never cuts inside a sentence to make a short fit: a talk with few whole thoughts of the right
  length gives fewer shorts, and the run says so.
- Two people talking over each other are followed by whoever the mouths say is talking; it does not
  know who said the better line.
- Each short is heard again after it is made, and any word lost at a cut is reported in the index.
- From a close selfie, or a wide video made 9:16, your head can fill the short from the top tenth
  nearly to the bottom fifth. The captions are then set on one or two lines above you, and may
  cross the top of your hair (never your face) to stay inside the safe zone; the run counts those
  frames as `frames_over_hair`.

To change a short without rendering again, open it in AskEditor and ask "Turn this into a reel": each
cut stays on the timeline to widen or move by a sentence, and the captions stay words you can retype.
