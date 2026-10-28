# Sound effects

A whoosh on every cut, a pop as each card lands, a ding on the word that matters. The sounds are
the pack's own, made from code, and each one lands on the first frame of its moment and sits well
under your voice.

```
/video-editing sound-effects my-edit.mp4
/video-editing sound-effects my-edit.mp4 on=cards kind=pop
/video-editing sound-effects my-talk.mp4 on=words words="number one, number two" kind=tick
/video-editing sound-effects my-talk.mp4 sounds="quality riser; 4.5 ding"
```

Or just ask: "add a whoosh on every cut", "pop when each card appears", "a ding when I say the price".

## The sounds

All eight are in `sounds/`, each a short WAV made by `sounds/make_sounds.py` from noise and tones (no
recording or sample from anywhere), levelled so their loudest tenth of a second is the same.

| Sound | What it is | For |
|---|---|---|
| `whoosh` | air rushing past, left to right, rising to its peak and falling away | a cut, a wipe, a fast move |
| `swoosh-in` | something sliding in and stopping | a card or a word landing |
| `pop` | a round, quick pop with a soft thump | a card, sticker or word appearing |
| `click` | a crisp press | a tap or a button |
| `tick` | a small, dry tick | a step in a list, a key |
| `hit` | a low boom with a punch and a crack, still felt on a phone | a big word, a reveal |
| `riser` | noise and a tone climbing three octaves into the moment | building to a reveal |
| `ding` | a clear bell that rings out | a right answer, a number |

## Where they go

- **On the cuts** (`on=cuts`): read from the plan of the edit that made them (`clean-cut` writes
  `<video>.plan.json` beside its video, and it is found by itself), then checked against the picture:
  the frame where the picture really changes. With no plan, the cuts are found in the picture.
- **On the cards** (`on=cards`): every card or graphic another effect laid: each tier-list card as
  it lands, a call to action, a headline, a word behind your head, a before-and-after wipe.
- **On words** (`on=words`): the words you list, every time they are said (a phrase is marked when
  it starts), or with none listed, the most specific word of each sentence. Misheard words are put
  right with `fix`.
- **At moments you give** (`sounds`): seconds or words, each with its sound.
- `on=auto` (the default) reads the plan beside the video: cuts for a cut film, cards for cards,
  else the cuts in the picture.

## How it lands

Each moment is moved to the first frame that shows it: the frame after the cut, the frame the card
sits in its slot, the frame the word begins. The sound is placed so its own landing point is on the
start of that frame, to the sample: a struck sound (pop, click, tick, hit, ding) on its attack, a
whoosh or a swoosh-in on its peak so it rises into the moment, a riser on its end. Two sounds closer
than they can be told apart become one.

Every sound's loudest tenth of a second sits `level` dB under your voice's measured loudness, the
same for every kind, so a click is as present as a whoosh. With `variety`, each repeat is a little
higher or lower, and whooshes alternate their direction, so a run of them never sounds stamped.

## What it checks

- **Stills first.** `--still auto` draws the frames where sounds land and the **sound sheet**: six
  frames over your voice and the sounds on one time line, and a close-up of every landing, the frame
  before beside the frame on the moment, with a line on every frame boundary.
- **After the render**, each sound is found again in the encoded file (your voice taken away) and its
  landing measured against its frame, in milliseconds and frames, with its level under your voice.
- The plan (`<name>.plan.json`) lists every sound with its frame. Edit a time or a kind and run again
  with `--plan`.

## Options

| Option | Default | What it does |
|---|---|---|
| `sounds` | | moments of your own: `[{"at": 3.2, "kind": "pop"}]`, or `3.2 pop; quality ding` |
| `on` | `auto` | `cuts`, `cards`, `words`, `none`, or `auto` |
| `kind` | `auto` | the sound for those moments; `auto` is a whoosh on a cut and a pop on a card or word |
| `words` | | for `on=words`: the words to mark, separated by commas |
| `events` | | the plan to read cuts or cards from, when it is not beside the video |
| `level` | `-9` | how far under your voice the sounds sit, in dB |
| `variety` | `0.3` | how much each repeat differs: `0` identical, `1` the most |
| `fix` | | misheard words, as `wrong=right; …` |

## Honest limits

- **A sound on every cut is a lot.** On a fast edit, most people mark the cuts that matter. Cuts
  closer than 0.45 s share one whoosh.
- **Cards need the plan of the effect that laid them.** A graphic added somewhere else has no plan;
  give its moments with `sounds`.
- **Cuts found in the picture** are jumps of the whole picture. A slow dissolve is not seen as a cut,
  and a camera flash can be.
- **Eight sounds, one character each.** `variety` changes the pitch, not the sound. For a different
  sound, AskEditor makes one to order.
- The sounds sit under your voice, not instead of it: a sound on a word overlaps the word.
- The picture is encoded again.

## Keep editing it

The render is a flat video. To slide a sound, swap it or turn it down without rendering again, open it
in [AskEditor](https://askeditor.com) and ask "Add sound effects to my edit": each sound stays its own
clip on the timeline.
