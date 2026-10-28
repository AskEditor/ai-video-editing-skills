# Transitions

The join between two shots, done on purpose: a punch, a whip, a crossfade, a wipe, a glitch, a
light leak, a flash or a board of dots. Each one sits exactly on its cut, lasts as long as its look
needs, and comes with a sound of its own, made in code and set under your voice.

```
/video-editing transitions my-edit.mp4
/video-editing transitions my-talk-clean-cut.mp4 look=punch
/video-editing transitions my-edit.mp4 look=whip joins=every
/video-editing transitions lake.mp4 clips="~/Movies/trip/" look="crossfade, whip, light-leak"
/video-editing transitions lake.mp4 clips="lake.mp4 4-7; roofs.mp4 6-9; plane.mp4" look=glitch
```

Or just ask: "zoom punch on my jump cuts", "whip pan between these clips", "a light leak between
each idea", "join these clips with crossfades".

## The looks

| Look | What you see | Its sound | Length on its own |
|---|---|---|---|
| `punch` | the new shot lands closer, with a kick of blur, your face growing where it is; the next punch goes back out | a hit | 0.25 s |
| `whip` | the camera is flung sideways: both shots race across, smeared by their own speed, the seam passing the middle on the cut | a whoosh | 0.3 s |
| `crossfade` | one shot melts into the next, blended as light blends, not paint | a soft breath of air | 0.4 s (0.8 s between clips) |
| `wipe` | a soft, slightly leaning edge crosses the frame, a thin line of light riding it | a swish that travels with the edge | 0.45 s (0.6 s) |
| `glitch` | torn bands, colours pulled apart, coarse blocks and scan lines, with bands of the next shot flashing through | a digital crackle | 0.3 s (0.4 s) |
| `light-leak` | light spills in from one side, deep red at its edge and white at its core, covers the cut, and drifts away | a shimmer | 0.8 s (1 s) |
| `dot-matrix` | the old shot breaks into a board of round lights and goes dark; the new one lights up from the middle and snaps into focus | blips, then a click | 0.9 s (1.1 s) |
| `flash` | a burst of white; the cut happens inside it and the new shot comes out of it slowly | a camera snap | 0.3 s (0.4 s) |

The light leak and the dots are drawn here, frame by frame: no downloaded footage of a leak, no
overlay files. `look` takes several, separated by commas, and they take turns, one per join.
`look=auto` is a punch on a talking video and a crossfade otherwise.

## Which joins

A talking video should not get a transition on every cut: that is noise, and it hides the edit
instead of helping it. So:

- `joins=auto` (the default): on a talking video, the joins between ideas; on footage with nobody
  talking, every join; between clips, every join.
- `joins=between-ideas`: the joins where a sentence ends, at least three seconds apart. When two
  are closer, the one with the longer pause wins. A join in the middle of a sentence is left a
  plain cut.
- `joins=every`: every join found.
- `joins=4.5, 12`: at those seconds, each moved onto a cut within a quarter of a second if there is
  one. With no cut there, a punch, whip, glitch, flash, light leak or dots still works inside the
  one shot (a punch on a word, say); a crossfade or a wipe needs two shots and says so.

A talking video with no cut at all (one long take) gets them between its sentences.

## Where the joins come from

**Inside one video:**

- **The cuts `clean-cut` made.** Its plan sits beside the video it wrote, and gives every join to the
  frame; each is then checked against the picture and moved onto the frame where the picture
  really changes, if that is within two frames. `events=` points to the plan when it is elsewhere.
- **The cuts in your own edit**, found in the picture: a frame that differs from the one before far
  more than its neighbours do. In a video of somebody talking, the face must be there on both
  sides, so a hand over the lens is not taken for a cut. A jump cut where you barely moved can
  change too little to find; give its second with `joins=`.

**Joining clips** (`clips=`): the video named first, then a folder of videos (taken in name order)
or a list separated by `;`. Any clip can be followed by the seconds to use, as in `lake.mp4 4-7`.
The film takes the first clip's shape, every clip filling it without being stretched, as large as
the largest clip allows (never past 1080 by 1920), at the lowest frame rate among them, so no clip
shows the same frame twice.

## What happens at a join, honestly

**Inside one video there is nothing spare.** The frames before a cut and the frames after it are all
the file holds, so nothing can overlap. The film keeps its length and your sound exactly as it was:
the voice is never touched, only the transition's own sound is laid under it.

- A punch, a flash, a light leak and the dots need nothing extra. Each is drawn over the frames
  either side, and the cut happens at its peak (inside the kick, the white, the light, the dark
  between the dots). A whip and a glitch also show a glimpse of the other side for the instant both
  are on screen: its nearest frame, smeared by the blur or torn by the noise, which is all that
  instant needs.
- A crossfade and a wipe show both shots at once, which a single file cannot do. So the last frame
  before the cut is held while the new shot comes in, and the first frame after it is held while
  the old one goes out, each for half the transition. On a talking video that freeze lasts a few
  frames and reads as a soft cut.

**Between clips each side has its own frames**, so a crossfade or a wipe overlaps the two: the film
is shorter by each overlap, and the clips' sounds cross over exactly the same frames as their
pictures. When both clips talk, the overlap is never longer than the quiet between the last word
going out and the first word coming in, and the sound switches inside that quiet, so a voice is
never doubled or cut off. If a word runs right up to the cut on both sides, the crossfade holds a
frame either side instead, as it does inside one video, and the run says so. The other looks do
not overlap: the clip going out plays to its last frame, and the next starts on the cut. A punch
between clips lands on the new clip and lasts as long as that clip does.

## The sounds

Each look's sound lands on its join to the sample: a struck sound (the hit, the snap) on the first
frame of the new shot, a swell (the whoosh, the air, the swish, the shimmer) with its peak there, the
blips running through the build and the click on the snap into focus. The whoosh and the hit are
the pack's own (`sounds/`); the rest are made here from noise and tones, with fixed seeds, so the
same run gives the same sound. Each one's loudest tenth of a second sits 10 dB under your voice (a
punch going back out, 13 dB); in a film where nobody talks, they sit at a fixed level instead of
under the wind. `sound=off` keeps
your video's sound only.

## Putting it with other effects

Transitions come straight after `clean-cut`, before anything is drawn over the picture: a punch or a
whip moves the whole frame, captions included. So for a clean talking reel:

```
/video-editing clean-cut my-talk.mp4 captions=off
/video-editing transitions my-talk-clean-cut.mp4
/video-editing captions my-talk-clean-cut-transitions.mp4
```

When the plan beside the video says `clean-cut` drew captions into it, the run warns.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `auto` | `punch`, `whip`, `crossfade`, `wipe`, `glitch`, `light-leak`, `dot-matrix` or `flash`; several separated by commas take turns |
| `joins` | `auto` | `auto`, `every`, `between-ideas`, or seconds |
| `duration` | `auto` | seconds each one lasts; `auto` is the look's own length (the table above) |
| `clips` | | more clips to join after the first: a folder, or paths separated by `;`, each with optional seconds (`lake.mp4 4-7`) |
| `sound` | `on` | `off` keeps only your video's sound |
| `color` | `auto` | the colour of a flash, a light leak, a wipe's line, the dots (one colour, as bright as the picture), or a crossfade that dips through it (`color=black`) |
| `events` | | the plan of the edit that made the cuts, when it is not beside the video |

## What it checks

- `--still auto` draws the frames where the looks peak and a **sheet**: every transition as a row of
  six frames (its start, on the way, the last frame of the old shot, the first of the new, on the way
  out, its end) over a lane of its sound in decibels, your film's sound in blue and the transition's
  in orange, the join marked.
- Every run says where each transition is (its join, to the frame, and its first and last frames),
  how long the film is and why, and, between clips, how each sound crossed.
- After the render the file is read back: the frames written against the frames planned, each sound
  found again in the encoded file and its landing measured against its join, and whether your own
  sound is still in step with the picture.
- The plan (`<name>.plan.json`) lists every join found, which ones were chosen and why the others
  were not. Change a look, a time or a length there and run again with `--plan`.

## Honest limits

- **A crossfade or a wipe inside one video holds a frame either side.** On a talking head it is a few
  frames and reads as a soft cut; across a big change of shot the held frame shows. Between clips
  there is no hold.
- **Jump cuts that barely change the picture** are not found in your own edit. Cuts `clean-cut` made
  are always found, from its plan.
- **A punch needs room round the head.** It goes in far enough that your face ends up larger than
  just before the cut (at least 1.15x, at most 1.3x), but it keeps the whole head in for every frame
  until the next punch, a tenth of the frame above the crown, and the band under the chin free for
  the captions that come after (the bottom of the head above 70% of the frame). So on a very close
  shot, or when you move a lot, it goes in less, or not at all; the run says which and why, and the
  hit still plays. A crown your recording already cut is left as it was.
- **Between clips with no face**, a punch is centred on the frame.
- **Talking clips joined** cross only in the quiet between their words; with no quiet, the crossfade
  holds frames instead.
- **Phone clips filmed in HDR** come in as recorded, without tone mapping, so beside ordinary clips
  they look flat; export them as SDR first.
- **One look at each join**, drawn over the whole frame for its fraction of a second. For a
  transition shaped round something in the shot (a wipe that follows a passing car), AskEditor.
- The picture is encoded again.

## Keep editing it

The render is a flat video. To swap a look, move a transition or change its length without
rendering again, open it in [AskEditor](https://askeditor.com) and ask "Put a transition on each
join of my video": each transition stays on its cut in the timeline.
