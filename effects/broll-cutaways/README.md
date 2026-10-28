# B-roll cutaways

Your own B-roll, cut in over your talking video on the words that name what it shows. Your voice
runs on underneath: the picture goes to the lake as you say "lake" and comes back to you between
two words. The B-roll can fill the frame, sit above you in a split, or fill the frame with you in a
small window in one corner.

```
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll mode=split
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll mode=inset transition=slide
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll map="lake=IMG_5922.MOV; city=IMG_5930.MOV"
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll captions=on broll_audio=duck
/video-editing broll-cutaways my-talk.mp4 broll=~/Movies/broll on="2.5, 7, 12.4"
```

Or just ask: "cut to my b-roll when I mention each place", "show my clips while I talk", "b-roll
on top and me below", "put me in a corner over my footage".

## How a clip finds its word: by name and tag, never by looking

This pack does not look inside your clips to see what they show. It goes by the words you give it,
in this order:

1. **`map`**, if you give one: `map="lake=IMG_5922.MOV; the old town=IMG_5930.MOV"` puts each clip
   on the first time you say its word or words.
2. **The clip's file name.** `lake.mp4`, `rice-fields.mov` and `HarbourAtDusk.mp4` land on "lake",
   "rice" or "fields", "harbour" or "dusk". A plural finds its singular ("cities" and "city").
3. **A `tags.txt` beside your clips**, one clip a line, for clips your camera named:

   ```
   IMG_5925.MOV: town, street
   IMG_5930.MOV: city, rooftops
   DJI_0850.MP4: lake
   ```

4. **Clips named only by a camera's number** (`IMG_5922.MOV`, `DJI_0850.MP4`) and not in
   `tags.txt` say nothing about what they show, so they go, in name order, on the strongest things
   you name that no clip has taken: nouns, places and things, found by rules (never grammar, filler,
   a doing word, a describing word like "better", a time like "a second", your own name as you
   introduce yourself, or a thing said again). That lays your footage on the beat of what you say,
   but a picture of a lake can land on the word "product". Name or tag a clip when its picture
   should match its word.

A clip whose name or tags you never say is left out, and the run lists it with the reason, rather
than laying it on a word it does not show.

**A sheet to look at.** Every run also writes a contact sheet of your B-roll (its path is on the
`broll_sheet` line): one frame of each clip, numbered, with the word it landed on or why it was left
out. The pack itself never looks at it. You, or Claude running the pack, can: when a camera-numbered
clip of the sea has landed on "product", the sheet shows it at a glance, and a `map` or a line in
`tags.txt` puts it on "sea" instead. `on=map` uses only steps 1 to 3. `on="2.5, 7"` puts your
clips, in name order, at those seconds instead, each moved onto a word that starts within a third of
a second.

## When it cuts

- **On the word.** A cutaway starts on the first frame of its word. The transcriber marks every word
  a little late; the pack pulls each start back and checks it against the sound, so the B-roll
  arrives as the word does, not a beat after.
- **Back between words.** A cutaway runs one to three seconds and comes back to you in a pause or
  at the end of a clause when there is one near, holding through a pause rather than cutting back
  to you saying nothing. It never comes back in the middle of a word. `duration=1.5` aims for a
  length of your own.
- **Room for you.** The film opens on you, you are on screen at least a second between two
  cutaways, and it ends on you when it can. Two things named close together ("the town, or the
  city") cut from one B-roll straight to the next instead of flashing your face between them;
  never three in a row on keywords alone.
- **Half at most.** B-roll never covers more than half the film unless `max_share` says so. When one
  more clip would go over, it is tried shorter before it is left out.

## How the B-roll is framed

- **An upright clip is shown as filmed.** A clip of another shape is cropped to the frame round
  where it has the most going on: the detail in it (edges and contrast) and what moves, measured
  over the seconds that are used, leaning toward the middle when nothing clearly stands out. A wide
  clip and a photo push in slowly while they are up.
- **The best stretch of each clip.** The pack looks at the clip's key frames and starts the
  cutaway where the camera is steadiest and the picture sharpest, with something moving in it if it
  can: never the first half second, when the record button is being pressed.
- **Split** puts the B-roll above you and fits the seam to you: your whole face (and your hair when
  there is room) stays in your panel, above the bottom part of the frame where the app's caption
  and buttons go, so a close shot gets a taller panel. A face far from the camera is brought a
  little closer.
- **Inset** keeps you in a rounded window with a light edge and a soft shadow, in a top corner, or a
  bottom one when your captions sit above your head, on the side of the B-roll with less going on.
- **Transitions.** `cut` changes on the word's own frame. `dip` goes through black for a fifth of a
  second. `slide` pushes the B-roll in across you and back out; in split the B-roll slides down
  while you move into your panel, and in inset you shrink into your corner and grow back.
- **HDR clips** (most phones film it) are brought down to normal range the same way the pack reads
  your own HDR talking clip, so they never look washed out next to you.

## Captions over it all

- `captions=on` (or a look: `stacked`, `one-word`, `serif-chip`, `fitted`) adds the pack's captions
  over everything, so they carry on across every cutaway. Over a full-frame cutaway they keep the
  place they had over you, so they never jump when the picture changes. In split and inset they
  keep clear of where your face now is.
- A clip that already has this pack's captions (made by `captions` or `clean-cut`, with its
  `.plan.json` beside it) keeps them: they are drawn again in exactly the same place over each
  full-frame cutaway, so they never vanish under the B-roll. `captions=auto` (the default) does this
  by itself. Split and inset would cut through captions burned into your picture, so on such a clip
  they stop and say how to go on.

## Sound

Your voice is left exactly as it was. The B-roll's own sound is off, or with `broll_audio=duck` it
sits about 18 dB under your voice, never louder than it was filmed, starting a moment before its
picture and trailing a moment after, so the place is heard as well as seen.

## Options

| Option | Default | What it does |
|---|---|---|
| `broll` | | your B-roll: a folder, or clips and photos separated by commas |
| `map` | | which clip goes on which word: `lake=lake.mp4; city=roofs.mov` or `{"lake": "lake.mp4"}` |
| `mode` | `full` | `full`, `split` (B-roll above, you below) or `inset` (you in a corner window) |
| `on` | `keywords` | `keywords` (names, tags, then the things you name), `map` (names and tags only), or seconds |
| `duration` | `auto` | `auto` (1 to 3 s, ending at a pause or a clause end near two seconds) or seconds from 1 to 3 |
| `max_share` | `0.5` | the most of the film the B-roll may cover, from 0.05 to 1 |
| `transition` | `cut` | `cut`, `dip` or `slide` |
| `broll_audio` | `mute` | `mute`, or `duck` for the B-roll's own sound under your voice |
| `captions` | `auto` | `auto` (keep this pack's captions across), `off`, `on`, or a look's name |
| `frame` | `keep` | `9:16` reframes a wider video round your face first |
| `fix` | | misheard words, as `wrong=right; …`, so names and tags find them |

## What it checks

`--still auto` draws you just before the first cutaway, the middle of the first two, and you again
after the first comes back, exactly as the video will show them. Every run also prints:

- each cutaway: when, the word it lands on, the clip, and why (map, name, tags, keyword or time);
- that each starts on its word, and how many come back to you inside a word (it should be 0);
- how much of the film is B-roll, against the most allowed;
- how each clip was framed, and which clips were left out and why;
- where the contact sheet of your B-roll is;
- for split and inset, on how many frames your face stays whole in its panel;
- `frames_over_face` and `frames_outside_safe_zone` for the captions and the inset (both 0).

The plan it writes (`<video>-broll-cutaways.plan.json`) holds every cutaway: its seconds, its clip,
where in the clip it starts, and the point it is centred on. Swap a clip, move or remove a cutaway
there, and run again with `--plan` to get exactly that; an end that falls inside a word is moved to
the nearest place between words, and the run says so.

## Honest limits

- **Matching is by name and tag, not by looking.** A clip called `beach.mp4` that shows a car park
  still goes on "beach". Camera-numbered clips land on the strongest words in name order, whatever
  they show.
- **English words and simple rules.** Nouns are told from other words by rules, not by a model: a
  describing word can still slip through, and a word misheard by the transcriber is missed unless
  `fix` puts it right.
- **Each clip once.** A clip is used for one cutaway, on the first mention that fits; something said
  twice gets one cutaway.
- **A clip needs about two seconds.** Its first half second and its last moments are never used, so
  a clip under 1.8 s is left out.
- **Screens are cropped like footage.** A screen recording in full mode keeps only a slice of its
  width; `split` shows more of it, and `split-layout` shows it whole.
- **Split and inset want a tall video.** On a wide one, add `frame=9:16` first.
- **Other apps' captions are not carried.** Only this pack's captions are known (from the plan beside
  the clip); burned-in captions from anywhere else are hidden under a full-frame cutaway.
- **The B-roll's sound is one level per clip**, set against your voice; music in a clip is treated
  like any other sound.

## In AskEditor

To swap a clip, move a cutaway or change the layout without rendering again, open the video in
[AskEditor](https://askeditor.com) and ask "Cut my B-roll in where I name each thing": every
cutaway stays its own clip on the timeline, to trim, slide or replace.
