# Music bed

Your own track under your voice. It dips while you talk and comes back up when you stop, fades in
at the start, and is cut to the film's length on the beat, so the music ends when the video ends,
not a bar early and not on a black frame. The finished video is set to the loudness the short-video
apps play at.

```
/video-editing music-bed my-talk.mp4 music=~/Music/my-song.mp3
/video-editing music-bed my-talk.mp4 music=song.mp3 fit=loop
/video-editing music-bed my-talk.mp4 music=song.mp3 level=-10 duck=12
```

Or just ask: "put my song under this", "add background music and make it duck when I talk",
"make the music end when the video ends".

**Bring your own music.** The pack ships none. Use a track you made, bought, or have the rights to.

## What it does

- **It finds where you talk** from the words and the pauses in the sound. A breath between two
  sentences is still talking; a pause longer than 1.2 seconds is a pause, and the music rises in it.
- **It dips under you, and only between phrases.** The music starts going down a quarter of a second
  before your first word, stays at one level for the whole phrase, waits a moment after your last
  word, then rises over half a second. It never rides your syllables, so it never pumps. When you
  talk from the first second, it starts already down; when a pause is too close to the end to rise
  and settle, it stays down.
- **It sits at a level set against your voice.** Your voice is measured; the music sits `level` dB
  under it where you are not talking (6 by default) and `duck` dB lower while you are (16 dB under
  your voice by default).
- **It fits the film.** It finds the track's beat and bars, then:
  - `end-aligned`: the track's own ending (its fade or its last chord) lands on the film's last
    frame. It starts the track later, or takes whole bars out of the middle, or plays some twice;
  - `trim`: the track plays from its start and fades out from a bar line, the fade finishing on the
    last frame;
  - `loop`: whole bars repeat, jumping back where the music before the jump sounds most like the
    music where it lands, as often as the film needs, then it fades the same way;
  - `auto` (the default): `end-aligned` when the track ends by itself, otherwise `trim` when it is
    long enough, otherwise `loop`.

  Every join is a short crossfade that finishes just before a beat, the two beats lined up on their
  attacks, so the beat after a join is played whole and never doubled.
- **It sets the loudness.** The finished video is set to -14 LUFS with its true peak under -1 dBTP,
  measured and applied with ffmpeg's two-pass loudnorm. A peak that would force loudnorm out of its
  linear mode is brought down first by a gentle limiter, so the level of the whole mix is moved by
  one gain and nothing in it pumps.

## What it checks

- **Stills first.** `--still auto` draws a few frames and the **sound sheet**: six frames over your
  voice and the music on one time line, the music's gain drawn on it, and close-ups of every join
  and of the last frame. You can see the dips, the joins and the ending before anything renders.
- **After the render**, the file itself is measured: its loudness and true peak, how far your voice
  sits over the music while you talk and in the open, where the music ends against the last frame,
  that every join has one beat attack, and the words heard again by the transcriber with the music
  under them, compared with the words heard without it.
- The plan (`<name>.plan.json`) holds the track's pieces, the stretches of talk and the gain. Edit
  them and run again with `--plan` to move a join or a dip.

## Options

| Option | Default | What it does |
|---|---|---|
| `music` | | your track: an mp3, wav, m4a, flac, or a video whose sound is the music |
| `level` | `-6` | how far under your voice the music sits when you are not talking, in dB |
| `duck` | `10` | how many dB lower again it goes while you talk (`0` keeps it level) |
| `fade` | `0.5,1.5` | seconds it fades in at the start and out at the end (one number sets both) |
| `fit` | `auto` | `end-aligned`, `trim`, `loop` or `auto` |
| `target_lufs` | `-14` | the finished video's loudness; `off` leaves the level as it is |

People say "quieter music" (`level=-10`), "duck it more" (`duck=14`), "no fade-in" (`fade=0,1.5`),
"loop it" (`fit=loop`), "end on the song's ending" (`fit=end-aligned`).

## Honest limits

- **You bring the music,** and its rights are yours to have.
- **The beat finder wants a steady beat.** A free-tempo piece (an ambient pad, a solo piano played
  loosely) gets no bars: the joins become long crossfades at the most alike moments, and are more
  likely to be heard.
- **A join can still be heard** when the music changes a lot from bar to bar (a song with words, a
  build into a drop). The check line says how alike the two sides were; below about two thirds, try
  another `fit`.
- **It finds talk, not voices.** Someone else talking in the background, or a song with words in the
  clip itself, is treated as part of the clip's sound; the music dips only where words were heard.
- **Quiet recordings are lifted.** Reaching -14 LUFS raises a quiet clip's room noise along with the
  voice. `target_lufs=off` leaves the level alone.
- English speech only, like the rest of the pack. The picture is encoded again.

## Keep editing it

The render is a flat video. To change the track, its level or where it dips without rendering
again, open it in [AskEditor](https://askeditor.com) and ask "Add background music": the music stays
its own clip on the timeline.
