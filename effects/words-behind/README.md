# Words behind you

One giant word appears behind your head on the line it belongs to, so you stand in front of the
text. It is chosen from what you say, it lands on the frame you say it, and your head and hair stay
in front of every letter.

```
/video-editing words-behind ~/Movies/my-take.mp4
```

Or just ask: "put a big word behind me", "text behind my head on the hook", "the word HOOKS behind
me when I say it".

## How it decides

- **The word.** Unless you give one, it is taken from your opening line: the word that names the
  thing you are talking about, which is usually a noun at the end of its phrase, a name or a number.
  Not the longest word ("perspective" loses to "hooks" in "from the perspective of hooks"), never a
  word of grammar or filler, and never a word the transcription was unsure of. Never a word you take
  back, either: "the lively town, or the lively city" sets CITY, not TOWN, and a line you say again
  is taken from the take you kept. A clip that starts in the middle of a sentence skips that scrap.
  The run lists the next best words, and the ones it passed over because you took them back, so you
  can ask for another by name.
- **The moment.** The word is in place on the frame its sound starts. Transcription marks words a
  little late, so its times are corrected against the sound first, and the entrance is drawn before
  that frame, not after it. It stays to the end of its line, two seconds at least.
- **The size and the place.** As large as the space above and behind your head allows, and never
  outside the part of the screen the apps leave clear (x from 120 to 960 on a 1080-wide video, below
  the top tenth, above the bottom fifth). It sits low enough for your head to hide a little of the
  letters' foot, which is what makes it read as behind you. How much your head hides is measured on
  every frame the word is on screen, letter by letter: a letter told apart by its foot (an E, a B,
  an L) may lose only a sliver, so an E never turns into an F. When no letter can go behind your head
  and stay readable, the word sits just above it rather than shrinking.
- **The colour.** Light or dark, whichever stands out from the wall right behind the letters and
  from your hair or hat where they cross. When the wall is busy, or close in tone, the wall right
  round the letters is softly shaded until they reach a readable contrast.
- **The edge.** A pixel on the edge of your hair is part you and part wall. Only the wall's part
  takes the word, using the wall's colour from nearby, so no pale rim of the old wall shows round
  your head where it crosses the letters. A faint guess of the cut-out beyond your hair (a plant or a
  shelf it half took for you) is dropped, so the letters never go see-through there, and inside you
  the cut-out is fully solid, so nothing of the word shows through your face.

## Options

| Option | Default | What it does |
|---|---|---|
| `text` | `auto` | the word or short phrase to show, up to four words; `auto` takes it from what you say |
| `when` | `auto` | `auto` is the opening line; a time such as `2.4`; or words you say, such as `"what draws people in"`, and the word lands as they are said |
| `font` | `condensed` | `condensed` (tall heavy capitals, the largest for a long word), `bold` (a wide heavy sans) or `serif` |
| `color` | `auto` | light or dark, chosen for you; or a name (`yellow`, `red`) or a hex colour such as `#ffd400` |
| `size` | `auto` | as large as fits; or a share of that from `0.3` to `1`, such as `0.7` |
| `animation` | `rise` | `rise` (comes up out of a line behind your head), `drop` (falls into place), `fade` or `scale` (lands from larger) |
| `hold` | `auto` | to the end of its line, two seconds at least; or seconds such as `3`; or `end` for the rest of the clip |
| `style` | `solid` | `solid` letters, `outline` (hollow letters) or `glow` (lit letters; the room behind you dims while it is up, and you stay lit) |
| `frame` | `keep` | `9:16` reframes a wider video around your face |
| `fix` | | misheard words, as `wrong=right; …` |

A phrase is set on one to three lines, whichever lets the letters be largest. When its words are
the words you say, each lands as you say it.

## What it checks

`--still auto` draws four moments exactly as the video will show them. Every run also prints:

- the word, the line it came from, and when it lands against when it is said;
- the frames where any letter would show over your face (it should be 0) and outside the clear part
  of the screen (also 0);
- how much of the letters your head hides, at most and for the most-hidden letter;
- the contrast the letters reach against the wall, and how much the wall was shaded for it.

The plan it writes (`<video>.plan.json`) holds every decision: the word, the lines, the size, where
it sits, its colour and its moments. Change any of them and run again with `--plan` to render that
instead of deciding again; the plan's own look (style, animation) comes back with it.

## Honest limits

- It needs a person in the shot, and room above or beside the head. In a close-up with little room,
  the word is smaller, or sits just above your head; a four-word phrase rarely fits there, one word does.
- One person is cut out. Someone walking behind you counts as wall; someone beside you counts as you.
- The word stays still on the screen while you move. On a moving camera it does not stick to the wall
  behind you.
- Hair the same colour as the letters hides its own edge in them; the colour choice works against
  this, and `color` overrides it.
- English speech only, and the word choice is a careful guess, not a reading of your mind: check the
  word it printed, and give `text` when you want another.
- Fast movement can leave the cut-out a frame behind at the very edge of the hair.

## Keep editing it

The render is a flat video file. To retype the word, move it or retime it without rendering again,
open it in [AskEditor](https://askeditor.com) and ask "Add a Words behind hook".
