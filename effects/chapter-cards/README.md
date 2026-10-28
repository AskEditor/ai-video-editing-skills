# Chapter cards

Your talk in parts. Each part's title lands in the pause just before it, and a thin rail under the
app's top bar fills as the video plays, one segment a part, so a viewer can see how far in they are
and how much is left.

```
/video-editing chapter-cards my-talk.mp4
/video-editing chapter-cards my-talk.mp4 look=serif
/video-editing chapter-cards my-talk.mp4 titles="Why it matters | 0:24 The setup | 1:02 What it cost"
```

## What it does

1. **Finds the parts of your talk** from what you say, with nothing paid and nothing uploaded. A new
   part begins where three signs agree: your voice stops at the end of a sentence (the longer the
   pause, the stronger the sign); the next sentence opens with a signpost such as "first of all",
   "the second thing", "next", "now let's" or "finally"; and the words you use change from one
   stretch to the next. No part is shorter than `min_part`, and a talk that stays on one idea stays
   one part.
2. **Names each part by its own key words**: the word or pair of words said in that part more than
   anywhere else in the talk, such as "Hotel" or "Retirement place". These are a good first guess,
   not a writer's titles: the run prints every part with its time and title (`check: parts:`), and
   `titles=` replaces them.
3. **Lands each title in the pause before its part.** The card appears once your voice stops and is
   gone before you speak again, so it never sits over your words. In a long pause it comes at the
   end, leading straight into the part. The first part gets no card: your opening line is the hook.
4. **Uses the small title where there is no pause.** If you go straight on (a pause under about a
   second), the part's title hangs small from the rail, at the mark where the part begins, for a
   few seconds while you talk.
5. **Never covers your face.** A card goes above your head or below your chin, whichever has more
   room, clear of your head over every frame it is on screen; on a tight shot it may cross the top of
   your hair, never your face. Everything stays inside the part of the screen the apps leave free.
   If you lift the camera and your face rises into the rail's band, the rail fades out until it is
   clear.

Your sound is kept as it is.

## The looks

| `look` | Card | Rail |
|---|---|---|
| `bold` (default) | the number on a coloured tab, the title in heavy white capitals with a dark edge | one segment a part, the played part in the accent |
| `serif` | a large number in the accent beside a thin rule and the title in an editorial serif; the letters turn dark on a bright wall | a hairline with a dot where each part begins, filled once it is passed |

## Options

| Option | Default | What it does |
|---|---|---|
| `titles` | `auto` | `auto` names each part by its key words. A number (`4`) asks for that many parts. Or your own titles between `\|` marks, one per part; a title may begin with the time its part starts (`0:31 The train`), and the part begins at the nearest real pause within a couple of seconds. |
| `cards` | `full` | `full`: the card in the pause, or the small title where the pause is too short. `corner`: always the small title. `off`: the rail only. |
| `rail` | `on` | `off` hides the rail. |
| `look` | `bold` | `bold` or `serif`. |
| `min_part` | `auto` | The shortest a part may be, in seconds. `auto` is a tenth of the video, between 6 and 15 seconds. Raise it for fewer, longer parts. |
| `color` | `auto` | The accent: a name such as `yellow` or `pink`, or a hex colour. |

Misheard words go with `fix`, as with every effect (`fix="Hawaiian=Hua Hin"`), so the titles come out
right.

Every run writes a plan beside the video (`<video>-chapter-cards.plan.json`): each part's start,
title and why it begins there, and each card's time and place. Change a title or a start time
there, and run again with `--plan` to get exactly that. The plan also carries the parts as YouTube
chapter lines, and says whether YouTube would take them; the `post-copy` effect writes them out with
the rest of what you post.

## Honest limits

- **The titles are a first guess.** They are the part's most telling words, not a summary, and a
  word the transcriber got wrong shows up in them. Read the `check: parts:` line, and pass `titles=`
  (or `fix=`) whenever one is not what the part is about.
- **It needs a talk that moves on.** A clip of a few seconds, or one that stays on a single idea, is
  one part, and the run says so rather than inventing parts. `titles=` with times marks parts
  anyway.
- **A card needs a pause.** A quick talker who never stops for a second gets the small titles under
  the rail. To leave room for cards, pause for a beat between parts when you film.
- **Close shots get smaller cards.** When your face fills most of the frame, the only room is below
  your chin, and the card is made smaller to fit there.
- **Run it after `clean-cut`, never before**, because cutting changes the times. And it is one more
  thing over the picture: run it on a video without captions or other graphics, since it cannot see
  what an earlier run drew.
- **English only**, like the rest of the pack.

## In AskEditor

There, each part can open with its number and name, or with the list of parts and the current one
lit, as graphics on the timeline: retitle a part, move where it begins or change the look on the
canvas without rendering again.
