# Social frame

Your video plays full screen, then pulls back into a frame and keeps playing there: a post card, a
phone, three phones fanned out, or the middle tile of a grid of your posts. Your name, a line and
your counts sit round it, only if you type them.

```
/video-editing social-frame my-talk.mp4 name=yourname
/video-editing social-frame my-talk.mp4 look=phone name=yourname caption="Behind the scenes"
/video-editing social-frame my-talk.mp4 look=fan pictures=~/Movies/my-last-reels
/video-editing social-frame my-talk.mp4 look=grid name=yourname caption="Videos worth posting" at="here's the thing"
/video-editing social-frame my-talk.mp4 look=card likes=1,204 comments=86 duration=4
```

## The looks

| `look` | What it is |
|---|---|
| `card` (default) | A light card with your clip in it: your face in a small circle and your name above, your counts and a line under it. |
| `phone` | A graphite phone of our own design, with a camera dot and side buttons; your name and line under it. |
| `fan` | Three phones: yours in the middle, two more fanning out from behind it as it lands. |
| `grid` | A three-by-three grid of upright tiles, yours in the middle with a thin light ring; your name and line above. |

None of them is any app's screen: no app's layout, icons, logo or colours. The heart and the
speech bubble by the counts are plain shapes.

## Options

| Option | Default | What it does |
|---|---|---|
| `look` | `card` | `card`, `phone`, `fan` or `grid`. |
| `at` | `auto` | When it pulls back: `auto` (the start of a strong line in the first half of the clip), `start`, a time in seconds, or words you say. |
| `duration` | `0` | Seconds it stays framed before the picture fills the screen again; `0` keeps it framed to the end. |
| `name` | (empty) | Your name or handle, as you type it. |
| `caption` | (empty) | One line under the card or the phone (on the grid, the line under your name). |
| `likes`, `comments` | (empty) | Counts to show, exactly as typed. Card and phone only. |
| `pictures` | (empty) | For `fan` and `grid`: a folder, or files separated by `;`, of your own pictures or clips for the other phones and tiles. |
| `background` | `auto` | The ground behind the frame: your clip's own colour, made deep and quiet, or a colour. |
| `captions` | `off` | `on` lays the pack's captions over the finished frame, clear of your face; or name a caption look. |
| `sound` | `on` | A soft whoosh under your voice as it pulls back (and as the phones or the tiles land). |
| `fix` | (empty) | Misheard words, `wrong=right;…`, for `at` and the captions. |

## How it works

The whole frame is laid out once at full size, with your clip's rectangle where it plays. The
camera starts framed tight on that rectangle, so the first frame is your video exactly as shot,
and eases back to the whole layout. The clip is drawn from its own full-size pixels at every step,
so it never goes soft as it shrinks. The words fade in as it lands, outside the picture and inside
the part of the screen the app leaves free.

## Honest limits

- **Nothing is made up.** No name, line or count appears unless you type it; with none, the frame
  carries no words. A count is shown exactly as typed, so type the real one.
- **The other phones and tiles need your pictures.** Without `pictures=`, they show stills of this
  same clip, each toned differently, and the run says they are stand-ins. A grid of one moment of
  you reads as a grid of one moment.
- **Your clip gets smaller.** The card and the phone show it at about half the width, the grid at
  under a third. Fine print in the video itself will not read there.
- **Upright clips.** The frame takes your clip's own shape; a wide clip makes a wide card.
- **The fan and the grid reach the edges of the screen.** They are pictures, so they may; the
  words never do.
