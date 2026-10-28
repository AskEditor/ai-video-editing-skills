# Stickers

Stickers that slap on with the word: a rubber stamp, a price tag, a starburst or an emoji-style
badge. Each lands on the frame you say its word, at a slight angle, beside your head and never on
your face.

```
/video-editing stickers my-talk.mp4
/video-editing stickers my-talk.mp4 at="forty nine dollars" kind=price text='$49'
/video-editing stickers my-talk.mp4 at="wrong|that's the fix" kind="stamp|badge" text="WRONG|check"
/video-editing stickers my-talk.mp4 at=2.1,5.4 kind=starburst
```

## Which words (`at`)

- **`auto`** (default): the strongest words of your take, any money or percentage you say, and
  words of feeling ("love", "wrong", "wow"), at least two seconds apart: about one for every five
  seconds, three at most (`count=` for more). A word you took back, a first attempt or an aside is
  never used.
- **Words you say**, several with `|`: one sticker on each, where you say it.
- **Times** in seconds (`2.1,5.4`): each moved onto the word you start nearest that moment.

## The kinds (`kind`)

| `kind` | What it is | `text` |
|---|---|---|
| `stamp` | Tall capitals in a double frame, inked unevenly, like a rubber stamp. | the word (default: the one you say) |
| `price` | A swing tag with a hole, the amount in heavy figures. | the price (default: the amount you say) |
| `starburst` | A sixteen-point burst with the word across it. | the word |
| `badge` | A round emoji-style badge drawn here: `heart`, `fire`, `wow`, `laugh`, `check`, `cross` or `star`. | the badge's name |
| `auto` | Chosen by the word: an amount gets a price tag, a feeling a badge, "new", "free" or a number a starburst, a verdict ("wrong", "done") a stamp. | |

`kind` and `text` take one value for all, or one per sticker with `|`. Every sticker is drawn in
code: nothing is downloaded, and no app's emoji art is used.

## How it lands

It drops from a little larger over three frames, squashes on the frame of its word, settles, and
pops away after `hold` seconds (1.6). A slap made in code sits under your voice on that frame (a
heavier thud for a stamp); `sound=off` keeps your sound exact. Each spot in the safe zone is
weighed and the one kept is clear of your head on every frame (its drop from larger included), of
the captions (`captions=stacked`) and of the other stickers, near your head, alternating sides.

## Options

| Option | Default | What it does |
|---|---|---|
| `at` | `auto` | `auto`, words you say (`a\|b`), or times (`2.1,5.4`). |
| `kind` | `auto` | `stamp`, `price`, `starburst`, `badge`, or one per sticker. |
| `text` | (empty) | What each says, one per sticker with `\|`. |
| `color` | `auto` | One colour for every sticker; auto gives each kind its own. |
| `count` | `0` | How many `at=auto` places (0: about one per five seconds, three at most). |
| `hold` | `1.6` | Seconds each stays. |
| `captions` | `off` | A caption look; stickers keep clear of it. |
| `sound` | `on` | `off` keeps your sound untouched. |
| `fix` | (empty) | Misheard words, `wrong=right; …`. |

## Honest limits

- **No price is made up.** A price tag says what you say or what you type; on a word with no
  amount, `kind=price` needs `text=`.
- **Auto words are the strongest, not always the funniest.** A sticker is a joke you choose; name
  the words with `at=` for the ones that matter.
- **Close shots leave little room**: stickers come out smaller (down to 55%), or are left out,
  and the run says so.
- **One face.** Someone else in the frame is not avoided.

## In AskEditor

The nearest there is the Emoji pops hook and the stamps inside the documentary moves, where a
sticker moves, swaps or retimes on the canvas without rendering again.
