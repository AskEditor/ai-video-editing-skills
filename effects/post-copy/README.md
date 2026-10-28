# Post copy

The words you post with your video, written from what you say in it: a title and a caption for
Reels, TikTok, Shorts, YouTube and LinkedIn, the hashtags, YouTube chapters with their times, and a
short post to share it anywhere. Nothing is rendered and nothing is posted.

```
/video-editing post-copy my-talk.mp4
/video-editing post-copy my-talk.mp4 platforms="reels, youtube"
/video-editing post-copy my-talk.mp4 titles="Why I left | 0:42 What changed | 1:30 What I'd do again"
```

## How it works

The pack does the part a computer can be sure of. Your agent (Claude or Codex, whichever is running
the pack) does the writing, and the pack checks it.

1. **`post-copy` writes `<video>.post.md`** beside your video. In it:
   - **what you say, in its parts**, each with its times and a title of its key words: the same
     parts `chapter-cards` finds (where the talk moves on, from your pauses, signposts and words);
   - **a brief**: the video's length and shape, the line that makes the point (picked the way
     `headline` picks its line) and the strongest line of each part, the key words, the names and the
     numbers heard, and how to write the copy;
   - **each platform's limits**, as its own help pages gave them on the day they were read, with
     the pages named. A limit no platform publishes is not listed and not checked;
   - **YouTube's chapter lines**, exactly as YouTube takes them: `00:00` first, at least three, in
     order, each part 10 seconds or more. A video that cannot hold three such parts gets none, and
     the file says why;
   - **a block to fill for each piece of copy**, each between `<!-- … -->` markers.
2. **The agent writes the copy** into each block, from the brief and your words.
3. **`check=on` checks it**: every block written, each within its platform's limits, counted the
   way that platform counts, the chapter lines in the YouTube description as they are. Anything
   wrong is named, and the command ends with an error until it is right.

## For the agent writing the copy

Run it, open the file it names after `post:`, and read the parts and the brief first. Then replace
each `FILL:` line with your own copy, keeping the markers, and check:

```
.venv/bin/python -m fx.run post-copy "/full/path/my-talk.mp4"
.venv/bin/python -m fx.run post-copy "/full/path/my-talk.mp4" --opt check=on
```

- **Only what the speaker says or shows.** No claim, number, price, result or promise they did not
  make. Their own phrases usually beat marketing ones.
- **Titles say what a viewer gets**, plainly. Reels, TikTok and LinkedIn have no title field, so the
  title there is the caption's first line (the check notes when a caption does not start with it).
- **Names:** the brief lists the names heard; the transcriber spells these worst, so check each
  against what the person told you, and use `fix=` on the run if one is wrong.
- **YouTube chapters:** put the lines into the YouTube description exactly as written. A title may be
  reworded in the chapters block; the times stay.
- **Hashtags:** only for what the video is about, most specific first, one word each (no spaces, $
  or %). Reels takes 5 at most.
- **Fix every `FAIL` line and check again** until the verdict says every piece is within its limits.
  Then show the person the copy, or the file.

A file that already has copy in it is never written over: the run stops and says so. Move it away to
start again.

## Options

| Option | Default | What it does |
|---|---|---|
| `titles` | `auto` | The parts' titles: `auto` (their key words), a number of parts, or your own titles between `\|` marks, each may begin with its time. Give `chapter-cards` the same titles and the chapters match the cards. |
| `min_part` | `auto` | The shortest a chapter may be, in seconds: `auto` is 10 (YouTube's least) or a tenth of a long video. |
| `platforms` | `all` | Which to write for: any of `reels`, `tiktok`, `shorts`, `youtube`, `linkedin`. |
| `check` | `off` | `on` checks the copy already written into the file, instead of writing it. |

## The limits it checks, and where they come from

Read from each platform's own pages on 28 October 2026. They change: when a platform changes one,
change it in `fx/effects/post_copy.py`, with the page and the day.

| Where | What is checked | Read from |
|---|---|---|
| Instagram Reels | the caption up to 2,200 characters, 5 hashtags, 20 @ mentions | Instagram Help Center, "Use hashtags on Instagram" (5 hashtags); Instagram Platform docs, "IG User Media" (2,200 characters, 20 mentions; it still says 30 hashtags, so the app's 5 is kept) |
| TikTok | the caption up to 2,200, counted in UTF-16 units | TikTok for Developers, Content Posting API, "Direct Post" |
| YouTube and Shorts | the title up to 100 characters, the description up to 5,000 characters and 5,000 bytes, no `<` or `>`; more than 60 hashtags and YouTube ignores them all | YouTube Help, "Edit video settings"; YouTube Data API, "Videos"; YouTube Help, "Find playlists & videos using hashtags" |
| YouTube chapters | the first at 00:00, at least three, in order, each 10 seconds or more | YouTube Help, "Video Chapters" |
| LinkedIn | the post up to 3,000 characters | LinkedIn Help, "Post and share updates" |
| A short post | 280 as X counts them: a link 23, an emoji 2, CJK and most other scripts 2 | X Developer Platform, "Counting characters", and the configuration it links to |

## Honest limits

- **The pack does not write the copy itself.** It finds the parts, the times, the lines and the
  limits; the writing is your agent's. Without an agent, the file is a brief and an empty form.
- **The chapter titles are a first guess** from each part's key words. Reword them in the file, or
  give `titles=`.
- **Short videos get no chapters.** YouTube takes chapters only when there are three parts of 10
  seconds or more; a reel of 30 seconds rarely has them.
- **It cannot see how much of a caption shows** before the app folds it away: no platform publishes
  that, so nothing is checked against it. Keep the first line strong.
- **English only**, like the rest of the pack.
- **It posts nothing.** It never signs in to any platform.

## In AskEditor

The copy can live with the video there, changed per platform, and each post scheduled from the same
place.
