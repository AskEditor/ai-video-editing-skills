# Notifications

Message notifications pop in one after another while you talk, each landing on a word. The newest
sits on top and the ones before slide down a place. The cards are our own design, never a copy of
a phone's, and they never sit on your face.

```
/video-editing notifications my-talk.mp4
/video-editing notifications my-talk.mp4 sender=@myhandle count=4
/video-editing notifications my-talk.mp4 items="Is the edit done?; Posting at 6@4.2"
/video-editing notifications my-talk.mp4 items=messages.json style=solid
```

## What they say

- **Your own phrases.** Leave `items` empty and the cards carry the short phrases you say that
  stand best as a line by themselves (`count` of them, three by default), chosen by the same
  picker every effect in the pack uses: never a word you took back, a first attempt, an "um" or an
  aside such as "testing, testing".
- **Your messages.** `items="Is the edit done?; Posting at 6@4.2"`, separated by `;`, each with
  a time in seconds after `@` if you like. Or a `.json` file: a list of `{"text": …, "at": …,
  "from": …}`, where `from` names that card's sender.
- **The sender.** `sender=` is written on every card (your name, a handle, a brand). Left on auto
  it says "New message".
- A message too long for two lines is shortened by whole words, and the run says which.

## When each lands

Each card is whole and still on the first frame of its word, and drops in over the moment before
with a soft chime. A phrase you said lands on its first word; a message with a time lands on the
word starting there; a message without one lands where its words are said, else where one of your
lines starts (the run says so). Two cards never land on top of each other: a card waits for the
one before it to settle.

## The stack

The newest is on top. Each time a card arrives the ones before slide down a place; when more
arrive than the room holds (`most`, three by default), the oldest fades as it slides out. The whole
stack goes `hold` seconds (2.2) after the last card lands.

## Where it sits

Above your head or under your chin, wherever the stack has most room over every frame it is up,
inside the part of the frame the app's own buttons and caption leave free, shadows included. When
the room holds fewer cards than `most` at a size that reads on a phone, fewer are shown at once
and the run says so. `position=above` or `below` picks a side.

## The card

Frosted dark glass, your video blurred behind it and smoked just enough that white type reads
(`style=solid` is a plain dark card for a busy background). A rounded tile in the accent colour
(`accent=`, coral by default) with a speech mark of our own, the sender, "now", and up to two lines.

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | | your messages, `;` between them, `@seconds` optional, or a `.json` file |
| `count` | `3` | how many of your phrases to take when `items` is empty |
| `sender` | `auto` | who the cards are from |
| `style` | `glass` | `glass` or `solid` |
| `position` | `auto` | `above` or `below` |
| `most` | `3` | the most cards on screen at once |
| `accent` | `auto` | the tile's colour |
| `hold` | `2.2` | seconds the stack stays after the last card |
| `sound` | `on` | a soft chime as each card lands, under your voice |
| `fix` | | misheard words, `wrong=right` |

## Honest limits

- The phrases it picks are the strongest short lines it can find, not always the ones you would
  choose; give `items` to choose.
- When your head moves a lot, the room is what stays free over the whole stretch, so a close or
  moving shot may hold only one card at a time.
- Pictures and replies are not drawn: a card is a sender and words.
