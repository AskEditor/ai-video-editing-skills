# This versus that

A small grid above your head with two columns. Say "editing by hand versus editing with a tool" and the
two sides land as its heads, each on its first word. Say a run of rules ("If your hook is slow, cut the
first second. If nobody comments, ask a question.") and an if/then grid fills row by row as you say each
part, the row you are on lit. "Before, … Now, …" makes a before/after grid. Or give the cells yourself.

## What to type

```
/video-editing comparison-grid my-talk.mp4
/video-editing comparison-grid my-talk.mp4 items="Phone | Camera; Cheap | Costly; In your pocket | In a bag"
/video-editing comparison-grid my-talk.mp4 look=if-then items="If | Then; Slow hook | Cut the first second"
```

With `items=`, rows are split by `;` and the two cells of a row by `|`. The first row is the two heads.
Each cell lands where its words are said; one you never say drops in between the others and the run
tells you, or give it a time: `Costly@6.2`.

## Options

| Option | Default | What it does |
|---|---|---|
| `items` | (heard) | the cells, as above, or a `.json` file of rows |
| `look` | auto | versus (VS between the heads), if-then, before-after, or table |
| `title` | | a heading across the top |
| `position` | auto | top, side or below; never over your face whichever you pick |
| `style` | glass | glass (frosted, your video blurred behind), solid or paper |
| `accent` | auto | the colour of the VS, the labels and the row being said |
| `leave` | 0 | seconds the grid stays after its last cell; 0 keeps it to the end |
| `hold` | 0 | seconds the last frame is held at the end |
| `sound` | on | a soft pop under your voice as each cell lands |
| `frame`, `size`, `fix` | | as in every effect |

Run with `--still` first to see the grid empty, filling and full.

## Honest limits

- It hears three ways of saying a comparison: "X versus Y" (or "compared to"), "if …, then …" (or a
  question followed by "Then …"), and "Before, … Now, …". Anything else needs `items=`. A versus said once
  gives the two heads only; the rows under them come from `items=`.
- A cell is a few words. A longer one goes onto two lines, then is shortened by whole words, and the run
  says which. Four rows at most.
- On a close shot with no room above the head it goes under the chin, smaller.
- The grid keeps clear of your face in every frame; where your hair rises into it, you are laid back over it.

To change it without rendering again, open it in AskEditor and ask "Add a comparison grid": retype a cell,
add a row or move it on the canvas.
