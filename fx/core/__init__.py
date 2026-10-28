"""The core every effect shares.

- video.py:   frames and sound in and out through ffmpeg pipes, at the clip's own frame rate.
- hdr.py:     HDR footage (HLG or PQ, what phones film) turned into ordinary SDR as it is read.
- audio.py:   the speech envelope that finds pauses, and joins that never click.
- sound.py:   a track read in, loudness (BS.1770) in numpy, a sound laid on a moment, where the voice talks.
- loudness.py: a finished mix measured and set with ffmpeg's two-pass loudnorm, peaks limited first.
- soundsheet.py: what a sound effect did, as a picture: frames over a lane per sound, and close-ups.
- words.py:   word times, cached next to the clip as JSON.
- moments.py: the word or line an effect sets by itself: what the speaker took back (a word replaced,
              a first attempt said again, an aside) is never it, and how strong each word and line is.
- asides.py:  the sentences where the speaker talks about the recording ("testing, testing", "give
              me a second"), not to the viewer: clean-cut cuts them, and no picker sets one.
- amounts.py: the amounts said in the words, or typed, each with its unit, and how a card writes them.
- counting.py: a number that counts: figures that change every frame, when a figure said begins, and
              soft ticks under the count (chart and number-counter).
- items.py:   the things an effect shows one by one (cards, rows, steps), given or found in the words,
              when each lands, lists said in order, steps said done, and an item made shorter by whole words.
- parts.py:   the parts of a talk: where the speaker moves on to a new idea, each with a title.
- match.py:   a person's own files matched to the words said: a map, the words in file names, a
              tags.txt, or the strongest things named, in folder order; a brand's logo on its whole name.
- face.py:    a face box per frame, smoothed, cached next to the clip.
- person.py:  a person mask per frame, soft-edged, cached next to the clip.
- matte.py:   the person over a whole new room: a firm edge with no rim of the old room, held
              steady, two models agreeing; `Cutout` makes it for a render frame, `face_shown` checks it.
- shots.py:   the clip's own cuts, where the picture jumps from one shot to the next.
- layout.py:  the safe zone and placing text clear of the face.
- text.py:    fonts and drawing words.
- phrasing.py: which words share a caption card, cut where a reader expects; every caption uses it.
- compose.py: laying images over frames, and behind the person.
- depth.py:   pictures laid in depth in the room behind the person: cards in perspective, their soft
              shadows, how much of each the person leaves seen, and the room's drift with the head.
- glass.py:   frosted glass panes: what is behind blurred and bent at the rim, toned so type reads, edge-lit.
- timeline.py: which source seconds are kept, where they land in the output, and where the picture
              stops for a moment (a freeze) before playing on.
- framing.py: the output's shape and size, and a window that follows the face calmly (it holds, then
              glides; it cuts where the clip cuts); or the whole picture fitted over a blurred copy.
- speakers.py: who is talking when two people are in the picture: the mouth that moves with the sound.
- clip.py:    one clip with everything above found once and kept.
- view.py:    the clip as a layout shows it (a canvas or one panel), for effects laid inside the layout.
- glitch.py:  a broken signal torn over a frame (bands, blocks, split colours, scan lines) and its
              crackle; shared by transitions and glitch.
- zoom.py:    a closer framing on the face (the whole head kept), how it moves, and the picture in it.
- render.py:  the whole video, or a few stills, drawn by the same code.
- models.py:  the model files, pinned by checksum; paths.py: where things live.
"""
