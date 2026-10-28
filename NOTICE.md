# Notices

The code in this pack is written for it and is MIT licensed (`LICENSE`). No code from any other
project is copied or adapted into it. If that ever changes, the adapted lines and their licence are
listed here.

## Sounds (in `sounds/`)

The eight sound effects are made by `sounds/make_sounds.py`, from noise and sine waves shaped in
code. No recording, sample or sound library from anywhere else is in them, and they are part of this
pack under its MIT licence. The pack ships no music: `music-bed` plays the track a person brings.

A few sounds are made while a video renders, because they follow its own timing (a neon sign's hum
switches on and off with its letters): the light switch, the record scratch, the neon's hum and
crackle, the LED sign's ticks and the shimmer of light, in `fx/core/synth.py`. They are made the
same way, from noise and sine waves, with no recording in them.

## Fonts (in `fonts/`, each with its licence beside it)

All seven come from Google Fonts' own repository (`github.com/google/fonts`, commit
`ddf386ba1b9fbc6b2c818f1ddcd9148f54ced94b`; VT323 at commit `9862abc066c9b83bf84c3e8ed1fc2d8d48f9bbf0`) and are licensed under the SIL Open Font License,
Version 1.1. The full licence is the `OFL.txt` in each font's folder.

| Font | Files | Copyright |
|---|---|---|
| Anton | `Anton-Regular.ttf` | Copyright 2020 The Anton Project Authors |
| DM Serif Display | `DMSerifDisplay-Regular.ttf`, `DMSerifDisplay-Italic.ttf` | Copyright 2014-2018 Adobe, with Reserved Font Name 'Source'; Copyright 2019 Google LLC |
| Instrument Serif | `InstrumentSerif-Regular.ttf`, `InstrumentSerif-Italic.ttf` | Copyright 2022 The Instrument Serif Project Authors |
| Poppins | `Poppins-SemiBold.ttf`, `Poppins-Bold.ttf`, `Poppins-ExtraBold.ttf`, `Poppins-Black.ttf` | Copyright 2020 The Poppins Project Authors |
| Caveat Brush | `CaveatBrush-Regular.ttf` | Copyright 2015 Google Inc. All Rights Reserved. |
| VT323 | `VT323-Regular.ttf` | Copyright 2011, The VT323 Project Authors (peter.hull@oikoi.com) |
| IBM Plex Mono | `IBMPlexMono-Medium.ttf`, `IBMPlexMono-Bold.ttf` | Copyright © 2017 IBM Corp. with Reserved Font Name "Plex" |

## Models (fetched by `setup.sh` into `models/`; not part of this repository)

Each file is downloaded from its official home at a pinned version and checked against the SHA-256
recorded in `fx/core/models.py`.

| Model | What it does here | From | Licence |
|---|---|---|---|
| BlazeFace (short range) | finds the face in each frame | MediaPipe model storage (`storage.googleapis.com/mediapipe-models`) | Apache License 2.0 (its model card) |
| Selfie Segmenter | the person mask | MediaPipe model storage | Apache License 2.0 (its model card) |
| Selfie Multiclass 256×256 | the detailed person mask (hair, skin, clothes), optional | MediaPipe model storage | Apache License 2.0 (its model card) |
| Face Landmarker | where each mouth is, to tell who is talking when two people are in the picture | MediaPipe model storage | Apache License 2.0 (its model card) |
| faster-whisper small.en | word timings from speech | Hugging Face, `Systran/faster-whisper-small.en`, revision `d1d751a5f8271d482d14ca55d9e2deeebbae577f` | MIT (its model card). It is a conversion of OpenAI's Whisper small.en, which is also MIT licensed |

## Python packages (installed by `setup.sh` into `.venv`; not part of this repository)

`requirements.txt` pins them. The ones the pack uses directly are faster-whisper (MIT), CTranslate2
(MIT), MediaPipe (Apache License 2.0), OpenCV (Apache License 2.0), NumPy (BSD) and Pillow
(MIT-CMU). Everything they bring with them keeps its own licence.

## Programs (installed by the person, not by this pack)

ffmpeg and ffprobe read and write the video. They are run as separate programs and are not
distributed with this pack.
