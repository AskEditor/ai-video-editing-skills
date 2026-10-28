"""One clip, with everything about it found once and kept: its picture, sound, words, face and person."""

from __future__ import annotations

import os
from functools import cached_property
from typing import Optional

import numpy as np

from . import audio, video
from . import face as facemod
from . import person as personmod
from . import words as wordsmod


class Clip:
    def __init__(self, path: str | os.PathLike, fix: Optional[str] = None):
        self.path = str(path)
        self.info = video.probe(self.path)
        self.fix = fix or ""

    # ---- sound ----
    @cached_property
    def sound(self) -> np.ndarray:
        """The clip's sound at 48 kHz stereo, lined up with frame 0 and as long as the picture."""
        return video.read_audio(self.info)

    @cached_property
    def envelope(self) -> audio.Envelope:
        return audio.envelope(video.mono(self.sound, video.AUDIO_RATE, 16000), 16000)

    # ---- words ----
    def _quiet_for(self, heard: list[wordsmod.Word]) -> list[tuple[float, float]]:
        middles = [((w.start + w.end) / 2 - 0.03, (w.start + w.end) / 2 + 0.03) for w in heard]
        return audio.quiet_runs(self.envelope, protect=middles)

    @cached_property
    def heard(self) -> list[wordsmod.Word]:
        """Every word heard, times as heard: the transcriber's words, with the stretches it could
        not explain heard a second time (`words.relisten`) and judged (`words.merge`), and any
        corrections applied. What the second listen heard but could not be sure of is here too,
        marked `unsure`; nothing reads it but a report (`words` leaves it out)."""
        first = wordsmod.load(self.path)
        quiet = self._quiet_for(first)
        spans = wordsmod.doubtful_spans(wordsmod.timed(first, quiet), quiet, self.info.frames / self.info.rate)
        words = wordsmod.merge(first, wordsmod.relisten(self.path, spans), self.envelope) if spans else first
        self.second_listen = spans
        return wordsmod.apply_fixes(words, self.fix)

    @cached_property
    def unsure(self) -> list[wordsmod.Word]:
        """The words heard but not used, each with why (`Word.unsure`): never shown or cut on."""
        return [w for w in self.heard if w.unsure]

    @cached_property
    def quiet(self) -> list[tuple[float, float]]:
        """Stretches where nobody is speaking, from the sound. An unsure word never keeps a
        stretch from counting as quiet."""
        return self._quiet_for(wordsmod.sure(self.heard))

    @cached_property
    def words(self) -> list[wordsmod.Word]:
        """The words as effects use them: the sure ones, corrected for the transcriber's lateness
        and checked against the sound (`words.timed`)."""
        return wordsmod.timed(self.heard, self.quiet)

    # ---- picture ----
    @cached_property
    def face(self) -> facemod.FaceTrack:
        return facemod.track(self.info)

    def person(self, model: str = personmod.MODEL) -> personmod.PersonMasks:
        return personmod.masks(self.info, model)
