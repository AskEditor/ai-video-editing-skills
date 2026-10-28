"""Video in and out through ffmpeg pipes, keeping the clip's frame rate and its sound.

Frames are RGB `uint8` arrays of shape (height, width, 3). Frame `i` is the picture at `i / fps`
seconds: every reader runs the clip through ffmpeg's `fps` filter at the clip's own rate, so a
clip with a dropped or doubled frame still reads as a steady run, and the frame count always
matches the sound.

A `VideoInfo` can also stand for a picture that is drawn rather than read from its file: the
canvas of a layout, as an effect laid inside it sees it (`view.py`). Its `picture` gives frame `i`;
`read_frame` asks it, and `FrameReader` refuses it in plain words.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from . import hdr

# On a busy machine, FX_THREADS caps the threads each ffmpeg (and OpenCV) uses; unset, they choose.
THREADS = os.environ.get("FX_THREADS", "").strip()


def threads() -> list[str]:
    return ["-threads", THREADS, "-filter_threads", THREADS] if THREADS.isdigit() else []


if THREADS.isdigit():
    try:
        import cv2

        cv2.setNumThreads(int(THREADS))
    except ImportError:
        pass

AUDIO_RATE = 48000
AUDIO_CHANNELS = 2


class MediaError(RuntimeError):
    """ffmpeg could not read or write something; the message says what in plain words."""


def need_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError("ffmpeg is not installed. Run setup.sh: it says how to get it.")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int  # as displayed, after any rotation flag
    height: int
    fps: Fraction
    duration: float  # seconds of picture
    frames: int  # frames at `fps` across `duration`
    has_audio: bool
    audio_offset: float  # when the sound starts, relative to the first frame (seconds)
    color_matrix: str  # how the clip's colours are coded: "bt709", "bt601" or "bt2020"
    transfer: str = "sdr"  # "hlg" or "pq" for HDR footage, which every reader turns into SDR (hdr.py)
    # A picture drawn rather than read from `path` (a layout's canvas): frame number -> RGB frame.
    picture: Optional[Callable[[int], np.ndarray]] = field(default=None, compare=False, repr=False)

    @property
    def rate(self) -> float:
        return float(self.fps)

    def time(self, frame: int) -> float:
        return frame / self.rate

    def frame_at(self, t: float) -> int:
        return int(min(self.frames - 1, max(0, round(t * self.rate))))

    @property
    def aspect(self) -> float:
        return self.width / self.height


def _fraction(text: str | None) -> Fraction:
    try:
        num, _, den = (text or "0/1").partition("/")
        f = Fraction(int(num), int(den or 1))
        return f if f > 0 else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def probe(path: str | os.PathLike) -> VideoInfo:
    need_ffmpeg()
    path = str(path)
    if not Path(path).exists():
        raise MediaError(f"there is no file at {path}")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise MediaError(f"this file could not be read as a video: {path}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise MediaError(f"this file has no picture: {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if not rotation and "rotate" in video.get("tags", {}):
        rotation = int(video["tags"]["rotate"])
    if abs(rotation) % 180 == 90:
        width, height = height, width

    fps = _fraction(video.get("r_frame_rate"))
    avg = _fraction(video.get("avg_frame_rate"))
    if fps == 0 or fps > 240 or (avg > 0 and fps > 2 * avg):
        fps = avg
    if fps == 0:
        raise MediaError(f"could not tell the frame rate of {path}")

    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise MediaError(f"could not tell how long {path} is")
    v_start = float(video.get("start_time") or 0)
    a_start = float(audio.get("start_time") or 0) if audio else 0.0

    space = (video.get("color_space") or "").lower()
    transfer = hdr.kind(video.get("color_transfer"))
    if space in ("smpte170m", "bt470bg", "bt601"):
        matrix = "bt601"
    elif space in ("bt2020nc", "bt2020c") or transfer != "sdr":
        matrix = "bt2020"
    elif space in ("bt709",) or height >= 720 or width >= 1280:
        matrix = "bt709"
    else:
        matrix = "bt601"

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=max(1, int(round(duration * float(fps)))),
        has_audio=audio is not None,
        audio_offset=a_start - v_start,
        color_matrix=matrix,
        transfer=transfer,
    )


def _fps_text(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


class FrameReader:
    """Frames of a clip, one after another, optionally scaled, from `start` seconds for `duration`.

    `extra` is more of ffmpeg's filters, run on the picture once it is RGB at its final size (a 3D
    LUT, say: `lut3d=file=look.cube`); they must keep it in the same format. `cwd` is the folder
    ffmpeg runs in, so a filter can name its file by a bare name: a full path inside a filter needs
    escaping that differs between systems (a Windows drive's colon breaks it).

    HDR footage (HLG or PQ) comes out as ordinary SDR, converted before anything else sees it
    (`hdr.py`), so every effect, the face finder and the person mask see the picture as a phone
    shows it rather than flat and grey.
    """

    def __init__(
        self,
        info: VideoInfo,
        size: Optional[tuple[int, int]] = None,
        start: float = 0.0,
        duration: Optional[float] = None,
        gray: bool = False,
        extra: Optional[str] = None,
        cwd: Optional[str | os.PathLike] = None,
    ):
        if info.picture is not None:
            raise MediaError("this picture is drawn by the layout it sits in, so it cannot be streamed from a "
                             "file: lay this effect on the clip before the layout, or run it on the layout's result")
        self.info = info
        self.width, self.height = size or (info.width, info.height)
        self.channels = 1 if gray else 3
        self.frame_bytes = self.width * self.height * self.channels
        cmd = ["ffmpeg", "-v", "error", "-nostdin", *threads()]
        if start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        cmd += ["-i", str(Path(info.path).resolve())]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        scale = f"scale={self.width}:{self.height}:flags=bicubic+accurate_rnd+full_chroma_int"
        scale += f":in_color_matrix={info.color_matrix}"
        fmt = "gray" if gray else "rgb24"
        if info.transfer in ("hlg", "pq"):
            to_sdr, cwd = hdr.reader_filter(info.transfer, cwd)
            vf = f"fps={_fps_text(info.fps)},{scale},format=rgb48le,{to_sdr},format={fmt}"
        else:
            vf = f"fps={_fps_text(info.fps)},{scale},format={fmt}"
        if extra:
            vf += f",{extra},format={fmt}"
        cmd += ["-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-pix_fmt", fmt, "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_bytes * 2,
                                      cwd=str(cwd) if cwd else None)

    def __iter__(self) -> Iterator[np.ndarray]:
        assert self._proc.stdout is not None
        shape = (self.height, self.width) if self.channels == 1 else (self.height, self.width, 3)
        try:
            while True:
                buf = self._proc.stdout.read(self.frame_bytes)
                if len(buf) < self.frame_bytes:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(shape)
        finally:
            self.close()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()


def read_frame(info: VideoInfo, index: int, size: Optional[tuple[int, int]] = None,
               extra: Optional[str] = None, cwd: Optional[str | os.PathLike] = None) -> np.ndarray:
    """One frame by its number, at the clip's size or `size` (`extra` and `cwd` as for FrameReader).
    A drawn picture (`info.picture`) is asked for the frame instead."""
    index = max(0, min(info.frames - 1, index))
    if info.picture is not None:
        if extra:
            raise MediaError("a picture drawn by a layout cannot be run through more filters")
        frame = np.asarray(info.picture(index))
        if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
            import cv2

            frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame, np.uint8).copy()
    # Seek a quarter of a frame early: the first frame at or after that time is frame `index`.
    start = max(0.0, (index - 0.25) / info.rate)
    for frame in FrameReader(info, size=size, start=start, duration=2.0 / info.rate + 0.5, extra=extra, cwd=cwd):
        return frame.copy()
    # Past the last decodable frame: fall back to the last one there is.
    frames = list(FrameReader(info, size=size, start=max(0.0, info.duration - 1.0), extra=extra, cwd=cwd))
    if not frames:
        raise MediaError(f"could not read frame {index} of {info.path}")
    return frames[-1].copy()


def read_audio(info: VideoInfo, rate: int = AUDIO_RATE, channels: int = AUDIO_CHANNELS) -> np.ndarray:
    """The clip's sound as float32 samples of shape (n, channels), lined up with frame 0 and
    exactly as long as the picture. A clip with no sound gives silence."""
    n = int(round(info.frames / info.rate * rate))
    if not info.has_audio:
        return np.zeros((n, channels), np.float32)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", info.path, "-map", "0:a:0", "-ac", str(channels),
         "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    if out.returncode != 0:
        raise MediaError(f"could not read the sound of {info.path}")
    samples = np.frombuffer(out.stdout, np.float32).reshape(-1, channels)
    shift = int(round(info.audio_offset * rate))
    if shift > 0:
        samples = np.concatenate([np.zeros((shift, channels), np.float32), samples])
    elif shift < 0:
        samples = samples[-shift:]
    if len(samples) < n:
        samples = np.concatenate([samples, np.zeros((n - len(samples), channels), np.float32)])
    return np.ascontiguousarray(samples[:n])


def mono(samples: np.ndarray, rate: int, to_rate: int = 16000) -> np.ndarray:
    """Mix down to one channel and resample by simple averaging (used only for analysis)."""
    x = samples.mean(axis=1) if samples.ndim == 2 else samples
    if rate == to_rate:
        return x.astype(np.float32)
    n_out = int(len(x) * to_rate / rate)
    pos = np.arange(n_out) * (rate / to_rate)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


class FrameWriter:
    """Writes RGB frames to an MP4 (H.264, the clip's frame rate) with the given sound (AAC).

    The file is written under a temporary name and moved into place only when it is complete, so a
    render that stops halfway never leaves a broken file with the final name.
    """

    def __init__(
        self,
        out_path: str | os.PathLike,
        width: int,
        height: int,
        fps: Fraction,
        audio: Optional[np.ndarray] = None,
        audio_rate: int = AUDIO_RATE,
        crf: int = 18,
        preset: str = "medium",
        audio_kbps: int = 192,
    ):
        need_ffmpeg()
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.frames = 0
        self._tmpdir = tempfile.mkdtemp(prefix="fx-")
        self._partial = Path(self._tmpdir) / ("partial" + self.out_path.suffix)
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-framerate", _fps_text(fps), "-i", "-"]
        self._audio_file = None
        if audio is not None and len(audio):
            self._audio_file = Path(self._tmpdir) / "sound.f32"
            np.ascontiguousarray(audio, np.float32).tofile(self._audio_file)
            cmd += ["-f", "f32le", "-ar", str(audio_rate), "-ac", str(audio.shape[1]), "-i", str(self._audio_file)]
        cmd += ["-map", "0:v:0"]
        if self._audio_file:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", f"{int(audio_kbps)}k"]
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                *(["-threads", THREADS] if THREADS.isdigit() else []),
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
                "-flags:v", "+bitexact", "-flags:a", "+bitexact", str(self._partial)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"frame is {frame.shape} {frame.dtype}, expected ({self.height}, {self.width}, 3) uint8")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as e:
            raise MediaError(self._error()) from e
        self.frames += 1

    def _error(self) -> str:
        err = (self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else "").strip()
        return "the video could not be written" + (f": {err.splitlines()[-1]}" if err else "")

    def close(self) -> Path:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        code = self._proc.wait()
        try:
            if code != 0:
                raise MediaError(self._error())
            shutil.move(str(self._partial), str(self.out_path))
            return self.out_path
        finally:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def save_jpeg(path: str | os.PathLike, frame: np.ndarray, quality: int = 90) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path, quality=quality)
    return path
