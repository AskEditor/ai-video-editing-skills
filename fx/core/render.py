"""Running a clip through an effect: every output frame, or a few stills.

Both paths call the same `draw(frame, j, t)` on the same output frame, so a still is exactly the
frame the finished video shows at that moment (a preview drawn by other code is a preview of
something else).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio, video
from .clip import Clip
from .framing import Framing
from .timeline import Timeline

Draw = Callable[[np.ndarray, int, float], None]


HOLD_FADE = 0.12  # seconds the sound fades out over before a held last frame, so it never stops dead
FREEZE_FADE = 0.025  # seconds the sound fades out over into a freeze, and back in after it


def with_freezes(s: np.ndarray, rate: int, timeline: Timeline) -> np.ndarray:
    """The joined sound `s` with silence laid in for each freeze, faded out into it and back in
    after it, so the sound stops and starts with the picture and nothing clicks."""
    n = max(1, int(round(FREEZE_FADE * rate)))
    s = np.array(s, np.float32)
    pieces, last = [], 0
    for p, seconds, _ in timeline.freezes_in_play():
        cut = int(np.clip(round(p * rate), last, len(s)))
        m = min(n, cut - last)
        if m:
            s[cut - m:cut] *= np.cos(np.linspace(0, np.pi / 2, m, dtype=np.float32))[:, None]
        m = min(n, len(s) - cut)
        if m:
            s[cut:cut + m] *= np.sin(np.linspace(0, np.pi / 2, m, dtype=np.float32))[:, None]
        pieces += [s[last:cut], np.zeros((int(round(seconds * rate)), s.shape[1]), np.float32)]
        last = cut
    pieces.append(s[last:])
    return np.concatenate(pieces)


def sound_for(clip: Clip, timeline: Timeline, fade: float = 0.02) -> np.ndarray:
    """The output's sound: the kept stretches joined with short crossfades, exactly as long as the
    output's frames. A freeze inside the clip is silent, the sound fading out into it and back in
    after it; a held last frame (`timeline.hold`) is silent, and the sound fades out into it."""
    rate = video.AUDIO_RATE
    n_out = timeline.output_frames(clip.info.rate)
    want = int(round(n_out / clip.info.rate * rate))
    if timeline.is_identity(clip.info.frames / clip.info.rate):
        s = clip.sound
    else:
        s = audio.join(clip.sound, rate, timeline.sample_ranges(rate), fade=fade)
    if timeline.freezes:
        s = with_freezes(s, rate, timeline)
    if timeline.hold > 0:
        s = np.array(s[:int(round(timeline.played * rate))], np.float32)
        n = min(len(s), int(round(HOLD_FADE * rate)))
        if n:
            s[len(s) - n:] *= np.cos(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
    if len(s) < want:
        s = np.concatenate([s, np.zeros((want - len(s), s.shape[1]), np.float32)])
    return np.ascontiguousarray(s[:want])


def frames(info: video.VideoInfo):
    """Every frame of the clip, in order: decoded from its file, or asked of a drawn picture
    (`info.picture`: a film put together from several clips, say) one frame at a time."""
    if info.picture is not None:
        return (np.asarray(info.picture(i)) for i in range(info.frames))
    return iter(video.FrameReader(info))


def render(clip: Clip, timeline: Timeline, framing: Framing, draw: Optional[Draw], out: Path,
           fade: float = 0.02, crf: int = 18, audio_kbps: int = 192,
           mix: Optional[Callable[[np.ndarray, int], np.ndarray]] = None) -> dict:
    """Write the whole output video. Returns what was written and how long it took. `crf` is the
    picture's quality (lower is better and larger: 18 is near the source, 23 half the size). `mix`,
    when given, lays the effect's own sounds under the clip's (it gets the output's sound and its
    rate and returns the sound to write; the length is kept). A clip whose picture is drawn
    (`clip.info.picture`) is rendered from what it draws."""
    info = clip.info
    fps = info.rate
    n_out = timeline.output_frames(fps)
    started = time.time()
    sound = sound_for(clip, timeline, fade)
    if mix is not None:
        mixed = np.asarray(mix(sound, video.AUDIO_RATE), np.float32)
        if mixed.shape != sound.shape:
            raise ValueError("an effect's sound changed the length of the video's sound")
        sound = np.ascontiguousarray(mixed)
    writer = video.FrameWriter(out, framing.out_w, framing.out_h, info.fps, audio=sound, crf=crf,
                               audio_kbps=audio_kbps)
    reader = frames(info)
    current, k_read = None, -1
    try:
        for j in range(n_out):
            k = timeline.source_frame(j, fps, info.frames)
            while k_read < k:
                nxt = next(reader, None)
                if nxt is None:  # the clip ran a frame short of its stated length: hold the last
                    k_read = k
                    break
                current, k_read = nxt, k_read + 1
            if current is None:
                raise video.MediaError(f"could not read the picture of {info.path}")
            frame = framing.apply(current, k)
            if frame is current or not frame.flags.writeable:
                frame = frame.copy()
            if draw is not None:
                draw(frame, j, j / fps)
            writer.write(frame)
        path = writer.close()
    except BaseException:
        writer.abort()
        raise
    return {"path": str(path), "frames": n_out, "seconds_out": round(n_out / fps, 3),
            "render_s": round(time.time() - started, 2)}


def still_frames(clip: Clip, timeline: Timeline, framing: Framing, draw: Optional[Draw],
                 times: list[float]) -> list[tuple[float, np.ndarray]]:
    """The output frames at `times` (output seconds), drawn exactly as the render draws them."""
    info = clip.info
    fps = info.rate
    n_out = timeline.output_frames(fps)
    out = []
    for t in times:
        j = int(min(n_out - 1, max(0, round(t * fps))))
        k = timeline.source_frame(j, fps, info.frames)
        frame = framing.apply(video.read_frame(info, k), k).copy()
        if draw is not None:
            draw(frame, j, j / fps)
        out.append((j / fps, frame))
    return out


def save_stills(frames: list[tuple[float, np.ndarray]], out: Path) -> list[Path]:
    """Each still as its own JPEG beside `out`, plus one image with all of them side by side
    (`<out>.stills.jpg`), which is the one to look at."""
    from PIL import Image, ImageDraw

    paths = []
    stem = out.with_suffix("")
    for t, f in frames:
        p = Path(f"{stem}.still-{t:06.2f}s.jpg")
        video.save_jpeg(p, f, quality=90)
        paths.append(p)
    if frames:
        h = 960
        tiles = []
        for t, f in frames:
            scale = h / f.shape[0]
            img = Image.fromarray(f).resize((int(round(f.shape[1] * scale)), h), Image.LANCZOS)
            label = Image.new("RGB", (img.width, 44), (18, 18, 18))
            ImageDraw.Draw(label).text((12, 10), f"{t:.2f} s", fill=(235, 235, 235))
            tile = Image.new("RGB", (img.width, h + 44))
            tile.paste(img, (0, 0))
            tile.paste(label, (0, h))
            tiles.append(tile)
        sheet = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h + 44), (40, 40, 40))
        x = 0
        for tile in tiles:
            sheet.paste(tile, (x, 0))
            x += tile.width + 8
        p = Path(f"{stem}.stills.jpg")
        sheet.save(p, quality=88)
        paths.insert(0, p)
    return paths
