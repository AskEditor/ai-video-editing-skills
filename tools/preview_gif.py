"""The README's moving previews, made from the example films (which are not in this repository).

    .venv/bin/python tools/preview_gif.py [--lab DIR]     # writes assets/preview.gif and assets/clean-cut.gif

The grid plays the middle seconds of twelve effects side by side, looping; the second shows Clean
Cut's before and after. Both are GIFs because GitHub plays a GIF in a README and never a video.
"""
import argparse, os, subprocess, tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
# Each tile and the second its effect shows from in its full example film (catalogue/films/full/),
# read off the measured windows in catalogue/films/TRIMS.md and checked by eye.
GRID = [("words-behind", 7.3), ("backdrop", 5.0), ("spotlight", 7.6), ("magazine-cover", 0.6),
        ("chart", 7.0), ("glass-widgets", 8.0), ("tier-list", 13.0), ("checklist", 12.0),
        ("freeze-point", 7.4), ("broll-cutaways", 5.0), ("glitch", 1.4), ("receipt", 15.5)]
COLS, TILE_W, TILE_H, FPS, SECONDS = 4, 150, 266, 10, 3.0


def middle(film: Path) -> float:
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(film)], capture_output=True, text=True).stdout or 0)
    return max(0.0, d / 2 - SECONDS / 2)


def label_png(path: Path, text: str) -> Path:
    """A small dark pill with white type, drawn here because many ffmpeg builds have no drawtext."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(PACK / "fonts" / "Poppins" / "Poppins-Bold.ttf"), 20)
    w, h = int(font.getlength(text)) + 24, 34
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(0, 0, 0, 150))
    d.text((12, 3), text, font=font, fill=(255, 255, 255, 255))
    im.save(path)
    return path


def gif(filter_inputs: list[str], graph: str, out: Path, colors: int = 128) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pal = Path(tmp) / "pal.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *filter_inputs, "-filter_complex",
                        graph + f",palettegen=max_colors={colors}:stats_mode=diff", str(pal)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *filter_inputs, "-i", str(pal), "-filter_complex",
                        graph + f"[g];[g][{filter_inputs.count('-i')}:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=os.environ.get("FX_LAB") or str(PACK / "tests" / "lab"))
    lab = Path(ap.parse_args().lab)
    web, full = lab / "catalogue" / "films" / "web", lab / "catalogue" / "films" / "full"
    (PACK / "assets").mkdir(exist_ok=True)
    inputs, parts = [], []
    for k, (name, at) in enumerate(GRID):
        inputs += ["-ss", f"{at:.2f}", "-t", f"{SECONDS}", "-i", str(full / f"{name}.mp4")]
        parts.append(f"[{k}:v]fps={FPS},scale={TILE_W}:{TILE_H}:force_original_aspect_ratio=increase,"
                     f"crop={TILE_W}:{TILE_H},setsar=1[t{k}]")
    rows = len(GRID) // COLS
    layout = "|".join(f"{(k % COLS) * TILE_W}_{(k // COLS) * TILE_H}" for k in range(len(GRID)))
    graph = ";".join(parts) + ";" + "".join(f"[t{k}]" for k in range(len(GRID))) + \
        f"xstack=inputs={len(GRID)}:layout={layout}"
    gif(inputs, graph, PACK / "assets" / "preview.gif")
    before, after = web / "clean-cut-before.mp4", web / "clean-cut-after.mp4"
    tmp = Path(tempfile.mkdtemp())
    labels = [label_png(tmp / f"{t}.png", t) for t in ("BEFORE", "AFTER")]
    inputs = ["-ss", "0", "-t", "6", "-i", str(before), "-ss", "0", "-t", "6", "-i", str(after),
              "-i", str(labels[0]), "-i", str(labels[1])]
    graph = (f"[0:v]fps={FPS},scale=240:-2,setsar=1[a0];[1:v]fps={FPS},scale=240:-2,setsar=1[b0];"
             "[a0][2:v]overlay=(W-w)/2:16[a];[b0][3:v]overlay=(W-w)/2:16[b];[a][b]hstack=inputs=2")
    gif(inputs, graph, PACK / "assets" / "clean-cut.gif")
    for f in ("preview.gif", "clean-cut.gif"):
        p = PACK / "assets" / f
        print(f"{p.relative_to(PACK)}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
