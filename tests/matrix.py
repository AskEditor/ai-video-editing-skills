"""Render one effect, or all, on the test clips, and write what a person needs to judge them.

    .venv/bin/python tests/matrix.py <effect|all> [--only A,D-fitted] [--lab DIR] [--out DIR] [--warm] [--qc]
    .venv/bin/python tests/matrix.py <effect|all> --qc-only [--qc-out DIR]

For each clip it writes, in `<lab>/<effect>/` (or `<out>/<effect>/`):

- `<clip>.mp4`: the render, made by `python -m fx.run`, the command a person runs;
- `<clip>-sheet.jpg`: six frames of that MP4 across its seconds, each while words are on screen;
- `<clip>.plan.json` (and, for clean-cut, `<clip>.cuts.md`): what was decided;

and one `notes.md` with the timings, the checks and the plan for every clip. Only the part of
`notes.md` between the matrix's markers is rewritten; what a person wrote around it stays.

Each clip is copied to a fresh folder first, so the timings are a first run (words, face and all).
`--warm` runs on the clips where they are, reusing what earlier runs found.

`--qc` adds the final step: `qc-check` on every file the run made, against the clip it was made
from (`<clip>.qc.md` and `<clip>.qc.jpg`; the verdict goes into notes.md). `--qc-only` checks the
renders already in the lab without making them again; with `--qc-out DIR` the reports go to
`DIR/<effect>/` and a `DIR/summary.md` lists every verdict, and nothing in the effect's own folder
is touched.

Nothing is deleted: a file about to be replaced is moved into `history/<time>/` first.
"""

from __future__ import annotations

import os
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
# The test clips are not in this repository. Point FX_LAB at a folder holding them (test-clips/,
# raw-takes/), or put them, or a link to them, at tests/lab (ignored by git).
LAB = Path(os.environ.get("FX_LAB") or Path(__file__).resolve().parent / "lab")
if __name__ == "__main__" and not LAB.exists():
    print("skipped: these checks need the test clips, which are not in this repository. "
          "Set FX_LAB to their folder, or put them at tests/lab.")
    raise SystemExit(0)
CLIPS = {
    "A": "test-clips/A-demo-day-talk-clean-8s-720x1280.mp4",
    "B": "test-clips/B-airport-headroom-1080x1920.mp4",
    "C": "test-clips/C-lake-path-hard-1080x1920.mp4",
    "D": "test-clips/D-hotel-plain-wall-1080x1920.mp4",
    "E": "test-clips/E-closeup-long-pauses-1080x1920.mp4",
    "F": "test-clips/F-train-moving-1080x1920.mp4",
    "33": "raw-takes/33-demo-day-pt2-talking-only.mp4",
    # Raw take 31 where it lies (never copied into the lab): our longest whole talk, 107 s at
    # the Hua Hin hotel. D is cut from it. Run without --warm, so its caches go to a temporary copy.
    "31": "../ig-series/2026-10-05-scenic-talk-huahin/source/IMG_6064.MOV",
    # A spliced from itself: the first sentence said twice, then a phrase broken off and begun again.
    "A-retake": "clean-cut/inputs/A-with-retake-and-false-start.mp4",
    # Videos the pack itself made, for effects that work on an edit's own events (its cuts, its cards):
    # clean-cut's E (two cuts) and tier-list's B (six cards). Their plans sit beside them.
    "E-cut": "clean-cut/E.mp4",
    "tier-B": "tier-list/B.mp4",
    # Takes where we really says a list, cut from raw 31, 15 and 17 for the lists effects:
    # the Hua Hin journey ("First of all … I then boarded … and then eventually … I eventually went"),
    # three actions in one breath, and three caption looks with a last "or".
    "journey": "checklist/inputs/journey-steps-720x1280.mp4",
    "close-actions": "checklist/inputs/closeup-three-actions-1080x1920.mp4",
    "close-looks": "ranked-list/inputs/closeup-three-looks-1080x1920.mp4",
    # A computer voice reading a priced list and its total over a grey frame (receipt/inputs/README.md).
    "prices": "receipt/inputs/spoken-prices-720x1280.mp4",
    # our own landscape and square takes, for reframe (`reframe/inputs/make_inputs.sh`
    # says how each was made): a green-screen take, the square white room (HLG HDR, as the phone
    # filmed it), and two of his green-screen takes side by side with one voice at a time.
    "wide": "reframe/inputs/wide-greenscreen-16s-1904x1080.mp4",
    "square": "reframe/inputs/square-white-room-hlg-1080x1080.mp4",
    "two": "reframe/inputs/two-of-the-founder-side-by-side-1904x1080.mp4",
    # A computer voice reading lines with amounts over a grey frame (made for chart with the Mac's own
    # `say`): the only clips here that say numbers, for number-counter's auto runs.
    "numbers": "chart/inputs/spoken-numbers.mp4",
    "numbers-2": "chart/inputs/spoken-numbers-2.mp4",
}
LOOKS = ["one-word", "stacked", "serif-chip", "fitted"]
# Each run: (name, clip, options). The name is what the files are called.
RUNS = {
    "captions": [(c, c, {}) for c in ("A", "B", "C", "33")]
    + [(f"{c}-{look}", c, {"look": look}) for c in ("A", "D") for look in LOOKS],
    "clean-cut": [(c, c, {}) for c in ("A", "B", "C", "33", "E", "A-retake")],
    # headline: the line it picks by itself on each clip, then each style on B (the clip with room
    # above the head), the type move, the tight room on A, and words of your own over a spoken line.
    "headline": [(c, c, {}) for c in ("A", "B", "C", "D")]
    + [(f"B-{style}", "B", {"style": style}) for style in ("serif", "highlight", "sticker")]
    + [("B-type", "B", {"animation": "type"})]  # rise and wipe are the serif's and the highlight's own
    + [("A-highlight", "A", {"style": "highlight"}),
       ("B-text", "B", {"text": "Edit *less*. | Post more.", "when": "improve your production quality"})],
    # cta-card: the default (a follow button, no handle made up) on every clip, then the comment card
    # and the comment bar on A (a close shot, the card below the chin) and B (room above the head),
    # and a handle beside the button.
    "cta-card": [(c, c, {}) for c in ("A", "B", "C", "D")]
    + [(f"{c}-{kind}", c, {"kind": kind, "word": "GUIDE"}) for c in ("A", "B") for kind in ("comment", "comment-box")]
    + [("B-handle", "B", {"kind": "follow", "word": "@yourname"})],
}
# An effect not listed above names its own runs as `MATRIX` in its module, in the same shape;
# without one it runs on A, B and C. An option's value may say `{lab}` for the lab's own folder.
DEFAULT_RUNS = [(c, c, {}) for c in ("A", "B", "C")]


def runs_for(effect: str) -> list:
    if effect in RUNS:
        return RUNS[effect]
    sys.path.insert(0, str(PACK))
    from fx import effects

    return list(getattr(effects.load(effect), "MATRIX", DEFAULT_RUNS))


# The one word the transcriber mishears on these clips is the speaker's name. A person would tell
# Claude, and Claude passes it as `fix`; the matrix does the same, and notes.md says so.
FIXES = {"C": "Urban=Eben"}
# The labels an effect that makes one file per shape gives its other files (`<name>-4x5.mp4`).
EXTRA_LABELS = ("9x16", "4x5", "1x1", "16x9")
# Test renders are for judging, not for keeping at full quality: about half the size of the default.
CRF, AUDIO_KBPS = 23, 128
MIN_FREE_GB = 5  # below this the matrix refuses to render: a full disk takes down everything else
START, END = "<!-- matrix:start -->", "<!-- matrix:end -->"


def _keep_history(folder: Path, names: list[str], stamp: str) -> None:
    for name in names:
        p = folder / name
        if p.exists():
            dest = folder / "history" / stamp
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest / name))


def _sheet(video: Path, times: list[float], out: Path) -> None:
    """Six frames of the rendered video itself, side by side, each labelled with its time."""
    from PIL import Image, ImageDraw, ImageFont

    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, t in enumerate(times):
            png = Path(tmp) / f"{k}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", str(png)],
                           check=True)
            tiles.append((t, Image.open(png).convert("RGB")))
    h = 640
    font = ImageFont.truetype(str(PACK / "fonts/Poppins/Poppins-SemiBold.ttf"), 22)
    cells = []
    for t, img in tiles:
        img = img.resize((int(round(img.width * h / img.height)), h), Image.LANCZOS)
        cell = Image.new("RGB", (img.width, h + 40), (20, 20, 20))
        cell.paste(img, (0, 0))
        ImageDraw.Draw(cell).text((10, h + 6), f"{t:.2f} s", fill=(230, 230, 230), font=font)
        cells.append(cell)
    sheet = Image.new("RGB", (sum(c.width for c in cells) + 6 * (len(cells) - 1), h + 40), (45, 45, 45))
    x = 0
    for c in cells:
        sheet.paste(c, (x, 0))
        x += c.width + 6
    sheet.save(out, quality=88)


def _sheet_times(plan: dict, seconds: float, fps: float) -> list[float]:
    """Six moments across the output, each in the middle of a phrase with all its words showing, or
    the six the effect's own plan names (`sheet_times`: an effect on screen for a few seconds shows
    them across those seconds)."""
    if plan.get("sheet_times"):
        return [min(seconds - 0.5 / fps, max(0.0, float(t))) for t in plan["sheet_times"]][:6]
    cards = plan.get("cards") or []
    if not cards:
        return [seconds * (k + 0.5) / 6 for k in range(6)]
    keyed = [c for c in cards if c.get("key") is not None]
    if len(keyed) >= 6:  # the frames that show the most: a phrase with its key word
        cards = keyed
    picks = [round(k * (len(cards) - 1) / 5) for k in range(6)] if len(cards) >= 6 else list(range(len(cards)))
    out = []
    for i in picks:
        c = cards[i]
        last_word = max(float(w["s"]) for w in c["words"])
        t = max(last_word + 0.12, (c["start"] + c["end"]) / 2)
        out.append(min(c["end"] - 1.0 / fps, t))
    while len(out) < 6:  # a clip with fewer than six phrases: fill in between
        out.append(seconds * (len(out) + 0.5) / 6)
    return sorted(out)[:6]


def _module(effect: str):
    sys.path.insert(0, str(PACK))
    from fx import effects

    return effects.load(effect)


def _options(effect: str) -> dict:
    return json.loads((PACK / "effects" / effect / "effect.json").read_text())["options"]


def run_clip(effect: str, name: str, clip_key: str, given: dict, lab: Path, warm: bool, stamp: str,
             out_root: Path | None = None, qc: bool = False) -> dict:
    folder = (out_root or lab) / effect
    folder.mkdir(parents=True, exist_ok=True)
    src = lab / CLIPS[clip_key]
    keep = [f"{name}.mp4", f"{name}-sheet.jpg", f"{name}.plan.json", f"{name}.cuts.md", f"{name}.words.json",
            f"{name}.face.json", f"{name}.person.mkv", f"{name}.path.jpg", f"{name}.qc.md", f"{name}.qc.jpg"]
    # An effect that makes one file per shape (reframe) writes the others beside the first, by label.
    # An effect that names other labels for its extra files (long-to-shorts: `-2`, `-3`) and files it
    # writes beside the first (`BESIDE`) gets those kept too.
    labels = EXTRA_LABELS + tuple(getattr(_module(effect), "LABELS", ()))
    keep += [f"{name}{ext}" for ext in getattr(_module(effect), "BESIDE", ())]
    for label in labels:  # not `lab`, which is the lab's folder and fills `{lab}` in the options below
        keep += [f"{name}-{label}{ext}" for ext in (".mp4", ".plan.json", ".path.jpg", ".qc.md", ".qc.jpg", "-sheet.jpg")]
    _keep_history(folder, keep, stamp)
    out = folder / f"{name}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        clip = src
        if not warm:  # a fresh copy: nothing found before is reused
            clip = Path(tmp) / src.name
            shutil.copy2(src, clip)
        cmd = [sys.executable, "-m", "fx.run", effect, str(clip), "-o", str(out), "--crf", str(CRF),
               "--audio-kbps", str(AUDIO_KBPS)]
        for k, v in given.items():
            cmd += ["--opt", f"{k}={str(v).replace('{lab}', str(lab))}"]
        if clip_key in FIXES and "fix" in _options(effect):
            cmd += ["--opt", f"fix={FIXES[clip_key]}"]
        started = time.time()
        run = subprocess.run(cmd, cwd=PACK, capture_output=True, text=True)
        took = time.time() - started
        words_json = clip.with_name(clip.stem + ".words.json")
        transcribe_s = None
        if words_json.exists():
            transcribe_s = json.loads(words_json.read_text()).get("seconds_to_transcribe")
    lines = [ln for ln in run.stdout.splitlines() if ln.strip()]
    if run.returncode != 0:
        err = [ln for ln in run.stderr.splitlines() if ln.strip() and "objc[" not in ln]
        return {"clip": name, "ok": False, "error": err[-1] if err else "failed", "seconds": round(took, 1)}
    plan = json.loads(out.with_suffix(".plan.json").read_text())
    fps = float(plan["output"]["fps"])
    videos = [ln.split(": ", 1)[1].split(" (")[0] for ln in lines if ln.startswith("video: ")]
    # An effect that knows which six moments show it best names them in its plan.
    times = plan.get("sheet_times") or _sheet_times(plan, float(plan["output"]["seconds"]), fps)
    # An effect whose work is heard draws its own sheet (the frames over its sound) and says where.
    own = next((ln.split(": ", 1)[1] for ln in lines if ln.startswith("sound sheet: ")), None)
    if own and Path(own).exists():
        shutil.move(own, folder / f"{name}-sheet.jpg")
    else:
        _sheet(out, times, folder / f"{name}-sheet.jpg")
    # The other files of a run that makes one per shape get a sheet each, at their own moments.
    for extra in [Path(v) for v in videos if Path(v) != out and Path(v).exists()]:
        try:
            ep = json.loads(extra.with_suffix(".plan.json").read_text())
            et = ep.get("sheet_times") or _sheet_times(ep, float(ep["output"]["seconds"]), fps)
        except (OSError, ValueError, KeyError):
            et = times
        _sheet(extra, et, extra.with_name(extra.stem + "-sheet.jpg"))
    checks = {ln.split(": ", 2)[1]: ln.split(": ", 2)[2] for ln in lines if ln.startswith("check: ")}
    # What the effect measured on the finished file: the lines between its plan and its video.
    marks = [i for i, ln in enumerate(lines) if ln.startswith(("plan: ", "video: "))]
    after = lines[marks[0] + 1:marks[-1]] if len(marks) >= 2 else []
    after = [ln for ln in after if not ln.startswith(("check: ", "plan: ", "video: ", "stills: "))]
    result = {"clip": name, "ok": True, "source": str(src), "seconds": round(took, 1), "transcribe_s": transcribe_s,
              "checks": checks, "plan": plan, "fix": FIXES.get(clip_key) if "fix" in _options(effect) else None,
              "after": [ln for ln in after if ": " in ln and not ln.startswith(("sound sheet: ", "path sheet: "))],
              "videos": videos,
              "stdout": [ln for ln in lines if not ln.startswith(("check: ", "plan: "))]}
    for ln in lines:
        if ln.startswith("over-cut check: "):
            result["overcut"] = ln[len("over-cut check: "):]
    if qc:  # the last step: every file this run made, checked against the clip it was made from
        made = [Path(v) for v in videos] or [out]
        result["qc"] = [run_qc(v, src, v.with_name(v.stem + ".qc.md")) for v in made if v.exists()]
    return result


def run_qc(video: Path, source: Path, report: Path) -> dict:
    """qc-check on one file against its source, as a person runs it; what it found, in short."""
    cmd = [sys.executable, "-m", "fx.run", "qc-check", str(video), "--opt", f"source={source}", "-o", str(report)]
    started = time.time()
    run = subprocess.run(cmd, cwd=PACK, capture_output=True, text=True, env=_env())
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("check: ")]
    verdict = next((ln.split(": ", 2)[2] for ln in lines if ln.startswith("check: verdict: ")), None)
    if verdict is None:
        err = [ln for ln in run.stderr.splitlines() if ln.strip() and "objc[" not in ln]
        return {"file": video.name, "verdict": "ERROR", "error": err[-1] if err else "failed",
                "seconds": round(time.time() - started, 1)}
    found = []
    for ln in lines[1:]:
        name, rest = ln[len("check: "):].split(": ", 1)
        level = rest.split(" ", 1)[0]
        if level in ("FAIL", "WARN"):
            found.append({"check": name, "level": level, "what": rest.split(" ", 1)[1] if " " in rest else ""})
    return {"file": video.name, "verdict": verdict.split(" ")[0], "summary": verdict, "found": found,
            "exit": run.returncode, "report": str(report), "seconds": round(time.time() - started, 1)}


def _rows_on_disk(out: Path) -> list:
    """(effect, file, source, result) for every report in out/<effect>/, read back from the reports."""
    import re

    rows = []
    for md in sorted(out.glob("*/*.qc.md")):
        if md.parent.name == "history":
            continue
        text = md.read_text()
        verdict = "PASS" if "**PASS**" in text else "FAIL" if "**FAIL**" in text else "ERROR"
        found = [{"check": c, "level": lv, "what": w} for c, lv, w in
                 re.findall(r"^\| ([a-z ]+) \| (FAIL|WARN) \| (.*?) \|$", text, flags=re.M)]
        against = re.search(r"against its source `([^`]+)`", text)
        rows.append((md.parent.name, md.name[:-len(".qc.md")], against.group(1) if against else "-",
                     {"verdict": verdict, "found": found}))
    return rows


def _env() -> dict:
    """The environment for a check: with FX_THREADS set (a busy machine), numpy's own threads are
    capped the same way as ffmpeg's and OpenCV's."""
    import os

    env = dict(os.environ)
    if env.get("FX_THREADS", "").isdigit():
        for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(k, env["FX_THREADS"])
    return env


def _qc_line(q: dict) -> str:
    if q.get("verdict") == "ERROR":
        return f"could not run: {q.get('error')}"
    parts = [f"{x['level']} {x['check']}" for x in q.get("found", [])]
    return f"{q['verdict']}" + (f" ({'; '.join(parts)})" if parts else "")


def _block(effect: str, results: list[dict], warm: bool) -> str:
    rows = [START, "", f"_Written by `tests/matrix.py {effect}` on {datetime.now():%d %B %Y at %H:%M}"
            f" ({'reusing earlier finds' if warm else 'each clip a fresh copy: first-run timings'}). "
            f"Renders at CRF {CRF}, AAC {AUDIO_KBPS}k._", ""]
    checked = any(r.get("qc") for r in results)
    rows += ["| Clip | In | Out | Took | Of which words | Over the face | Outside the safe zone | Placed |"
             + (" Checked before done |" if checked else ""),
             "|---|---|---|---|---|---|---|---|" + ("---|" if checked else "")]
    for r in results:
        if not r["ok"]:
            rows.append(f"| {r['clip']} | | | {r['seconds']} s | | | | failed: {r['error']} |")
            continue
        p, c = r["plan"], r["checks"]
        seconds_in = p.get("summary", {}).get("seconds_in", p["output"]["seconds"])
        qc = r.get("qc") or []
        rows.append(f"| {r['clip']} | {float(seconds_in):.2f} s | {p['output']['seconds']:.2f} s | {r['seconds']} s | "
                    f"{r['transcribe_s'] if r['transcribe_s'] is not None else '?'} s | {c.get('frames_over_face', '-')} frames | "
                    f"{c.get('frames_outside_safe_zone', '-')} frames | {c.get('placed', '-')} |"
                    + (f" {', '.join(q['verdict'] for q in qc) or '-'} |" if checked else ""))
    rows.append("")
    for r in results:
        if not r["ok"]:
            continue
        p = r["plan"]
        rows += [f"### {r['clip']}: `{Path(r['source']).name}`", ""]
        opts = {k: v for k, v in p["options"].items() if v not in ("", None)}
        rows.append(f"- Options: `{json.dumps(opts)}`" + (f" (the name fixed: `{r['fix']}`)" if r.get("fix") else ""))
        rows.append(f"- Output: {p['output']['width']}x{p['output']['height']} at {p['output']['fps']} fps, "
                    f"{p['output']['seconds']:.2f} s; {p['output']['framing']}.")
        if "summary" in p:
            s = p["summary"]
            reasons = ", ".join(f"{k} {v:.2f} s" for k, v in s["by_reason"].items()) or "nothing"
            rows.append(f"- Cuts: {s['cuts']}, {s['seconds_removed']:.2f} s removed ({reasons}). "
                        f"Report: `{r['clip']}.cuts.md`.")
            for rem in p["removals"]:
                said = f" “{rem['words']}”" if rem["words"] else ""
                why = rem["reason"] + (f", {rem['detail']}" if rem["detail"] else "")
                rows.append(f"  - {rem['start']:.2f}–{rem['end']:.2f} s ({rem['seconds']:.2f} s): {why}{said}")
        if r.get("overcut"):
            rows.append(f"- Over-cut check: {r['overcut']}.")
        if "card" in p:
            c = p["card"]
            rows.append(f"- Card: {c['kind']}" + (f" “{c['word']}”" if c.get("word") else "") +
                        f", {c['where']}, {c['start']:.2f}–{c['end']:.2f} s at {c['scale']:.0%}; "
                        f"{r['checks'].get('sounds', '0')} sounds; pointer over the face: "
                        f"{r['checks'].get('pointer_frames_over_face', '-')} frames.")
        if "headline" in p:
            h = p["headline"]
            rows.append(f"- Headline: “{h['text']}” ({h['style']}, {h['animation']}), {h['where']}, "
                        f"{h['start']:.2f}–{h['end']:.2f} s; {r['checks'].get('lands', '')}; "
                        f"{r['checks'].get('size', '')}.")
        cards = p.get("cards") or []
        if cards:
            keyed = sum(1 for c in cards if c.get("key") is not None)
            rows.append(f"- Captions: {len(cards)} phrases, {keyed} with a key word, placed {r['checks'].get('placed')}; "
                        f"shrunk to fit: {r['checks'].get('shrunk', 0)}.")
            rows.append("- Phrases: " + " | ".join(c["text"] for c in cards))
        if effect not in RUNS:  # a newer effect's own checks, each on its line
            for k, v in r["checks"].items():
                if k not in ("frames_over_face", "frames_outside_safe_zone", "placed", "warnings"):
                    rows.append(f"- {k.replace('_', ' ').capitalize()}: {v}")
            for ln in r.get("after", []):  # and what it measured on the finished file
                k, v = ln.split(": ", 1)
                rows.append(f"- Measured, {k}: {v}")
        if p.get("unsure"):
            rows.append("- Heard on a second listen but unsure, so never shown or cut on: " + "; ".join(
                f"“{u['w']}” at {u['s']:.2f} s ({u['why']})" for u in p["unsure"]) + ".")
        for q in r.get("qc") or []:
            rows.append(f"- Checked before done (`{q['file']}`): {_qc_line(q)}.")
        warn = r["checks"].get("warnings", "[]")
        if warn not in ("[]", ""):
            rows.append(f"- Warnings: {warn}")
        rows.append("")
    rows.append(END)
    return "\n".join(rows)


def write_notes(effect: str, folder: Path, block: str, stamp: str) -> Path:
    notes = folder / "notes.md"
    if notes.exists():
        dest = folder / "history" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notes, dest / "notes.md")
        text = notes.read_text()
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda m: block, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    else:
        text = f"# {effect} on the test clips\n\n{block}\n\n## Looked at\n\n(by hand, after reading every sheet)\n"
    notes.write_text(text)
    return notes


def runs_for(effect: str) -> list[tuple[str, str, dict]]:
    """The effect's runs: from `RUNS` above, else the `MATRIX` its own module lists (an effect built
    on its own branch names its clips there), else A, B and C."""
    if effect in RUNS:
        return RUNS[effect]
    from fx import effects as fx_effects

    module = fx_effects.load(effect)
    return [(n, c, dict(o)) for n, c, o in getattr(module, "MATRIX", [(c, c, {}) for c in ("A", "B", "C")])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("effect")
    ap.add_argument("--only", help="run only these names, e.g. A,D-fitted (default: every run of the effect)")
    ap.add_argument("--lab", default=str(LAB))
    ap.add_argument("--out", help="where to write the renders, sheets and notes (default: the lab itself)")
    ap.add_argument("--warm", action="store_true", help="reuse what earlier runs found (faster, not first-run timings)")
    ap.add_argument("--qc", action="store_true", help="check every render with qc-check against its clip, last")
    ap.add_argument("--qc-only", action="store_true", help="check the renders already in the lab; render nothing")
    ap.add_argument("--qc-out", help="with --qc-only: write the reports to DIR/<effect>/ and DIR/summary.md")
    ap.add_argument("--jobs", type=int, default=1, help="with --qc-only: how many files to check at once")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(PACK))
    if args.effect == "all":
        from fx import effects as fx_effects

        effects = fx_effects.names()
    else:
        effects = [args.effect]
    lab = Path(args.lab)
    out_root = Path(args.out).expanduser() if args.out else lab
    free_gb = shutil.disk_usage(lab).free / 1e9
    if free_gb < MIN_FREE_GB:
        print(f"refusing to render: only {free_gb:.1f} GB free on the disk that holds {lab} "
              f"(the matrix needs {MIN_FREE_GB} GB). Free some space first; nothing was written.")
        return 2
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if args.qc_only:
        return qc_only(effects, lab, args.only, Path(args.qc_out) if args.qc_out else None, stamp, args.jobs)
    failed = False
    for effect in effects:
        only = set(args.only.split(",")) if args.only else None
        results = []
        for name, clip_key, given in runs_for(effect):
            if only and name not in only:
                continue
            r = run_clip(effect, name, clip_key, given, lab, args.warm, stamp, out_root, qc=args.qc)
            status = "ok" if r["ok"] else f"FAILED: {r['error']}"
            print(f"{effect} {name}: {status} in {r['seconds']} s")
            failed |= not r["ok"]
            results.append(r)
        # The latest result of every run is kept in matrix.json, so running a few with --only
        # updates those and keeps the rest in notes.md.
        store = out_root / effect / "matrix.json"
        try:
            kept = json.loads(store.read_text())
        except (OSError, ValueError):
            kept = {}
        for r in results:
            kept[r["clip"]] = r
        store.write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n")
        order = [name for name, _, _ in runs_for(effect)]
        merged = [kept[n] for n in order if n in kept]
        notes = write_notes(effect, out_root / effect, _block(effect, merged, args.warm), stamp)
        print(f"notes: {notes}")
    return 1 if failed else 0


def qc_only(effects: list[str], lab: Path, only: str | None, out: Path | None, stamp: str, jobs: int = 1) -> int:
    """qc-check on the renders already in the lab, each against the clip its run names (`jobs` at a
    time: each check is its own process)."""
    from concurrent.futures import ThreadPoolExecutor

    picked = set(only.split(",")) if only else None
    rows = []
    pool = ThreadPoolExecutor(max_workers=max(1, jobs))
    pending = []  # (effect, run name, file, source, future), in the order the runs are listed
    for effect in effects:
        folder = lab / effect
        for name, clip_key, _ in runs_for(effect):
            if picked and name not in picked:
                continue
            first = folder / f"{name}.mp4"
            if not first.exists():
                continue
            made = [first] + [folder / f"{name}-{x}.mp4" for x in EXTRA_LABELS if (folder / f"{name}-{x}.mp4").exists()]
            src = lab / CLIPS[clip_key]
            for v in made:
                report = (out / effect / f"{v.stem}.qc.md") if out else v.with_name(v.stem + ".qc.md")
                report.parent.mkdir(parents=True, exist_ok=True)
                # Nothing is replaced without being kept: an earlier report moves into history/<time>/.
                sheet = report.with_name(report.name[:-len(".qc.md")] + ".qc.jpg")
                _keep_history(report.parent, [report.name, sheet.name], stamp)
                pending.append((effect, name, v, src, pool.submit(run_qc, v, src, report)))
    by_run: dict = {}
    for effect, name, v, src, fut in pending:
        q = fut.result()
        rows.append((effect, v.stem, src.name, q))
        by_run.setdefault((effect, name), []).append(q)
        print(f"{effect} {v.stem}: {_qc_line(q)} in {q['seconds']} s", flush=True)
    pool.shutdown()
    if not out:  # the verdicts go into each effect's own notes
        for effect in effects:
            store = lab / effect / "matrix.json"
            try:
                kept = json.loads(store.read_text())
            except (OSError, ValueError):
                continue
            for (e, name), results in by_run.items():
                if e == effect and name in kept:
                    kept[name]["qc"] = results
            store.write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n")
            order = [n for n, _, _ in runs_for(effect)]
            write_notes(effect, lab / effect, _block(effect, [kept[n] for n in order if n in kept], True), stamp)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        rows = _rows_on_disk(out)  # every report there, so a later partial run keeps the summary whole
        verdicts = [q.get("verdict") for *_, q in rows]
        lines = [f"# qc-check on the renders already in the lab", "",
                 f"_Written by `tests/matrix.py <effect> --qc-only --qc-out` on {datetime.now():%d %B %Y at %H:%M}: "
                 f"{len(rows)} files, {verdicts.count('PASS')} PASS, {verdicts.count('FAIL')} FAIL"
                 + (f", {verdicts.count('ERROR')} could not run" if 'ERROR' in verdicts else "") + "._", "",
                 "| Effect | File | Against | Verdict | Fails and warnings |", "|---|---|---|---|---|"]
        for effect, stem, src_name, q in rows:
            found = "; ".join(f"{x['level']} {x['check']}: {x['what'][:110]}" for x in q.get("found", []))
            lines.append(f"| {effect} | [{stem}]({effect}/{stem}.qc.md) | {src_name} | {q.get('verdict')} | "
                         f"{found.replace('|', '/') or '-'} |")
        summary = out / "summary.md"
        if summary.exists():
            dest = out / "history" / stamp
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary, dest / "summary.md")
        summary.write_text("\n".join(lines) + "\n")
        print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
