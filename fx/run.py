"""Run one effect on one video.

    python -m fx.run <effect> <video> [--opt k=v …] [--plan plan.json] [--still t,…|auto] [-o out.mp4] [--crf 18]
    python -m fx.run list                     # the effects and their options
    python -m fx.run words <video> [--opt fix=wrong=right]   # the words with their times

`--still` draws a few frames exactly as the render will and writes them beside the output
(`<out>.stills.jpg` has them side by side); nothing else is rendered. Every run writes
`<out>.plan.json`: what was decided, which `--plan` takes back after an edit.

The last lines say where the file is and how to keep changing it in AskEditor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from . import meta  # noqa: E402
from .core.paths import NeedSetup  # noqa: E402
from .core.video import MediaError  # noqa: E402

ASKEDITOR = "https://askeditor.com"


def askeditor_line(effect: str, what: str, lead: str = "To change it without rendering again") -> str:
    """The last line after a render: where to keep changing it in AskEditor. Once the skills pages
    answer (`meta.PAGES_LIVE`), it points at the effect's own page; until then, at AskEditor itself."""
    where = f"https://{meta.page(effect)}" if meta.PAGES_LIVE else ASKEDITOR
    return f"{lead}, open it in AskEditor: {where}, then ask \"{what}\"."


def _opts(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--opt needs name=value (got {p!r})")
        k, v = p.split("=", 1)
        out[k.strip().replace("-", "_")] = v.strip()
    return out


def _default_out(video: Path, effect: str, ending: str | None = None) -> Path:
    """Beside the video: `<video>-<effect>.mp4`, or `<video><ending>` for an effect that writes a file."""
    folder = video.parent if os.access(video.parent, os.W_OK) else Path.cwd()
    return folder / (f"{video.stem}{ending}" if ending else f"{video.stem}-{effect}.mp4")


def _still_times(spec: str, job) -> list[float]:
    if spec.strip().lower() in ("auto", ""):
        return job.still_times()
    try:
        return [float(x.strip().rstrip("s")) for x in spec.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--still takes seconds like 1.2,3.5,6 or auto (got {spec!r})") from None


def cmd_list() -> int:
    from . import effects

    for e in effects.manifest():
        module = effects.load(e["name"])
        print(f"{e['name']}: {e['line']}")
        print(f"  type: {e['type_this']}")
        for name, kind in module.OPTIONS.items():
            print(f"  {name}={e['options'][name]!s:<8} {kind.help}")
        print()
    return 0


def cmd_words(video: Path, given: dict) -> int:
    from .core import words
    from .core.clip import Clip

    clip = Clip(video, fix=given.get("fix"))
    ws = clip.words
    print(words.numbered(ws))
    print(f"\n{len(ws)} words, {clip.info.duration:.2f} s. Times are seconds into the clip; each start is when the voice begins.")
    if clip.unsure:
        print("\nHeard on a second listen but not sure, so never shown or cut on:")
        for u in words.unsure_stretches(clip.unsure):
            print(f"  {u['s']:6.2f} s  \"{u['w']}\": {u['why']}")
    return 0


def cmd_effect(effect: str, video: Path, given: dict, plan_path: str | None, still: str | None,
               out: Path | None, crf: int = 18, audio_kbps: int = 192) -> int:
    from . import effects
    from .core import render
    from .core.clip import Clip
    from .effects.options import parse_all

    try:
        module = effects.load(effect)
    except KeyError:
        raise SystemExit(f"there is no effect called {effect!r}; the effects are {', '.join(effects.names())}") from None
    entry = effects.entry(effect)
    # Every effect takes fix=, so a person's spelling can go with every command (SKILL.md says to):
    # an effect that does not list it still reads the words put right.
    given = dict(given)
    fix = None
    if "fix" not in module.OPTIONS:
        fix = given.pop("fix", None)
    opts = parse_all(module.OPTIONS, entry["options"], given)
    plan = None
    if plan_path:
        plan = json.loads(Path(plan_path).read_text())
        if plan.get("effect") != effect:
            raise effects.PlanError(f"that plan is for {plan.get('effect')!r}, not {effect!r}")
    if getattr(module, "REPORTS", False):
        # An effect that only reads the video and reports on it (qc-check): it renders nothing and
        # its exit code is its verdict.
        return module.report(video, opts, out, still=still)
    out = out or _default_out(video, effect, getattr(module, "WRITES", None))
    started = time.time()
    clip = Clip(video, fix=opts.get("fix") or fix)
    made = module.prepare(clip, opts, plan)
    # An effect may make several files from one run (one per shape, say): the first is written to
    # `out` and each other one beside it, named with its label (`talk-reframe-4x5.mp4`).
    jobs = made if isinstance(made, list) else [made]
    out.parent.mkdir(parents=True, exist_ok=True)
    first = jobs[0]
    if first.writes is not None:  # words, not pictures: the file is the result
        info = clip.info
        print(f"effect: {effect} · {info.width}x{info.height} at {info.rate:.3g} fps, {info.duration:.2f} s")
        for k, v in first.checks.items():
            print(f"check: {k}: {v}")
        if still is not None:
            print("(this effect draws nothing: it writes a file, so --still makes no stills)")
        extra = first.writes(out) or {}
        for k, v in extra.items():
            if k != "failed":
                print(f"{k}: {v}")
        what = entry["askeditor"]["what"]
        print(askeditor_line(effect, what, "To keep this with the video and change it there"))
        return 1 if extra.get("failed") else 0
    info = first.clip.info
    print(f"effect: {effect} · {info.width}x{info.height} at {info.rate:.3g} fps, {info.duration:.2f} s")
    for n, job in enumerate(jobs):
        target = out if n == 0 else out.with_name(f"{out.stem}-{job.label}{out.suffix}")
        tag = f"{job.label} " if len(jobs) > 1 and job.label else ""
        plan_file = target.with_suffix(".plan.json")
        plan_file.write_text(json.dumps(job.plan, indent=1, ensure_ascii=False) + "\n")
        for k, v in job.checks.items():
            print(f"check: {tag}{k}: {v}")
        print(f"plan: {plan_file}")
        if still is not None:
            times = _still_times(still, job)
            # The job's clip: the clip given, or a film the effect put together from it (several clips joined).
            frames = render.still_frames(job.clip, job.timeline, job.framing, job.draw, times)
            paths = render.save_stills(frames, target)
            print(f"stills: {paths[0]}")
            if job.preview is not None:
                for k, v in (job.preview(target) or {}).items():
                    print(f"{k}: {v}")
            continue
        result = render.render(job.clip, job.timeline, job.framing, job.draw, target, crf=crf, audio_kbps=audio_kbps,
                               mix=job.sound)
        extra = job.after(result, target) if job.after else {}
        for k, v in (extra or {}).items():
            print(f"{k}: {v}")
        took = time.time() - started
        print(f"video: {result['path']} ({result['seconds_out']:.2f} s, {job.framing.out_w}x{job.framing.out_h}) "
              f"in {took:.1f} s")
    if still is not None:
        print("(stills only: nothing was rendered. Look at them, then run again without --still.)")
        return 0
    what = entry["askeditor"]["what"]
    print(askeditor_line(effect, what))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "list":
        return cmd_list()
    ap = argparse.ArgumentParser(prog="python -m fx.run")
    ap.add_argument("effect")
    ap.add_argument("video")
    ap.add_argument("--opt", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--plan")
    ap.add_argument("--still", nargs="?", const="auto")
    ap.add_argument("-o", "--out")
    ap.add_argument("--crf", type=int, default=18, help="picture quality: 18 (default) is near the source, 23 is half the size")
    ap.add_argument("--audio-kbps", type=int, default=192)
    args = ap.parse_args(argv)
    video = Path(args.video).expanduser().resolve()
    try:
        given = _opts(args.opt)
        if args.effect == "words":
            return cmd_words(video, given)
        out = Path(args.out).expanduser().resolve() if args.out else None
        return cmd_effect(args.effect, video, given, args.plan, args.still, out, crf=args.crf, audio_kbps=args.audio_kbps)
    except NeedSetup as e:
        print(f"NEED_SETUP: {e}", file=sys.stderr)
        return 3
    except MediaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:  # an option or a plan that cannot be used
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
