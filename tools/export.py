"""Write the public copy of the pack: `main` as files, without the effects that are held.

    .venv/bin/python tools/export.py DEST

An effect whose effect.json says `"pages": false` is held: it is left out of the public copy (its
folder and its code), and the README, SKILL.md's table and effects.json are written again without it.
DEST must be a git repository; its files are replaced, its history is left to the caller.
"""
import json, shutil, subprocess, sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def main() -> int:
    dest = Path(sys.argv[1]).expanduser().resolve()
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} is not a git repository")
    for child in dest.iterdir():
        if child.name != ".git":
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    archive = subprocess.run(["git", "-C", str(PACK), "archive", "main"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    held = []
    for path in sorted((dest / "effects").glob("*/effect.json")):
        if json.loads(path.read_text()).get("pages") is False:
            name = path.parent.name
            held.append(name)
            shutil.rmtree(path.parent)
            module = dest / "fx" / "effects" / (name.replace("-", "_") + ".py")
            if module.exists():
                module.unlink()
    subprocess.run([sys.executable, "tools/build_manifest.py"], cwd=dest, check=True)
    print(f"exported to {dest}; held back: {', '.join(held) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
