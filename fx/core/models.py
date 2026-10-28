"""The model files, where each comes from, and fetching them (setup.sh runs `python -m fx.core.models fetch`).

Every file comes from its official home only, at a pinned version, and is checked against its
SHA-256 before it is used:

- the face, face landmark and person models from MediaPipe's model storage
  (storage.googleapis.com/mediapipe-models);
- the speech model from Hugging Face, `Systran/faster-whisper-small.en`, at a pinned revision.

A file already present with the right checksum is never fetched again, so running setup twice
touches the network once at most.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from .paths import MODELS, NeedSetup

_STORE = "https://storage.googleapis.com/mediapipe-models"

VISION = {
    "face": {
        "file": "blaze_face_short_range.tflite",
        "url": f"{_STORE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "sha256": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        "what": "the face model",
    },
    "person": {
        "file": "selfie_segmenter.tflite",
        "url": f"{_STORE}/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
        "sha256": "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b",
        "what": "the person model",
    },
    "person-parts": {
        "file": "selfie_multiclass_256x256.tflite",
        "url": f"{_STORE}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        "sha256": "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
        "what": "the detailed person model (hair, skin, clothes)",
    },
    "landmarks": {
        "file": "face_landmarker.task",
        "url": f"{_STORE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "sha256": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "what": "the face landmark model (where each mouth is, for who is talking)",
    },
}

# The speech model's files at one revision, each with its SHA-256 (model.bin's is the one Hugging
# Face records for it; the others were checked against Hugging Face's own records on 28 October 2026).
SPEECH = {
    "small.en": {
        "repo": "Systran/faster-whisper-small.en",
        "revision": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
        "files": {
            "config.json": "666a9605530ac1f61fa8177f3702b4dacec9966749e42610839fcc32661d5fae",
            "model.bin": "62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a",
            "tokenizer.json": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df",
            "vocabulary.txt": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf",
            "README.md": "c87ff776137c2ecb30ee3b324b4cf5f38abf9423464613641ee45430bf580851",
        },
    },
}
DEFAULT_SPEECH = "small.en"
_VERIFIED = ".verified.json"  # what was checked, so the next run need not hash 480 MB again


def vision_path(name: str) -> Path:
    entry = VISION[name]
    path = MODELS / entry["file"]
    if not path.exists():
        raise NeedSetup(f"{entry['what']} is missing. Run setup.sh first.")
    return path


def speech_path(name: str = DEFAULT_SPEECH) -> Path:
    if name not in SPEECH:
        raise ValueError(f"unknown speech model {name!r}; choose one of {', '.join(SPEECH)}")
    path = MODELS / f"faster-whisper-{name}"
    if not (path / "model.bin").exists():
        raise NeedSetup(f"the speech model {name} is missing. Run setup.sh first.")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path, sha256: str, what: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    print(f"  fetching {what}…", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = _sha256(part)
    if got != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{what} did not match its checksum (got {got[:12]}…); nothing was kept")
    part.replace(path)


def _verified(folder: Path) -> dict:
    try:
        return json.loads((folder / _VERIFIED).read_text())
    except (OSError, ValueError):
        return {}


def _present(path: Path, sha256: str, record: dict) -> bool:
    """The file is there and is the right one. A file checked before is trusted while its size and
    modification time are unchanged; anything else is hashed."""
    if not path.exists():
        return False
    st = path.stat()
    stamp = [sha256, st.st_size, int(st.st_mtime)]
    if record.get(path.name) == stamp:
        return True
    if _sha256(path) != sha256:
        return False
    record[path.name] = stamp
    return True


def fetch_vision(name: str) -> Path:
    entry = VISION[name]
    path = MODELS / entry["file"]
    record = _verified(MODELS)
    if not _present(path, entry["sha256"], record):
        _download(entry["url"], path, entry["sha256"], entry["what"])
        _present(path, entry["sha256"], record)
    (MODELS / _VERIFIED).write_text(json.dumps(record, indent=1) + "\n")
    return path


def fetch_speech(name: str = DEFAULT_SPEECH) -> Path:
    entry = SPEECH[name]
    target = MODELS / f"faster-whisper-{name}"
    target.mkdir(parents=True, exist_ok=True)
    record = _verified(target)
    for file, sha256 in entry["files"].items():
        if not _present(target / file, sha256, record):
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{file}"
            _download(url, target / file, sha256, f"the speech model ({name}): {file}")
            _present(target / file, sha256, record)
    record["revision"] = entry["revision"]
    (target / _VERIFIED).write_text(json.dumps(record, indent=1) + "\n")
    return target


def missing() -> list[str]:
    """What setup still has to fetch (by presence only; `fetch` checks the contents)."""
    out = [VISION[n]["what"] for n in VISION if not (MODELS / VISION[n]["file"]).exists()]
    folder = MODELS / f"faster-whisper-{DEFAULT_SPEECH}"
    if any(not (folder / f).exists() for f in SPEECH[DEFAULT_SPEECH]["files"]):
        out.append(f"the speech model ({DEFAULT_SPEECH})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fx.core.models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="fetch the models this pack needs, checking each file")
    sub.add_parser("check", help="say which models are present")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        for name in VISION:
            fetch_vision(name)
        fetch_speech(DEFAULT_SPEECH)
        print("  models ready (each file checked against its pinned checksum)")
        return 0
    gone = missing()
    print("models: " + ("all present" if not gone else "missing " + ", ".join(gone)))
    return 0 if not gone else 1


if __name__ == "__main__":
    sys.exit(main())
