#!/usr/bin/env bash
# Sets the pack up on this computer: a Python environment in .venv (inside this folder), the
# packages in requirements.txt, the models (each checked against its checksum), and a check for
# ffmpeg. Safe to run any number of times: it does only what is missing, so after the first run it
# takes a second or two and touches nothing.
#
# The last line it prints is one of:
#   READY         everything is in place (the line before it, PYTHON …, is the Python to use)
#   NEED_PYTHON   no Python 3.10, 3.11 or 3.12 was found
#   NEED_FFMPEG   ffmpeg (and ffprobe) are not installed
#   NEED_NETWORK  something had to be downloaded and could not be (the message above says what)

set -u
cd "$(dirname "$0")" || exit 1
PACK="$(pwd)"

say() { printf '%s\n' "$*"; }

# ---- 1. A Python the video libraries are built for --------------------------------------------
# The face, person and speech libraries publish their packages for Python 3.10 to 3.12. A newer
# Python (3.13, 3.14) has none, so it is skipped even when it is the computer's default.
supported() {
  "$@" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1
}

venv_python() {
  if [ -x .venv/bin/python ]; then echo ".venv/bin/python"
  elif [ -x .venv/Scripts/python.exe ]; then echo ".venv/Scripts/python.exe"
  fi
}

PY="$(venv_python)"
if [ -n "$PY" ] && supported "$PY"; then
  : # the environment is already here and is the right Python
else
  BASE=""
  for v in 3.12 3.11 3.10; do
    for c in "python$v" "/opt/homebrew/bin/python$v" "/usr/local/bin/python$v" "/usr/bin/python$v"; do
      if command -v "$c" >/dev/null 2>&1 && supported "$c"; then BASE="$c"; break 2; fi
    done
    if command -v py >/dev/null 2>&1 && supported py "-$v"; then BASE="py -$v"; break; fi  # Windows
  done
  if [ -z "$BASE" ]; then
    for c in python3 python; do
      if command -v "$c" >/dev/null 2>&1 && supported "$c"; then BASE="$c"; break; fi
    done
  fi
  if [ -z "$BASE" ]; then
    say "This pack needs Python 3.10, 3.11 or 3.12 (its video libraries are not built for newer ones yet)."
    say "On a Mac: brew install python@3.12   Elsewhere: https://www.python.org/downloads/"
    if ! command -v ffmpeg >/dev/null 2>&1; then say "ffmpeg is missing too; it will be asked for next."; fi
    say "NEED_PYTHON"
    exit 2
  fi
  say "Making the Python environment with $BASE …"
  # shellcheck disable=SC2086
  if ! $BASE -m venv --clear .venv; then
    say "Could not make a Python environment with $BASE."
    say "NEED_PYTHON"
    exit 2
  fi
  PY="$(venv_python)"
fi

# ---- 2. The packages, reinstalled only when requirements.txt changes ---------------------------
STAMP=".venv/.requirements.sha256"
WANT="$("$PY" -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())')"
HAVE="$(cat "$STAMP" 2>/dev/null || true)"
if [ "$WANT" != "$HAVE" ] || ! "$PY" -c 'import mediapipe, faster_whisper, cv2, PIL, numpy' >/dev/null 2>&1; then
  say "Installing the packages (the first time this is a few hundred MB) …"
  if ! "$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt; then
    say "The packages could not be installed. Check the internet connection and run setup again."
    say "NEED_NETWORK"
    exit 3
  fi
  printf '%s\n' "$WANT" > "$STAMP"
fi

# ---- 3. The models, each checked against its pinned checksum -----------------------------------
if ! "$PY" -m fx.core.models fetch; then
  say "A model could not be fetched. Check the internet connection and run setup again."
  say "NEED_NETWORK"
  exit 3
fi

# ---- 4. ffmpeg, which reads and writes every video ----------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  say "ffmpeg is not installed. Mac: brew install ffmpeg   Windows: winget install ffmpeg   Ubuntu: sudo apt install ffmpeg"
  say "NEED_FFMPEG"
  exit 2
fi

say "PYTHON $PACK/$PY"
say "READY"
