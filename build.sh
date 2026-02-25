#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[build] error: python3 not found" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "[build] error: pip is not available for $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "[build] PyInstaller not found. Installing with pip..."
  "$PYTHON_BIN" -m pip install --user pyinstaller
fi

echo "[build] cleaning previous artifacts"
rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"
rm -f "$ROOT_DIR/opwt.spec"

echo "[build] building binary"
"$PYTHON_BIN" -m PyInstaller \
  --onefile \
  --name opwt \
  --clean \
  --noconfirm \
  "$ROOT_DIR/opwt.py"

echo "[build] done: $ROOT_DIR/dist/opwt"
