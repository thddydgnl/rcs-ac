#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rcs-ac-v2-hygiene.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "== RCS-AC-v2 git hygiene =="
echo

echo "-- repo-wide nested .git directories (informational) --"
find . -path './.git' -prune -o -name .git -type d -prune -print
echo

echo "-- repo-wide nested .git files (informational) --"
find . -path './.git' -prune -o -name .git -type f -print
echo

echo "-- v2 nested .git metadata that must not be committed --"
V2_NESTED_GIT="$TMP_DIR/v2_nested_git.txt"
find RCS-AC-v2 \( -name .git -type d -o -name .git -type f \) -print > "$V2_NESTED_GIT"
cat "$V2_NESTED_GIT"
echo

echo "-- v2 heavy files that must not be committed --"
V2_HEAVY_FILES="$TMP_DIR/v2_heavy_files.txt"
find RCS-AC-v2 -type f \( \
  -name '*.pt' -o \
  -name '*.pth' -o \
  -name '*.ckpt' -o \
  -name '*.npz' -o \
  -name '*.npy' -o \
  -name '*.jsonl' \
\) -print > "$V2_HEAVY_FILES"
cat "$V2_HEAVY_FILES"
echo

echo "-- v2 files larger than 50 MB that must not be committed --"
V2_LARGE_FILES="$TMP_DIR/v2_large_files.txt"
find RCS-AC-v2 -type f -size +50M -print > "$V2_LARGE_FILES"
cat "$V2_LARGE_FILES"
echo

echo "-- git status for v2 and ignore policy --"
git status --short -- RCS-AC-v2 .gitignore

if [[ -s "$V2_NESTED_GIT" ]]; then
  echo "ERROR: nested .git metadata found under RCS-AC-v2" >&2
  exit 1
fi

if [[ -s "$V2_HEAVY_FILES" ]]; then
  echo "ERROR: heavy generated files found under RCS-AC-v2" >&2
  exit 1
fi

if [[ -s "$V2_LARGE_FILES" ]]; then
  echo "ERROR: files larger than 50 MB found under RCS-AC-v2" >&2
  exit 1
fi
