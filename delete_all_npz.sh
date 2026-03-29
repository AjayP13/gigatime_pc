#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mapfile -d '' npz_files < <(find . -type f -name "*.npz" -print0)
npz_count="${#npz_files[@]}"

if [[ "$npz_count" -eq 0 ]]; then
  echo "No .npz files found under $SCRIPT_DIR"
  exit 0
fi

echo "Found $npz_count .npz file(s) under $SCRIPT_DIR"
echo "This operation is destructive and cannot be undone."

read -r -p "First confirmation: delete ALL .npz files recursively? [y/N] " confirm1
if [[ ! "$confirm1" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

read -r -p "Second confirmation: are you absolutely sure? [y/N] " confirm2
if [[ ! "$confirm2" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

for file in "${npz_files[@]}"; do
  rm "$file"
done

echo "Deleted $npz_count .npz file(s)."
