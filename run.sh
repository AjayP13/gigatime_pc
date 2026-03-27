#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

if [[ -f ".env" ]]; then
  # Export variables sourced from .env.
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

python -m pip install --upgrade pip pip-tools
pip install -r requirements.txt

python main.py "$@"
