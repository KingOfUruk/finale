#!/usr/bin/env bash
# Helper script to bootstrap the dev venv, load .env variables, and launch the Flask app locally.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  pip install --upgrade pip >/dev/null
  pip install -r "$PROJECT_ROOT/requirements.txt"
fi

if [ ! -f "$PROJECT_ROOT/.env" ]; then
  echo "Missing .env file. Copy .env.sample and populate Oracle/Redis credentials." >&2
  exit 1
fi

set -a
source "$PROJECT_ROOT/.env"
set +a

exec python "$PROJECT_ROOT/main.py"
