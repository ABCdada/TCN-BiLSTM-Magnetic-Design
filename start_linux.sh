#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_weight_manifest.py --manifest weights/manifest.json
python scripts/verify_cold_start_manifest.py --manifest weights/cold_start_manifest.json

LISTEN_ADDRESS=${MAGNET_LISTEN:-127.0.0.1:8000}
echo "Starting at http://${LISTEN_ADDRESS}/"
exec python -m waitress --listen="$LISTEN_ADDRESS" app:app
