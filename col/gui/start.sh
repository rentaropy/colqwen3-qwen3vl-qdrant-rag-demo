#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --no-cache-dir -q --root-user-action=ignore -r /app/col/gui/requirements.txt
python3 /app/col/gui/pre_ingest.py
exec streamlit run /app/col/gui/app.py \
  --server.address 0.0.0.0 \
  --server.port "${GUI_PORT:-8501}"
