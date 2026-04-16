#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/cloud_full.yaml}"

echo "[cloud_full] using config: ${CONFIG}"
python -m src.main --config "${CONFIG}" --mode prepare_partition
python -m src.main --config "${CONFIG}" --mode train --method standalone
python -m src.main --config "${CONFIG}" --mode train --method fedgh
python -m src.main --config "${CONFIG}" --mode train --method pfedmoe
python -m src.main --config "${CONFIG}" --mode analyze

echo "[cloud_full] done."
