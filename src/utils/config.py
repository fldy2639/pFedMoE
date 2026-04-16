import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
