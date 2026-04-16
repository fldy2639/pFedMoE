import json
from pathlib import Path

from src.data.datasets import build_dataset
from src.data.partition import (
    load_partition,
    partition_hash,
    pathological_partition,
    practical_dirichlet_partition,
    save_partition,
)
from src.utils.config import ensure_dir


def _partition_file(cfg):
    ds = cfg["dataset"]["name"]
    ptype = cfg["dataset"]["partition_type"]
    n = cfg["federated"]["num_clients"]
    seed = cfg["seed"]
    return str(Path(cfg["dataset"]["partition_dir"]) / f"{ds}_{ptype}_n{n}_s{seed}.json")


def run(cfg: dict):
    ensure_dir(cfg["dataset"]["partition_dir"])
    train_set, _, _, _ = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["data_dir"],
        cfg["dataset"].get("offline_fallback", False),
    )
    labels = train_set.targets
    fpath = _partition_file(cfg)
    if Path(fpath).exists():
        payload = load_partition(fpath)
        print(f"Use cached partition: {fpath}")
    else:
        if cfg["dataset"]["partition_type"] == "pathological":
            payload = pathological_partition(
                labels=labels,
                num_clients=cfg["federated"]["num_clients"],
                classes_per_client=cfg["dataset"]["pathological"]["classes_per_client"],
                train_ratio=cfg["dataset"]["train_ratio"],
                seed=cfg["seed"],
            )
        else:
            payload = practical_dirichlet_partition(
                labels=labels,
                num_clients=cfg["federated"]["num_clients"],
                gamma=cfg["dataset"]["practical"]["dirichlet_gamma"],
                train_ratio=cfg["dataset"]["train_ratio"],
                seed=cfg["seed"],
            )
        save_partition(fpath, payload)
        print(f"Created partition: {fpath}")

    p_hash = partition_hash(payload)
    meta = {"partition_file": fpath, "partition_hash": p_hash}
    out_meta = Path(cfg["output_dir"]) / "prepare_partition_meta.json"
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Partition hash: {p_hash}")
