import json
from pathlib import Path

from src.data.datasets import build_dataset
from src.data.partition import load_partition, partition_hash
from src.federated.engine import run_federated
from src.utils.config import config_hash, ensure_dir


def _partition_file(cfg):
    ds = cfg["dataset"]["name"]
    ptype = cfg["dataset"]["partition_type"]
    n = cfg["federated"]["num_clients"]
    seed = cfg["seed"]
    return str(Path(cfg["dataset"]["partition_dir"]) / f"{ds}_{ptype}_n{n}_s{seed}.json")


def run(cfg: dict, method: str):
    ensure_dir(cfg["output_dir"])
    train_set, _, in_channels, num_classes = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["data_dir"],
        cfg["dataset"].get("offline_fallback", False),
    )

    partition_path = _partition_file(cfg)
    if not Path(partition_path).exists():
        raise FileNotFoundError(f"Partition file not found: {partition_path}, run prepare_partition first.")
    partition = load_partition(partition_path)

    out_dir = Path(cfg["output_dir"]) / method / cfg["dataset"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    history, sampled_clients_by_round = run_federated(
        method=method,
        train_set=train_set,
        partition=partition,
        num_classes=num_classes,
        in_channels=in_channels,
        cfg=cfg,
        out_dir=str(out_dir),
    )

    # rounds to target mean accuracy
    target = cfg["eval"]["target_accuracy"] * 100.0
    r_to_target = None
    for row in history:
        if row["mean_accuracy"] >= target:
            r_to_target = row["round"]
            break

    run_meta = {
        "method": method,
        "dataset": cfg["dataset"]["name"],
        "partition_file": partition_path,
        "partition_hash": partition_hash(partition),
        "config_hash": config_hash(cfg),
        "global_seed": cfg["seed"],
        "sampler_seed": cfg["seed"],
        "sampled_clients_by_round": sampled_clients_by_round,
        "target_accuracy": cfg["eval"]["target_accuracy"],
        "rounds_to_target_accuracy": r_to_target,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
