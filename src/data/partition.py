import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _labels_from_dataset(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    raise ValueError("Dataset has no targets/labels field")


def _split_train_test(indices: List[int], train_ratio: float, rng: np.random.Generator):
    if len(indices) == 0:
        return [], []
    arr = np.array(indices, dtype=np.int64)
    rng.shuffle(arr)
    cut = int(len(arr) * train_ratio)
    return arr[:cut].tolist(), arr[cut:].tolist()


def pathological_partition(labels: np.ndarray, num_clients: int, classes_per_client: int, train_ratio: float, seed: int):
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    clients = {str(i): {"train": [], "test": []} for i in range(num_clients)}

    # Pathological split via label-sorted shards:
    # each client gets `classes_per_client` shards, and all samples are used once.
    num_shards = max(1, num_clients * classes_per_client)
    sorted_indices = np.argsort(labels, kind="stable")
    shard_size = max(1, len(sorted_indices) // num_shards)
    usable = shard_size * num_shards

    shards: List[List[int]] = []
    for i in range(num_shards):
        start = i * shard_size
        end = start + shard_size
        shards.append(sorted_indices[start:end].tolist())

    # Distribute remaining tail samples to avoid dropping any data.
    tail = sorted_indices[usable:]
    for i, idx in enumerate(tail.tolist()):
        shards[i % num_shards].append(idx)

    shard_ids = np.arange(num_shards)
    rng.shuffle(shard_ids)
    ptr = 0
    for cid in range(num_clients):
        picked = []
        for _ in range(classes_per_client):
            sid = int(shard_ids[ptr % num_shards])
            ptr += 1
            picked.extend(shards[sid])
        tr, te = _split_train_test(picked, train_ratio, rng)
        clients[str(cid)]["train"] = tr
        clients[str(cid)]["test"] = te
    return clients


def practical_dirichlet_partition(labels: np.ndarray, num_clients: int, gamma: float, train_ratio: float, seed: int):
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(labels.tolist()):
        class_to_indices[y].append(idx)
    for y in class_to_indices:
        rng.shuffle(class_to_indices[y])

    per_client = {str(i): [] for i in range(num_clients)}
    for c in range(num_classes):
        idxs = np.array(class_to_indices[c], dtype=np.int64)
        if len(idxs) == 0:
            continue
        proportions = rng.dirichlet(np.full(num_clients, gamma))
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)
        chunks = np.split(idxs, cuts[:-1])
        for i, chunk in enumerate(chunks):
            per_client[str(i)].extend(chunk.tolist())

    clients = {str(i): {"train": [], "test": []} for i in range(num_clients)}
    for cid, idxs in per_client.items():
        tr, te = _split_train_test(idxs, train_ratio, rng)
        clients[cid]["train"] = tr
        clients[cid]["test"] = te
    return clients


def partition_hash(partition: Dict) -> str:
    payload = json.dumps(partition, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_partition(path: str, payload: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_partition(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
