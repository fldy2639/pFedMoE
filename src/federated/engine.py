import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models.simple_models import LinearGate, LocalHead, SimpleExtractor, clone_module


def _client_loader(dataset, indices, batch_size):
    if len(indices) == 0:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)


def _eval_client(extractor, head, dataset, indices, device):
    if len(indices) == 0:
        return 0.0
    loader = DataLoader(Subset(dataset, indices), batch_size=256, shuffle=False)
    extractor.eval()
    head.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            feat = extractor(x)
            logits = head(feat)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)


def _weighted_average(state_dicts: List[Dict], weights: List[int]):
    keys = state_dicts[0].keys()
    total = float(sum(weights))
    out = {}
    for k in keys:
        out[k] = sum(sd[k] * (w / total) for sd, w in zip(state_dicts, weights))
    return out


def _build_local_components(in_channels, feature_dim, num_classes, gate_hidden_dim, sample_shape):
    local_extractor = SimpleExtractor(in_channels=in_channels, feature_dim=feature_dim)
    local_head = LocalHead(feature_dim=feature_dim, num_classes=num_classes)
    gate = LinearGate(input_shape=sample_shape, hidden_dim=gate_hidden_dim)
    return local_extractor, local_head, gate


def _snapshot_client_cpu(extractor, head, gate, method: str) -> Dict:
    out = {
        "extractor": copy.deepcopy(extractor).cpu(),
        "head": copy.deepcopy(head).cpu(),
    }
    if method == "pfedmoe":
        out["gate"] = copy.deepcopy(gate).cpu()
    return out


def _train_one_client(
    method,
    train_set,
    client_train_ids,
    global_proxy,
    in_channels,
    feature_dim,
    num_classes,
    gate_hidden_dim,
    cfg,
    device,
    resume: Optional[Dict] = None,
):
    local_extractor, local_head, gate = _build_local_components(
        in_channels, feature_dim, num_classes, gate_hidden_dim, train_set[0][0].shape
    )
    global_proxy_local = clone_module(global_proxy)
    global_proxy_local.load_state_dict(global_proxy.state_dict())

    if resume is not None:
        local_extractor.load_state_dict(resume["extractor"].state_dict())
        local_head.load_state_dict(resume["head"].state_dict())
        if method == "pfedmoe" and resume.get("gate") is not None:
            gate.load_state_dict(resume["gate"].state_dict())

    local_extractor.to(device)
    local_head.to(device)
    gate.to(device)
    global_proxy_local.to(device)

    params = list(local_extractor.parameters()) + list(local_head.parameters())
    if method == "pfedmoe":
        params += list(gate.parameters()) + list(global_proxy_local.parameters())
    elif method == "fedgh":
        params += list(global_proxy_local.parameters())
    # standalone uses only local extractor/head

    optim = torch.optim.SGD(
        params,
        lr=cfg["optimizer"]["lr_local"],
        momentum=cfg["optimizer"]["momentum"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    loader = _client_loader(train_set, client_train_ids, cfg["federated"]["batch_size"])
    if loader is None:
        return None, 0

    local_extractor.train()
    local_head.train()
    gate.train()
    global_proxy_local.train()

    for _ in range(cfg["federated"]["local_epochs"]):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            local_repr = local_extractor(x)
            if method == "pfedmoe":
                global_repr = global_proxy_local(x)
                alpha = gate(x)
                fused = alpha[:, 0:1] * global_repr + alpha[:, 1:2] * local_repr
                logits = local_head(fused)
            elif method == "fedgh":
                global_repr = global_proxy_local(x)
                logits = local_head(0.5 * (local_repr + global_repr))
            else:
                logits = local_head(local_repr)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

    upload_proxy = None
    if method in ("pfedmoe", "fedgh"):
        upload_proxy = {k: v.detach().cpu().clone() for k, v in global_proxy_local.state_dict().items()}
    sample_count = len(client_train_ids)
    stored = _snapshot_client_cpu(local_extractor, local_head, gate, method)
    return {"modules_cpu": stored, "proxy": upload_proxy}, sample_count


def run_federated(method: str, train_set, partition: Dict, num_classes: int, in_channels: int, cfg: Dict, out_dir: str):
    device = torch.device(cfg["device"])
    num_clients = cfg["federated"]["num_clients"]
    rounds = cfg["federated"]["rounds"]
    rate = cfg["federated"]["participation_rate"]
    feature_dim = cfg["model"]["feature_dim"]
    gate_hidden_dim = cfg["model"]["gate_hidden_dim"]
    rng = np.random.default_rng(cfg["seed"])

    global_proxy = SimpleExtractor(in_channels=in_channels, feature_dim=feature_dim).cpu()
    client_state: Dict[str, Dict] = {}
    history = []
    sampled_clients_by_round = []

    for r in range(1, rounds + 1):
        k = max(1, int(num_clients * rate))
        sampled = sorted(rng.choice(np.arange(num_clients), size=k, replace=False).tolist())
        sampled_clients_by_round.append(sampled)

        proxy_states, weights = [], []
        for cid in sampled:
            resume = client_state.get(str(cid))
            payload, n = _train_one_client(
                method,
                train_set,
                partition[str(cid)]["train"],
                global_proxy,
                in_channels,
                feature_dim,
                num_classes,
                gate_hidden_dim,
                cfg,
                device,
                resume=resume,
            )
            if payload is None:
                continue
            client_state[str(cid)] = payload["modules_cpu"]
            if payload["proxy"] is not None:
                proxy_states.append(payload["proxy"])
                weights.append(n)

        if len(proxy_states) > 0:
            global_proxy.load_state_dict(_weighted_average(proxy_states, weights))

        # evaluate mean individual accuracy on each client test set
        accs = []
        for cid in range(num_clients):
            key = str(cid)
            if key not in client_state:
                continue
            st = client_state[key]
            acc = _eval_client(
                st["extractor"].to(device),
                st["head"].to(device),
                train_set,
                partition[key]["test"],
                device,
            )
            accs.append(acc)
        mean_acc = float(np.mean(accs)) if accs else 0.0
        std_acc = float(np.std(accs)) if accs else 0.0
        history.append({"round": r, "mean_accuracy": mean_acc, "std_accuracy": std_acc, "sampled_clients": sampled})
        print(f"[{method}] round={r} mean_acc={mean_acc:.2f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "metrics.jsonl", "w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return history, sampled_clients_by_round
