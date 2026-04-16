import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_metrics(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cfg: dict):
    out_root = Path(cfg["output_dir"])
    ds = cfg["dataset"]["name"]
    methods = ["standalone", "fedgh", "pfedmoe"]

    plt.figure(figsize=(8, 5))
    any_curve = False
    for m in methods:
        metrics_path = out_root / m / ds / "metrics.jsonl"
        rows = _read_metrics(metrics_path)
        if not rows:
            continue
        xs = [r["round"] for r in rows]
        ys = [r["mean_accuracy"] for r in rows]
        plt.plot(xs, ys, label=m)
        any_curve = True

    if not any_curve:
        print("No metrics found, skip plotting.")
        return

    plt.xlabel("Round")
    plt.ylabel("Mean Accuracy (%)")
    plt.title(f"Convergence on {ds}")
    plt.legend()
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"convergence_{ds}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved figure: {out}")
