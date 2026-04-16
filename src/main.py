import argparse

from src.runner import analyze, prepare_partition, train
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="pFedMoE Phase-0 runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["prepare_partition", "train", "analyze"])
    parser.add_argument("--method", type=str, default=None, choices=[None, "standalone", "fedgh", "pfedmoe"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.mode == "prepare_partition":
        prepare_partition.run(cfg)
    elif args.mode == "train":
        if args.method is None:
            raise ValueError("--method is required when mode=train")
        train.run(cfg, args.method)
    elif args.mode == "analyze":
        analyze.run(cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
