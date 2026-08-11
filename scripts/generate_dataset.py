from __future__ import annotations

import argparse
from pathlib import Path

from jspace_policy.dataset import generate_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-base", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--stage", choices=("toy", "composition"), default="toy")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/processed/prompts.jsonl")
    )
    args = parser.parse_args()
    rows = generate_dataset(n_base=args.n_base, seed=args.seed, stage=args.stage)
    write_jsonl(rows, args.output)
    print(f"wrote {len(rows)} rows ({args.n_base} grouped scenarios) to {args.output}")


if __name__ == "__main__":
    main()
