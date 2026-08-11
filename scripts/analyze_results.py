from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jspace_policy.analysis import (
    add_behavior_metrics,
    behavior_summary,
    paired_readout_trajectories,
)
from jspace_policy.plotting import (
    plot_behavior_gate,
    plot_intervention_sign_flip,
    plot_paired_policy_shift,
    plot_readout_trajectories,
)


def _read_table(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("artifacts/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("artifacts/figures"))
    parser.add_argument(
        "--behavior-file",
        default="behavior.jsonl",
        help="Behavior JSONL filename inside --raw-dir (default: behavior.jsonl)",
    )
    args = parser.parse_args()
    behavior_path = args.raw_dir / args.behavior_file
    if behavior_path.exists():
        behavior = add_behavior_metrics(_read_table(behavior_path))
        stem = behavior_path.stem
        figure_suffix = "" if stem == "behavior" else f"_{stem.removeprefix('behavior_')}"
        args.processed_dir.mkdir(parents=True, exist_ok=True)
        behavior.to_parquet(args.processed_dir / f"{stem}.parquet", index=False)
        behavior_summary(behavior, n_boot=5000).to_csv(
            args.processed_dir / f"{stem}_summary.csv", index=False
        )
        plot_behavior_gate(behavior, args.figure_dir / f"behavior{figure_suffix}_gate.png")
        plot_paired_policy_shift(
            behavior, args.figure_dir / f"paired_policy_shift{figure_suffix}.png"
        )
        print(f"generated behavioral summaries and figures for {behavior_path}")
    else:
        print(f"skipped behavior figures: {behavior_path} does not exist")

    readout_path = args.raw_dir / "lens_readout_27b.jsonl"
    if readout_path.exists():
        readout = _read_table(readout_path)
        policy, fact = paired_readout_trajectories(readout)
        args.processed_dir.mkdir(parents=True, exist_ok=True)
        policy.to_csv(args.processed_dir / "lens_policy_trajectory_27b.csv", index=False)
        fact.to_csv(args.processed_dir / "lens_fact_trajectory_27b.csv", index=False)
        plot_readout_trajectories(
            readout, args.figure_dir / "lens_readout_trajectories_27b.png"
        )
        print("generated observational J-lens trajectories")
    else:
        print(f"skipped readout figures: {readout_path} does not exist")

    intervention_path = args.raw_dir / "interventions.parquet"
    if intervention_path.exists():
        interventions = _read_table(intervention_path)
        plot_intervention_sign_flip(
            interventions, args.figure_dir / "intervention_sign_flip.png"
        )
        print("generated intervention figures")
    else:
        print(f"skipped intervention figures: {intervention_path} does not exist")


if __name__ == "__main__":
    main()
