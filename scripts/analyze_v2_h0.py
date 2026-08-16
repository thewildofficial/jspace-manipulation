from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 1729


def _bootstrap_topologies(frame: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    eligible = frame[frame["baseline_correct"]].copy()
    eligible["prompt_group"] = (
        eligible["category"]
        + "/"
        + eligible["function"]
        + "/"
        + eligible["source_argument"]
    )
    rows = []
    for topology, part in eligible.groupby("topology", sort=True):
        groups = list(part.groupby("prompt_group", sort=True))
        boot_success: list[float] = []
        boot_gain: list[float] = []
        for _ in range(n_boot):
            picks = rng.integers(0, len(groups), len(groups))
            sample = pd.concat([groups[index][1] for index in picks], ignore_index=True)
            boot_success.append(float(sample["swap_success"].mean()))
            boot_gain.append(float(sample["target_logodds_gain"].mean()))
        rows.append(
            {
                "topology": topology,
                "n_eligible": len(part),
                "swap_success_rate": float(part["swap_success"].mean()),
                "swap_success_low": float(np.quantile(boot_success, 0.025)),
                "swap_success_high": float(np.quantile(boot_success, 0.975)),
                "target_logodds_gain": float(part["target_logodds_gain"].mean()),
                "target_logodds_gain_low": float(np.quantile(boot_gain, 0.025)),
                "target_logodds_gain_high": float(np.quantile(boot_gain, 0.975)),
                "mean_delta_rms_ratio": float(part["mean_delta_rms_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    smoke = Path("results/v2_smoke_tests")
    workspace = Path("results/v2_workspace_mapping")
    flexible = pd.read_json(smoke / "raw/flexible_generalization.jsonl", lines=True)
    diagnostics = pd.read_json(workspace / "raw/workspace_diagnostics.jsonl", lines=True)
    diagnostics = diagnostics[diagnostics["record_type"] == "workspace_diagnostic"]
    topology = _bootstrap_topologies(flexible)
    smoke.joinpath("summaries").mkdir(parents=True, exist_ok=True)
    smoke.joinpath("figures").mkdir(parents=True, exist_ok=True)
    workspace.joinpath("summaries").mkdir(parents=True, exist_ok=True)
    workspace.joinpath("figures").mkdir(parents=True, exist_ok=True)
    topology.to_csv(smoke / "summaries/topology_summary.csv", index=False)

    mapping = (
        diagnostics.groupby(["layer", "layer_fraction", "selected"], as_index=False)
        .agg(
            mean_excess_kurtosis=("excess_kurtosis", "mean"),
            mean_jlens_logitlens_cosine=("jlens_logitlens_cosine", "mean"),
            selection_score=("selection_score", "mean"),
            n=("excess_kurtosis", "size"),
        )
        .sort_values("layer")
    )
    mapping.to_csv(workspace / "summaries/workspace_layer_summary.csv", index=False)

    order = [
        "single_layer_final_position",
        "single_layer_all_positions",
        "workspace_band_final_position",
        "workspace_band_all_positions",
    ]
    plot = topology.set_index("topology").loc[order].reset_index()
    x = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].errorbar(
        x,
        plot["swap_success_rate"],
        yerr=np.vstack(
            [
                plot["swap_success_rate"] - plot["swap_success_low"],
                plot["swap_success_high"] - plot["swap_success_rate"],
            ]
        ),
        fmt="o",
        capsize=4,
        color="#176B87",
    )
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Conditional target-answer top-1 rate")
    axes[1].errorbar(
        x,
        plot["target_logodds_gain"],
        yerr=np.vstack(
            [
                plot["target_logodds_gain"] - plot["target_logodds_gain_low"],
                plot["target_logodds_gain_high"] - plot["target_logodds_gain"],
            ]
        ),
        fmt="o",
        capsize=4,
        color="#D97745",
    )
    axes[1].axhline(0, color="0.3", linewidth=1)
    axes[1].set_ylabel("Target-vs-source answer log-odds gain")
    labels = ["1L / final", "1L / all", "band / final", "band / all"]
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(alpha=0.25)
    fig.suptitle("V2 H0: intervention topology on flexible generalization")
    fig.tight_layout()
    fig.savefig(smoke / "figures/topology_effect.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        mapping["layer_fraction"], mapping["selection_score"], label="Frozen score"
    )
    selected = mapping[mapping["selected"]]
    if not selected.empty:
        axis.axvspan(
            selected["layer_fraction"].min(),
            selected["layer_fraction"].max(),
            color="#D97745",
            alpha=0.2,
            label="Selected band",
        )
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_xlabel("Normalized layer depth")
    axis.set_ylabel("Kurtosis z − 0.5 × motor-agreement z")
    axis.set_title("Task-independent Qwen workspace-band selection")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(workspace / "figures/workspace_selection.png", dpi=160)
    plt.close(fig)

    summary = json.loads((smoke / "raw/h0_summary.json").read_text())
    print(json.dumps(summary["summary"]["gate"], indent=2))


if __name__ == "__main__":
    main()
