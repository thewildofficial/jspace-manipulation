from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 1729
ROOT = Path("results/v2_h0r_diagnostic")
RAW = ROOT / "raw"
SUMMARY = ROOT / "summaries"
FIGURES = ROOT / "figures"
SEMANTIC_FILES = [
    "layer_sweep",
    "position_sweep",
    "strength_sweep",
    "cumulative_layers",
    "coordinate_trajectories",
]


def _load(name: str) -> pd.DataFrame:
    frame = pd.read_json(RAW / f"{name}.jsonl", lines=True)
    frame["artifact"] = name
    return frame


def _bootstrap_gain(part: pd.DataFrame, draws: int = 2000) -> tuple[float, float]:
    groups = [group for _, group in part.groupby("scenario_id", sort=True)]
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    values = np.empty(draws)
    for index in range(draws):
        sample = rng.integers(0, len(groups), len(groups))
        values[index] = np.mean(
            [float(groups[group]["target_logodds_gain"].mean()) for group in sample]
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["baseline_correct"].astype(bool)].copy()
    eligible["layers_key"] = eligible["layers"].map(lambda value: tuple(value))
    keys = [
        "artifact",
        "configuration_id",
        "layers_key",
        "position_mask",
        "operation",
        "control",
        "alpha",
    ]
    rows = []
    for values, part in eligible.groupby(keys, sort=True):
        function_effects = part.groupby("function")["target_logodds_gain"].mean()
        low, high = _bootstrap_gain(part)
        rows.append(
            dict(zip(keys, values, strict=True))
            | {
                "n_eligible": len(part),
                "n_scenarios": part["scenario_id"].nunique(),
                "mean_target_logodds_gain": float(part["target_logodds_gain"].mean()),
                "gain_ci_low": low,
                "gain_ci_high": high,
                "positive_gain_fraction": float(
                    (part["target_logodds_gain"] > 0).mean()
                ),
                "target_top1_fraction": float(part["target_top1"].mean()),
                "mean_output_kl_nats": float(part["output_kl_nats"].mean()),
                "mean_delta_rms_ratio": float(
                    part["mean_delta_rms_ratio"].mean()
                ),
                "median_condition_number": float(
                    part["median_condition_number"].median()
                ),
                "high_cosine_fraction": float(
                    (part["maximum_absolute_cosine"] > 0.95).mean()
                ),
                "maximum_absolute_cosine": float(
                    part["maximum_absolute_cosine"].max()
                ),
                "minimum_function_effect": float(function_effects.min()),
                "positive_function_count": int((function_effects > 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["layers"] = result["layers_key"].map(lambda value: json.dumps(value))
    return result.drop(columns="layers_key")


def _select_candidate(summary: pd.DataFrame) -> pd.Series:
    pool = summary[
        (summary["control"] == "semantic")
        & (summary["median_condition_number"] <= 20)
        & (summary["high_cosine_fraction"] <= 0.05)
        & (summary["mean_output_kl_nats"] <= 1.0)
        & (summary["mean_delta_rms_ratio"] <= 0.25)
        & (summary["positive_function_count"] >= 3)
        & (summary["positive_gain_fraction"] >= 0.70)
    ].copy()
    metrics = {
        "mean_target_logodds_gain": False,
        "positive_gain_fraction": False,
        "target_top1_fraction": False,
        "mean_output_kl_nats": True,
        "mean_delta_rms_ratio": True,
        "minimum_function_effect": False,
    }
    rank_columns = []
    for metric, ascending in metrics.items():
        column = f"rank_{metric}"
        pool[column] = pool[metric].rank(
            method="average", ascending=ascending, pct=True
        )
        rank_columns.append(column)
    pool["mean_rank"] = pool[rank_columns].mean(axis=1)
    best_score = float(pool["mean_rank"].min())
    tied = pool[pool["mean_rank"] <= best_score + 0.02].copy()
    width_order = {
        "final_prompt_position": 0,
        "argument_token_position": 1,
        "argument_through_end": 2,
        "all_prompt_positions": 3,
    }
    tied["n_layers"] = tied["layers"].map(lambda value: len(json.loads(value)))
    tied["mask_width_order"] = tied["position_mask"].map(width_order)
    winner = tied.sort_values(
        ["n_layers", "mask_width_order", "alpha", "mean_rank"],
        ascending=True,
    ).iloc[0]
    pool.sort_values("mean_rank").to_csv(SUMMARY / "candidate_ranking.csv", index=False)
    return winner


def _plot_line(frame: pd.DataFrame, x: str, y: str, title: str, path: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(frame[x], frame[y], marker="o", color="#176B87")
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / path, dpi=160)
    plt.close(figure)


def _make_figures(all_rows: pd.DataFrame, summary: pd.DataFrame, winner: pd.Series) -> None:
    layer = summary[
        (summary["artifact"] == "layer_sweep") & (summary["control"] == "semantic")
    ].copy()
    layer["layer"] = layer["layers"].map(lambda value: json.loads(value)[0])
    layer = layer.sort_values("layer")
    _plot_line(
        layer,
        "layer",
        "mean_target_logodds_gain",
        "H0R-B target log-odds gain by layer",
        "01_gain_vs_layer.png",
    )
    _plot_line(
        layer,
        "layer",
        "target_top1_fraction",
        "H0R-B target top-1 conversion by layer",
        "02_top1_vs_layer.png",
    )

    position = summary[summary["artifact"] == "position_sweep"].copy()
    position["layer"] = position["layers"].map(lambda value: json.loads(value)[0])
    pivot = position.pivot(
        index="position_mask", columns="layer", values="mean_target_logodds_gain"
    )
    axis = pivot.plot(kind="bar", figsize=(10, 5), rot=25)
    axis.set_ylabel("Mean target log-odds gain")
    axis.set_title("Position-mask comparison at the three exploratory layers")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURES / "03_position_masks.png", dpi=160)
    plt.close(axis.figure)

    strength = summary[summary["artifact"] == "strength_sweep"].copy()
    figure, axis = plt.subplots(figsize=(9, 5))
    for config, part in strength.groupby(
        strength["configuration_id"].str.replace(r"__alpha_.*$", "", regex=True)
    ):
        part = part.sort_values("alpha")
        axis.plot(part["alpha"], part["mean_target_logodds_gain"], marker="o", label=config)
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_xlabel("Alpha")
    axis.set_ylabel("Mean target log-odds gain")
    axis.set_title("H0R-B strength dose response")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "04_alpha_dose_response.png", dpi=160)
    plt.close(figure)

    cumulative = summary[summary["artifact"] == "cumulative_layers"].copy()
    cumulative["label"] = cumulative["layers"].map(
        lambda value: ",".join(map(str, json.loads(value)))
    )
    cumulative = cumulative.sort_values("mean_target_logodds_gain")
    axis = cumulative.plot.barh(
        x="label", y="mean_target_logodds_gain", legend=False, figsize=(9, 7)
    )
    axis.set_xlabel("Mean target log-odds gain")
    axis.set_ylabel("Intervened layers")
    axis.set_title("Cumulative, prefix/suffix, and odd/even layer sets")
    axis.grid(axis="x", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(FIGURES / "05_cumulative_layers.png", dpi=160)
    plt.close(axis.figure)

    winner_rows = all_rows[
        (all_rows["artifact"] == winner["artifact"])
        & (all_rows["configuration_id"] == winner["configuration_id"])
        & all_rows["baseline_correct"].astype(bool)
    ]
    trajectory = []
    for _, row in winner_rows.iterrows():
        for layer_name, record in row["layer_diagnostics"].items():
            coordinates = record["coordinates"]
            trajectory.append(
                {
                    "layer": int(layer_name),
                    "pre": np.mean(coordinates["source_pre"])
                    - np.mean(coordinates["target_pre"]),
                    "post": np.mean(coordinates["source_post"])
                    - np.mean(coordinates["target_post"]),
                }
            )
    trajectory_frame = pd.DataFrame(trajectory).groupby("layer", as_index=False).mean()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(trajectory_frame["layer"], trajectory_frame["pre"], marker="o", label="pre-hook")
    axis.plot(
        trajectory_frame["layer"],
        trajectory_frame["post"],
        marker="o",
        label="post-hook",
    )
    axis.axhline(0, color="0.3", linewidth=1)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Mean source − target coordinate")
    axis.set_title("Coordinate trajectory through the selected repeated write")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "06_coordinate_trajectory.png", dpi=160)
    plt.close(figure)

    reconstruction = pd.read_csv(SUMMARY / "reconstruction.csv")
    _plot_line(
        reconstruction,
        "measurement_layer",
        "median_clipped_reconstruction_fraction",
        "One-shot return toward the clean source state",
        "07_reconstruction.png",
    )

    semantic = summary[summary["control"] == "semantic"]
    figure, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(
        semantic["mean_delta_rms_ratio"],
        semantic["mean_target_logodds_gain"],
        c=semantic["mean_output_kl_nats"],
        cmap="viridis",
        alpha=0.75,
    )
    axis.set_xlabel("Mean ΔRMS/RMS")
    axis.set_ylabel("Mean target log-odds gain")
    axis.set_title("Effect versus intervention distance (color = output KL)")
    figure.colorbar(scatter, ax=axis, label="Output KL (nats)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "08_effect_vs_distance.png", dpi=160)
    plt.close(figure)

    controls = summary[
        (summary["artifact"].isin(["coordinate_trajectories", "controls"]))
    ].copy()
    controls["label"] = controls["configuration_id"].str.replace(
        r"^(operation|control)_", "", regex=True
    )
    controls = controls.sort_values("mean_target_logodds_gain")
    colors = [
        "#D97745" if control == "semantic" else "#7A869A"
        for control in controls["control"]
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(controls["label"], controls["mean_target_logodds_gain"], color=colors)
    axis.axvline(0, color="0.3", linewidth=1)
    axis.set_xlabel("Mean target log-odds gain")
    axis.set_title("Semantic interventions versus matched controls")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "09_semantic_vs_controls.png", dpi=160)
    plt.close(figure)


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = [_load(name) for name in SEMANTIC_FILES]
    controls = _load("controls")
    all_rows = pd.concat([*frames, controls], ignore_index=True)
    topology = _aggregate(all_rows)
    topology.to_csv(SUMMARY / "topology_effects.csv", index=False)

    layer = topology[topology["artifact"] == "layer_sweep"].copy()
    layer["layer"] = layer["layers"].map(lambda value: json.loads(value)[0])
    layer.sort_values("layer").to_csv(SUMMARY / "layer_effects.csv", index=False)
    topology[
        [
            "artifact",
            "configuration_id",
            "layers",
            "median_condition_number",
            "high_cosine_fraction",
            "maximum_absolute_cosine",
        ]
    ].to_csv(SUMMARY / "conditioning.csv", index=False)

    reconstruction = _load("reconstruction")
    reconstruction = reconstruction[
        reconstruction["baseline_correct"].astype(bool)
        & reconstruction["reconstruction_fraction"].notna()
    ].copy()
    reconstruction["clipped"] = reconstruction["reconstruction_fraction"].clip(0, 1)
    reconstruction_summary = (
        reconstruction.groupby("measurement_layer", as_index=False)
        .agg(
            median_clipped_reconstruction_fraction=("clipped", "median"),
            mean_reconstruction_fraction=("reconstruction_fraction", "mean"),
            n=("reconstruction_fraction", "size"),
        )
        .sort_values("measurement_layer")
    )
    reconstruction_summary.to_csv(SUMMARY / "reconstruction.csv", index=False)

    winner = _select_candidate(topology)
    layers = json.loads(winner["layers"])
    winning_rows = all_rows[
        (all_rows["artifact"] == winner["artifact"])
        & (all_rows["configuration_id"] == winner["configuration_id"])
        & all_rows["baseline_correct"].astype(bool)
    ]
    config = json.loads(Path("configs/v2/h0r_diagnostic.json").read_text())
    candidate_path = Path("configs/v2/h0r_candidate_protocol.json")
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_locked_control_open",
        "diagnostic_run_ids": sorted(all_rows["run_id"].unique().tolist()),
        "operation": winner["operation"],
        "layers": layers,
        "normalized_layer_depths": [layer / 63 for layer in layers],
        "position_mask": winner["position_mask"],
        "alpha": float(winner["alpha"]),
        "vector_normalization": "unit_l2_per_layer",
        "coordinate_definition": "pinv([J_l^T W_U[source], J_l^T W_U[target]]) @ h",
        "hook_point": "transformer_block_output_residual",
        "source_tokenization_rule": "exactly_one_token_prefer_leading_space",
        "target_tokenization_rule": "exactly_one_token_prefer_leading_space",
        "validity_domain": config["validity_domain"],
        "expected_delta_rms_range": [
            float(winning_rows["mean_delta_rms_ratio"].quantile(0.025)),
            float(winning_rows["mean_delta_rms_ratio"].quantile(0.975)),
        ],
        "diagnostic_selection": {
            "configuration_id": winner["configuration_id"],
            "mean_rank": float(winner["mean_rank"]),
            "n_eligible": int(winner["n_eligible"]),
            "mean_target_logodds_gain": float(winner["mean_target_logodds_gain"]),
            "gain_ci_95": [float(winner["gain_ci_low"]), float(winner["gain_ci_high"])],
            "positive_gain_fraction": float(winner["positive_gain_fraction"]),
            "target_top1_fraction": float(winner["target_top1_fraction"]),
            "mean_output_kl_nats": float(winner["mean_output_kl_nats"]),
            "mean_delta_rms_ratio": float(winner["mean_delta_rms_ratio"]),
            "minimum_function_effect": float(winner["minimum_function_effect"]),
            "median_condition_number": float(winner["median_condition_number"]),
            "high_cosine_fraction": float(winner["high_cosine_fraction"]),
        },
        "prospective_thresholds": {
            "argument_control": {
                "minimum_baseline_accuracy": 0.80,
                "minimum_mean_gain": 0.75,
                "minimum_positive_gain_fraction": 0.70,
                "minimum_target_top1_fraction": 0.20,
                "minimum_semantic_minus_control_gain": 0.50,
                "require_bootstrap_ci_above_zero": True,
            },
            "intermediate_control": {
                "minimum_mean_gain": 0.50,
                "minimum_positive_gain_fraction": 0.65,
                "require_bootstrap_ci_above_zero": True,
                "require_semantic_above_random_and_unrelated": True,
                "require_direct_answer_not_comparable": True,
            },
        },
    }
    if candidate_path.exists():
        frozen = json.loads(candidate_path.read_text())
        selected_fields = ("operation", "layers", "position_mask", "alpha")
        mismatches = [
            field for field in selected_fields if frozen[field] != protocol[field]
        ]
        if mismatches:
            raise RuntimeError(
                "analysis no longer reproduces frozen candidate fields: "
                + ", ".join(mismatches)
            )
        protocol = frozen
    else:
        raise RuntimeError(
            "refusing to create a new candidate outside the original selection commit"
        )
    _make_figures(all_rows, topology, winner)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
