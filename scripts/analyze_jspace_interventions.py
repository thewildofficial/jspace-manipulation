from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jspace_policy.jspace_interventions import cluster_bootstrap_mean

RESULTS = Path("results/v3_jspace_interventions")
SEED = 1729
BOOTSTRAP_DRAWS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["burned", "fresh"])
    return parser.parse_args()


def load_artifact(phase: str) -> tuple[dict[str, Any], pd.DataFrame]:
    name = "burned_replication.json.gz" if phase == "burned" else "fresh_interventions.json.gz"
    with gzip.open(RESULTS / "raw" / name, "rt") as handle:
        artifact = json.load(handle)
    return artifact, pd.DataFrame(artifact["rows"])


def ci(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    return cluster_bootstrap_mean(
        frame[column].astype(float).tolist(),
        frame["cluster_id"].astype(str).tolist(),
        draws=BOOTSTRAP_DRAWS,
        seed=SEED,
    )


def summarize_primary(rows: pd.DataFrame) -> pd.DataFrame:
    primary = rows[
        rows["condition_id"].str.startswith("native_raw_a") & rows["eligible"]
    ].copy()
    summaries = []
    for (alpha, control), part in primary.groupby(["alpha", "control_kind"], sort=True):
        lower, upper = ci(part, "delta_target_minus_source_logodds")
        summaries.append(
            {
                "alpha": alpha,
                "control_kind": control,
                "raw_n": len(part),
                "cluster_n": part["cluster_id"].nunique(),
                "mean_delta": part["delta_target_minus_source_logodds"].mean(),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "positive_fraction": (part["delta_target_minus_source_logodds"] > 0).mean(),
                "target_top1_fraction": part["target_top1_patched"].mean(),
                "mean_output_kl": part["output_kl_clean_to_patched"].mean(),
                "median_output_kl": part["output_kl_clean_to_patched"].median(),
                "p90_output_kl": part["output_kl_clean_to_patched"].quantile(0.9),
                "p95_output_kl": part["output_kl_clean_to_patched"].quantile(0.95),
                "max_output_kl": part["output_kl_clean_to_patched"].max(),
                "mean_delta_rms_ratio": part["mean_delta_rms_ratio"].mean(),
                "p95_delta_rms_ratio": part["mean_delta_rms_ratio"].quantile(0.95),
                "max_delta_rms_ratio": part["max_delta_rms_ratio"].max(),
            }
        )
    return pd.DataFrame(summaries)


def paired_controls(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = rows[
        rows["condition_id"].str.startswith("native_raw_a") & rows["eligible"]
    ].copy()
    keys = ["cluster_id", "category", "function", "source_argument", "target_argument", "alpha"]
    semantic = primary[primary["control_kind"].eq("semantic")][
        [*keys, "delta_target_minus_source_logodds", "target_top1_patched"]
    ].rename(
        columns={
            "delta_target_minus_source_logodds": "semantic_delta",
            "target_top1_patched": "semantic_top1",
        }
    )
    detail = []
    summaries = []
    for control in ["random_delta_matched", "unrelated_semantic"]:
        comparator = primary[primary["control_kind"].eq(control)][
            [*keys, "delta_target_minus_source_logodds", "target_top1_patched"]
        ].rename(
            columns={
                "delta_target_minus_source_logodds": "control_delta",
                "target_top1_patched": "control_top1",
            }
        )
        merged = semantic.merge(comparator, on=keys, validate="one_to_one")
        merged["control_kind"] = control
        merged["paired_delta"] = merged["semantic_delta"] - merged["control_delta"]
        detail.append(merged)
        for alpha, part in merged.groupby("alpha", sort=True):
            lower, upper = ci(part, "paired_delta")
            summaries.append(
                {
                    "alpha": alpha,
                    "control_kind": control,
                    "raw_n": len(part),
                    "cluster_n": part["cluster_id"].nunique(),
                    "mean_paired_delta": part["paired_delta"].mean(),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "semantic_top1": part["semantic_top1"].mean(),
                    "control_top1": part["control_top1"].mean(),
                    "top1_advantage": part["semantic_top1"].mean()
                    - part["control_top1"].mean(),
                }
            )
    return pd.DataFrame(summaries), pd.concat(detail, ignore_index=True)


def spearman(left: pd.Series, right: pd.Series) -> float:
    return float(np.corrcoef(left.rank(method="average"), right.rank(method="average"))[0, 1])


def loading_summary(rows: pd.DataFrame) -> dict[str, Any]:
    semantic = rows[rows["condition_id"].eq("native_raw_a1_semantic") & rows["eligible"]]
    clusters = semantic.groupby(
        ["cluster_id", "category", "function", "source_argument"], as_index=False
    ).agg(
        mean_delta=("delta_target_minus_source_logodds", "mean"),
        paper_loading=("source_workspace_loading_paper", "first"),
        final_loading=("source_workspace_loading_final", "first"),
    )
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(clusters))
    boot = {"paper": [], "final": []}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = clusters.iloc[rng.choice(indices, len(indices), replace=True)]
        boot["paper"].append(spearman(sample["paper_loading"], sample["mean_delta"]))
        boot["final"].append(spearman(sample["final_loading"], sample["mean_delta"]))
    result: dict[str, Any] = {"cluster_n": len(clusters)}
    for name, column in [("paper", "paper_loading"), ("final", "final_loading")]:
        finite = np.array(boot[name], dtype=float)
        finite = finite[np.isfinite(finite)]
        result[name] = {
            "rho": spearman(clusters[column], clusters["mean_delta"]),
            "ci95": np.quantile(finite, [0.025, 0.975]).tolist(),
        }
    clusters.to_csv(RESULTS / "summaries/loading_effect.csv", index=False)
    return result


def power_grid(rows: pd.DataFrame) -> dict[str, Any]:
    semantic = rows[rows["condition_id"].eq("native_raw_a1_semantic") & rows["eligible"]]
    cluster_means = semantic.groupby("cluster_id")["delta_target_minus_source_logodds"].mean()
    observed_sd = float(cluster_means.std(ddof=1))
    rng = np.random.default_rng(SEED)
    grid = []
    for effect in [0.25, 0.5, 1.0, 2.0]:
        successes = 0
        for _ in range(2000):
            sample = rng.normal(effect, observed_sd, size=48)
            standard_error = sample.std(ddof=1) / np.sqrt(len(sample))
            successes += sample.mean() - 1.96 * standard_error > 0
        grid.append(
            {"true_effect": effect, "estimated_detection_probability": successes / 2000}
        )
    return {
        "method": (
            "normal simulation using burned alpha-1 source-prompt cluster SD "
            "and 48 planned fresh clusters"
        ),
        "burned_cluster_sd": observed_sd,
        "grid": grid,
        "minimum_effect_0_5_power_at_least_0_8": next(
            row["estimated_detection_probability"] for row in grid if row["true_effect"] == 0.5
        )
        >= 0.8,
    }


def classify(
    rows: pd.DataFrame, summaries: pd.DataFrame, controls: pd.DataFrame
) -> dict[str, Any]:
    semantic = summaries[
        summaries["alpha"].eq(1.0) & summaries["control_kind"].eq("semantic")
    ].iloc[0]
    alpha_controls = controls[controls["alpha"].eq(1.0)]
    tier1 = bool(
        semantic["ci95_lower"] > 0
        and semantic["positive_fraction"] >= 0.7
        and (alpha_controls["ci95_lower"] > 0).all()
        and (alpha_controls["mean_paired_delta"] >= 0.5).all()
    )
    selected = rows[rows["condition_id"].eq("native_raw_a1_semantic") & rows["eligible"]]
    geometry = selected.dropna(subset=["median_condition_number", "max_basis_cosine"])
    tier2 = bool(
        tier1
        and selected["output_kl_clean_to_patched"].mean() <= 1.0
        and selected["mean_delta_rms_ratio"].mean() <= 0.25
        and geometry["median_condition_number"].median() <= 20.0
        and (geometry["max_basis_cosine"].abs() > 0.95).mean() <= 0.05
    )
    function_positive = (
        selected.groupby(["category", "function"])["delta_target_minus_source_logodds"].mean()
        > 0
    ).mean()
    tier3 = bool(
        tier1
        and selected["target_top1_patched"].mean() >= 0.2
        and (alpha_controls["top1_advantage"] >= 0.1).all()
        and function_positive >= 0.75
    )
    label = "A — Replication failure"
    if tier1:
        label = "B — Directional causal replication"
    if tier2:
        label = "C — Low-distortion causal replication"
    if tier3:
        label = "D — Behavioral semantic substitution"
    return {
        "tier_1_directional": tier1,
        "tier_2_low_distortion_on_average": tier2,
        "tier_3_behavioral_substitution": tier3,
        "classification": label,
        "positive_function_fraction": float(function_positive),
    }


def figures(summaries: pd.DataFrame, rows: pd.DataFrame) -> None:
    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, name, ylabel in [
        ("mean_delta", "alpha_vs_logodds.png", "Mean target-vs-source log-odds gain"),
        ("target_top1_fraction", "alpha_vs_top1.png", "Target answer top-1 fraction"),
        ("mean_output_kl", "alpha_vs_kl.png", "Mean KL(clean || patched)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for control, part in summaries.groupby("control_kind"):
            ax.plot(part["alpha"], part[metric], marker="o", label=control)
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / name, dpi=180)
        plt.close(fig)

    selected = rows[rows["condition_id"].eq("native_raw_a1_semantic") & rows["eligible"]]
    clusters = selected.groupby("cluster_id", as_index=False).agg(
        loading=("source_workspace_loading_final", "first"),
        effect=("delta_target_minus_source_logodds", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(clusters["loading"], clusters["effect"], alpha=0.8)
    ax.set_xlabel("Final-position source loading")
    ax.set_ylabel("Mean alpha-1 semantic effect")
    fig.tight_layout()
    fig.savefig(figure_dir / "loading_vs_effect.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    artifact, rows = load_artifact(args.phase)
    if not artifact["phase_a"]["all_pass"]:
        raise RuntimeError("refusing analysis because Phase A did not pass")
    summary_dir = RESULTS / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_primary(rows)
    controls, detail = paired_controls(rows)
    summaries.to_csv(summary_dir / "primary_effects.csv", index=False)
    controls.to_csv(summary_dir / "control_comparison.csv", index=False)
    detail.to_csv(summary_dir / "control_comparison_detail.csv", index=False)
    power = power_grid(rows)
    (summary_dir / "power_analysis.json").write_text(json.dumps(power, indent=2) + "\n")
    if args.phase == "burned":
        experiment = json.loads(
            Path("configs/v3/jspace_interventions/experiment.json").read_text()
        )
        semantic = summaries[
            summaries["alpha"].eq(1.0) & summaries["control_kind"].eq("semantic")
        ].iloc[0]
        control_means = summaries[
            summaries["alpha"].eq(1.0)
            & summaries["control_kind"].isin(["random_delta_matched", "unrelated_semantic"])
        ].set_index("control_kind")["mean_delta"]
        gate = experiment["phase_b_proceed_gate"]
        result = {
            "phase_a_all_pass": True,
            "alpha1_semantic_mean_delta": semantic["mean_delta"],
            "alpha1_positive_fraction": semantic["positive_fraction"],
            "alpha1_random_mean_delta": control_means["random_delta_matched"],
            "alpha1_unrelated_mean_delta": control_means["unrelated_semantic"],
        }
        result["proceed_to_fresh"] = bool(
            result["alpha1_semantic_mean_delta"] >= gate["minimum_alpha1_semantic_mean_delta"]
            and result["alpha1_positive_fraction"] >= gate["minimum_alpha1_positive_fraction"]
            and result["alpha1_semantic_mean_delta"] > result["alpha1_random_mean_delta"]
            and result["alpha1_semantic_mean_delta"] > result["alpha1_unrelated_mean_delta"]
        )
        (summary_dir / "phase_b_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        loading = loading_summary(rows)
        evidence = classify(rows, summaries, controls)
        evidence["loading"] = loading
        (summary_dir / "evidence_tiers.json").write_text(json.dumps(evidence, indent=2) + "\n")
    figures(summaries, rows)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "rows": len(rows),
                "phase_a": artifact["phase_a"],
                "power": power,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
