"""Regenerate all V2 Stage 1 summaries, figures, and the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jspace_policy.stage1_analysis import (
    classification_metrics,
    paired_policy_effect,
    scenario_bootstrap,
)

ROOT = Path("results/v2_stage1")


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "raw" / name).read_text(encoding="utf-8"))


def _mean(rows: list[dict[str, Any]], column: str) -> float:
    return float(np.mean([float(row[column]) for row in rows]))


def _flatten_layers(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        for layer in row["layer_scores"]:
            records.append(
                {
                    "condition_id": row["condition_id"],
                    "substage": row["substage"],
                    "split": row["split"],
                    "family": row["family"],
                    "base_scenario_id": row["base_scenario_id"],
                    "world_state_id": row["world_state_id"],
                    "policy_id": row["policy_id"],
                    **{
                        key: value
                        for key, value in layer.items()
                        if not key.endswith("candidate_scores")
                    },
                }
            )
    return pd.DataFrame.from_records(records)


def _behavior_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for result in results:
        phase = result["metadata"]["phase"]
        for substage, summary in result["summary"]["substages"].items():
            records.append(
                {
                    "phase": phase,
                    "substage": substage,
                    "n_rows": summary["n_rows"],
                    "overall_accuracy": summary["overall_accuracy"],
                    "minimum_cell_accuracy": min(summary["cell_accuracy"].values()),
                    "minimum_family_accuracy": min(summary["family_accuracy"].values()),
                    "gate_pass": summary["gate_pass"],
                }
            )
    return pd.DataFrame.from_records(records)


def _effect_summaries(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    draws = int(config["statistics"]["bootstrap_draws"])
    seed = int(config["statistics"]["bootstrap_seed"])
    effects = []
    families = []
    for substage in ("1A", "1B"):
        part = [row for row in rows if row["substage"] == substage]
        transformed = [row for row in part if row["policy_id"] == "M"]
        estimands = {
            "transformed_band_K": lambda sample: _mean(
                [row for row in sample if row["policy_id"] == "M"], "band_K"
            ),
            "report_preparation_delta_M": lambda sample: paired_policy_effect(sample, "band_M"),
            "state_policy_delta_K": lambda sample: paired_policy_effect(sample, "band_K"),
            "state_report_interaction": lambda sample: (
                paired_policy_effect(sample, "band_M") - paired_policy_effect(sample, "band_K")
            ),
            "transformed_logit_lens_K": lambda sample: float(
                np.mean(
                    [
                        np.mean([layer["logit_lens_K"] for layer in row["layer_scores"]])
                        for row in sample
                        if row["policy_id"] == "M"
                    ]
                )
            ),
            "transformed_output_K": lambda sample: _mean(
                [row for row in sample if row["policy_id"] == "M"], "output_K"
            ),
        }
        for offset, (name, statistic) in enumerate(estimands.items()):
            point, low, high = scenario_bootstrap(
                part,
                statistic,
                draws=draws,
                seed=seed + offset + (0 if substage == "1A" else 100),
            )
            effects.append(
                {
                    "substage": substage,
                    "estimand": name,
                    "point": point,
                    "ci_95_low": low,
                    "ci_95_high": high,
                }
            )
        for family in sorted({row["family"] for row in transformed}):
            family_rows = [row for row in transformed if row["family"] == family]
            families.append(
                {
                    "substage": substage,
                    "family": family,
                    "n_base_scenarios": len({row["base_scenario_id"] for row in family_rows}),
                    "mean_transformed_band_K": _mean(family_rows, "band_K"),
                    "positive_direction": _mean(family_rows, "band_K") > 0,
                }
            )
    return pd.DataFrame.from_records(effects), pd.DataFrame.from_records(families)


def _probe_summaries(rows: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    draws = int(config["statistics"]["bootstrap_draws"])
    seed = int(config["statistics"]["bootstrap_seed"])
    records = []
    definitions = (
        (
            "state_all",
            "world_state_id",
            "state_probe_prediction",
            "state_probe_probabilities",
            None,
        ),
        (
            "report_all",
            "report_target",
            "report_probe_prediction",
            "report_probe_probabilities",
            None,
        ),
        (
            "state_truth_trained_to_transformed",
            "world_state_id",
            "cross_truth_trained_state_prediction",
            "cross_truth_trained_state_probabilities",
            "M",
        ),
        (
            "state_transformed_trained_to_truth",
            "world_state_id",
            "cross_transformed_trained_state_prediction",
            "cross_transformed_trained_state_probabilities",
            "T",
        ),
    )
    for substage in ("1A", "1B"):
        for index, (name, target, prediction, probability, policy) in enumerate(definitions):
            part = [
                row
                for row in rows
                if row["substage"] == substage
                and (policy is None or row["policy_id"] == policy)
            ]
            metrics = classification_metrics(
                part,
                target_column=target,
                prediction_column=prediction,
                probability_column=probability,
            )
            point, low, high = scenario_bootstrap(
                part,
                lambda sample, target=target, prediction=prediction: float(
                    np.mean([int(row[target]) == int(row[prediction]) for row in sample])
                ),
                draws=draws,
                seed=seed + 500 + index + (0 if substage == "1A" else 100),
            )
            records.append(
                {
                    "substage": substage,
                    "probe": name,
                    "n_rows": len(part),
                    **metrics,
                    "accuracy": point,
                    "accuracy_ci_95_low": low,
                    "accuracy_ci_95_high": high,
                }
            )
    return pd.DataFrame.from_records(records)


def _plots(layer_frame: pd.DataFrame, effects: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    transformed = layer_frame[layer_frame["policy_id"] == "M"]
    for substage, group in transformed.groupby("substage"):
        trajectory = group.groupby("layer", as_index=False)["K"].mean()
        axes[0].plot(trajectory["layer"], trajectory["K"], marker="o", label=substage)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Locked transformed true-state evidence", xlabel="Layer", ylabel="Mean K")
    axes[0].legend()

    primary = effects[effects["estimand"] == "transformed_band_K"]
    x = np.arange(len(primary))
    axes[1].bar(x, primary["point"], color=["#4C78A8", "#F58518"])
    axes[1].errorbar(
        x,
        primary["point"],
        yerr=np.vstack(
            [
                primary["point"].to_numpy() - primary["ci_95_low"].to_numpy(),
                primary["ci_95_high"].to_numpy() - primary["point"].to_numpy(),
            ]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, primary["substage"])
    axes[1].set(title="Primary band retention", ylabel="Mean K with 95% bootstrap CI")
    figure.tight_layout()
    figure.savefig(output / "stage1_primary_results.png", dpi=180)
    plt.close(figure)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _write_final_report(
    behavior: pd.DataFrame,
    effects: pd.DataFrame,
    families: pd.DataFrame,
    probes: pd.DataFrame,
    metadata: dict[str, Any],
    output: Path,
) -> None:
    conclusions = {}
    lines = [
        "# Stage 1 final report: latent state–report dissociation",
        "",
        "This report is generated deterministically from the immutable Stage 1 raw results.",
        "",
        "## Execution and behavioral validity",
        "",
        f"The locked run used model revision `{metadata['model_revision_resolved']}` "
        "and the pinned Jacobian Lens over layers "
        f"{metadata['primary_layers'][0]}–{metadata['primary_layers'][-1]} at "
        "the final pre-output prompt position.",
        "",
        "| Phase | Substage | Accuracy | Minimum cell | Minimum family | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in behavior.itertuples(index=False):
        lines.append(
            f"| {row.phase} | {row.substage} | {_fmt(row.overall_accuracy)} | "
            f"{_fmt(row.minimum_cell_accuracy)} | {_fmt(row.minimum_family_accuracy)} | "
            f"{'pass' if row.gate_pass else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Confirmatory results",
            "",
            "| Substage | Transformed K (95% CI) | Positive locked families | "
            "State probe accuracy (95% CI) | Criterion |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for substage in ("1A", "1B"):
        effect = effects[
            (effects.substage == substage) & (effects.estimand == "transformed_band_K")
        ].iloc[0]
        family_part = families[families.substage == substage]
        positive = int(family_part.positive_direction.sum())
        probe = probes[(probes.substage == substage) & (probes.probe == "state_all")].iloc[0]
        passed = effect.ci_95_low > 0 and positive >= 2 and probe.accuracy_ci_95_low > 0.25
        conclusions[substage] = passed
        lines.append(
            f"| {substage} | {_fmt(effect.point)} "
            f"[{_fmt(effect.ci_95_low)}, {_fmt(effect.ci_95_high)}] | "
            f"{positive}/3 | {_fmt(probe.accuracy)} [{_fmt(probe.accuracy_ci_95_low)}, "
            f"{_fmt(probe.accuracy_ci_95_high)}] | {'pass' if passed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The report-preparation, state-policy, interaction, baseline, "
            "family-specific, and full probe metrics are in the generated CSV "
            "summaries. The figure is generated from the same raw rows.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if conclusions["1A"] and conclusions["1B"]:
        lines.append(
            "Both substages meet the frozen criterion. In these controlled tasks, "
            "Qwen3.6-27B retained prospectively decodable information about "
            "explicitly supplied and internally inferred task "
            "state before emitting a policy-transformed report."
        )
    elif conclusions["1A"]:
        lines.append(
            "Stage 1A meets the frozen criterion but Stage 1B does not. The licensed "
            "conclusion is limited "
            "to persistence of explicitly supplied state under report transformation."
        )
    else:
        lines.append(
            "The frozen Stage 1 criterion is not met. The study does not establish "
            "latent state–report "
            "dissociation, and Stage 2 is not licensed by this result."
        )
    lines.extend(
        [
            "",
            "This observational result does not establish deception, deceptive intent, "
            "consciousness, scheming, a general-purpose lie detector, or causal use "
            "of the decoded state. No Stage 0 or H0R "
            "failure is reinterpreted.",
            "",
            "## Reproducibility",
            "",
            "- Dataset, configuration, probe, model, lens, and code identifiers are "
            "recorded in raw manifests.",
            "- All uncertainty uses the frozen 2,000-draw base-scenario bootstrap.",
            "- Run `uv run python scripts/analyze_stage1.py --phase locked` to "
            "regenerate this report, tables, and figure.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev", "locked"), required=True)
    args = parser.parse_args()
    config = json.loads(Path("configs/v2/stage1.json").read_text(encoding="utf-8"))
    summaries = ROOT / "summaries"
    figures = ROOT / "figures"
    summaries.mkdir(parents=True, exist_ok=True)
    behavior_results = [_load("behavior_dev.json")]
    if args.phase == "locked":
        behavior_results.append(_load("behavior_locked.json"))
    behavior = _behavior_table(behavior_results)
    behavior.to_csv(summaries / "behavior_gates.csv", index=False)

    mechanistic = _load(f"mechanistic_{args.phase}.json")
    rows = mechanistic["rows"]
    layer_frame = _flatten_layers(rows)
    layer_frame.groupby(["substage", "split", "policy_id", "layer"], as_index=False)[
        ["K", "Q", "M", "D", "logit_lens_K", "logit_lens_M"]
    ].mean().to_csv(summaries / f"layer_trajectories_{args.phase}.csv", index=False)
    effects, families = _effect_summaries(rows, config)
    probes = _probe_summaries(rows, config)
    effects.to_csv(summaries / f"primary_effects_{args.phase}.csv", index=False)
    families.to_csv(summaries / f"family_effects_{args.phase}.csv", index=False)
    probes.to_csv(summaries / f"probe_metrics_{args.phase}.csv", index=False)
    _plots(layer_frame, effects, figures)
    if args.phase == "locked":
        _write_final_report(
            behavior,
            effects,
            families,
            probes,
            mechanistic["metadata"],
            Path("docs/v2/stage1-final-report.md"),
        )


if __name__ == "__main__":
    main()
