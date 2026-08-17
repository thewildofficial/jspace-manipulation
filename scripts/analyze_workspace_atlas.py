from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("results/v2_workspace_atlas")
CONFIG_PATH = Path("configs/v2/workspace_atlas/experiment.json")
DATASET_PATH = Path("configs/v2/workspace_atlas/dataset.json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _frame_block(frame: pd.DataFrame) -> str:
    """Render a dependency-free, readable table for generated Markdown reports."""
    return (
        "```text\n"
        + frame.to_string(index=False, float_format=lambda value: f"{value:.3f}")
        + "\n```"
    )


def _behavior_tables(
    behavior: dict[str, Any], dataset: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = {row["condition_id"]: row for row in dataset["rows"]}
    rows = []
    for observed in behavior["rows"]:
        expected = truth[observed["condition_id"]]
        values = [float(value) for value in expected["solver"]["action_values"]]
        selected = expected["candidates"].index(observed["legal_choice"])
        regret = max(values) - values[selected]
        probabilities = np.exp(np.asarray(observed["legal_action_log_probs"], dtype=float))
        entropy = -float(np.sum(probabilities * np.log(probabilities.clip(1e-12))))
        rows.append(
            {
                **{
                    key: expected[key]
                    for key in (
                        "condition_id",
                        "matched_group_id",
                        "game",
                        "split",
                        "private_state",
                        "belief",
                        "objective",
                        "value_margin",
                        "strategy",
                        "expected_action",
                    )
                },
                "top1_text": observed["top1_text"],
                "formatting_compliant": observed["formatting_compliant"],
                "correct": observed["correct"],
                "legal_choice": observed["legal_choice"],
                "legal_choice_correct": observed["legal_choice_correct"],
                "regret": regret,
                "output_entropy": entropy,
                "chosen_probability": float(probabilities[selected]),
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["split", "game"], as_index=False)
        .agg(
            n=("condition_id", "size"),
            accuracy=("correct", "mean"),
            legal_choice_accuracy=("legal_choice_correct", "mean"),
            formatting_compliance=("formatting_compliant", "mean"),
            mean_regret=("regret", "mean"),
            mean_output_entropy=("output_entropy", "mean"),
        )
        .sort_values(["split", "game"])
    )
    return detail, summary


def _mechanistic_tables(
    mechanistic: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    token_rows: list[dict[str, Any]] = []
    commitment_rows: list[dict[str, Any]] = []
    for row in mechanistic["rows"]:
        prompt = row["task_text"].lower()
        layers = [int(layer["layer"]) for layer in row["layer_readouts"]]
        output_choice = "ABC"[int(np.argmax(row["output_legal_action_log_probs"]))]
        layer_choices = [layer["logit_lens_legal_choice"] for layer in row["layer_readouts"]]
        commitment = None
        for index, choice in enumerate(layer_choices):
            if choice == output_choice and all(
                later == output_choice for later in layer_choices[index:]
            ):
                commitment = layers[index]
                break
        commitment_rows.append(
            {
                "condition_id": row["condition_id"],
                "matched_group_id": row["matched_group_id"],
                "game": row["game"],
                "split": row["split"],
                "strategy": row["strategy"],
                "expected_action": row["expected_action"],
                "output_choice": output_choice,
                "commitment_layer": commitment,
            }
        )
        for layer in row["layer_readouts"]:
            for rank, (token_id, text, score) in enumerate(
                zip(
                    layer["top_token_ids"],
                    layer["top_token_texts"],
                    layer["top_scores"],
                    strict=True,
                ),
                start=1,
            ):
                normalized = str(text).strip().lower()
                token_rows.append(
                    {
                        "condition_id": row["condition_id"],
                        "matched_group_id": row["matched_group_id"],
                        "game": row["game"],
                        "split": row["split"],
                        "strategy": row["strategy"],
                        "expected_action": row["expected_action"],
                        "layer": int(layer["layer"]),
                        "rank": rank,
                        "token_id": int(token_id),
                        "token_text": text,
                        "score": float(score),
                        "prompt_echo": bool(normalized and normalized in prompt),
                    }
                )
    tokens = pd.DataFrame(token_rows)
    commitments = pd.DataFrame(commitment_rows)
    echo = tokens.groupby(["game", "split", "layer"], as_index=False).agg(
        prompt_echo_rate=("prompt_echo", "mean"), unique_tokens=("token_id", "nunique")
    )
    return tokens, commitments, echo


def _emergent_inventory(tokens: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = tokens[(tokens["rank"] <= 10) & ~tokens["prompt_echo"]]
    for (game, layer), part in subset.groupby(["game", "layer"]):
        counts = Counter(part["token_text"].astype(str))
        for rank, (token, count) in enumerate(counts.most_common(20), start=1):
            rows.append(
                {
                    "game": game,
                    "layer": layer,
                    "frequency_rank": rank,
                    "token_text": token,
                    "count": count,
                    "row_fraction": count / part["condition_id"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def _plot_behavior(summary: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid")
    open_rows = summary[summary["split"].isin(["discovery", "validation"])]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=open_rows, x="game", y="accuracy", hue="split", ax=axes[0])
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Full-vocabulary optimal-action accuracy")
    sns.barplot(data=open_rows, x="game", y="mean_regret", hue="split", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Mean solver regret")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_probe(probes: pd.DataFrame, output: Path) -> None:
    subset = probes[
        (probes["status"] == "ok")
        & (probes["variable"] == "strategy")
        & (probes["representation"].isin(["residual", "jspace"]))
    ].copy()
    if subset.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, game in zip(axes.flat, sorted(subset["game"].unique()), strict=False):
        part = subset[subset["game"] == game]
        sns.lineplot(
            data=part,
            x="layer",
            y="balanced_accuracy",
            hue="representation",
            ax=axis,
        )
        axis.set_title(game)
        axis.set_ylim(0, 1.05)
    figure.suptitle("Exploratory strategy decoding across unseen renderings")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _report(
    phase: str,
    behavior: dict[str, Any],
    behavior_summary: pd.DataFrame,
    probes: pd.DataFrame | None,
    commitments: pd.DataFrame | None,
) -> str:
    lines = [
        "# V2-E1 Strategic Workspace Atlas",
        "",
        f"Phase: **{phase}**.",
        "",
        "This is an observational exploratory result. It does not reopen the failed "
        "causal gate.",
        "",
        "## Behavioral gate",
        "",
        f"- Gate: **{'pass' if behavior['summary']['gate_pass'] else 'fail'}**",
        f"- Formatting compliance: {behavior['summary']['formatting_compliance']:.1%}",
        "- Exact-game optimal accuracy: "
        f"{behavior['summary']['exact_game_optimal_accuracy']:.1%}",
        "- Minimum exact-game family accuracy: "
        f"{behavior['summary']['exact_game_minimum_family_accuracy']:.1%}",
        "",
        _frame_block(behavior_summary),
    ]
    if probes is not None:
        strategy = probes[
            (probes["status"] == "ok")
            & (probes["variable"] == "strategy")
            & probes["balanced_accuracy"].notna()
        ]
        best = (
            strategy.sort_values("balanced_accuracy", ascending=False)
            .groupby(["game", "representation"], as_index=False)
            .first()[["game", "representation", "layer", "balanced_accuracy"]]
        )
        lines.extend(
            [
                "",
                "## Exploratory strategy decoding",
                "",
                "Best layer is selected post hoc and is descriptive only.",
                "",
                _frame_block(best),
            ]
        )
    if commitments is not None and not commitments.empty:
        commitment_summary = commitments.groupby("game", as_index=False).agg(
            median_commitment_layer=("commitment_layer", "median"),
            measured=("commitment_layer", "count"),
        )
        lines.extend(
            [
                "",
                "## Output commitment depth",
                "",
                _frame_block(commitment_summary),
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact map",
            "",
            "- `raw/`: immutable behavior and mechanistic returns.",
            "- `summaries/`: behavior, probes, token inventory, echo rates, and "
            "commitment tables.",
            "- `atlas/`: raw top-token inventory suitable for qualitative inspection.",
            "- `figures/`: deterministic behavioral and probe plots.",
            "",
            "All mechanistic patterns in the open phase are discovery findings until a "
            "replication freeze is committed.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(phase: str) -> None:
    _load(CONFIG_PATH)
    dataset = _load(DATASET_PATH)
    behavior = _load(ROOT / "raw" / f"behavior_{phase}.json")
    detail, summary = _behavior_tables(behavior, dataset)
    _write_csv(detail, ROOT / "summaries" / f"behavior_detail_{phase}.csv")
    _write_csv(summary, ROOT / "summaries" / f"behavior_summary_{phase}.csv")
    _plot_behavior(summary, ROOT / "figures" / f"behavior_{phase}.png")

    mechanistic_path = ROOT / "raw" / f"mechanistic_{phase}.json"
    probes = None
    commitments = None
    if mechanistic_path.exists():
        mechanistic = _load(mechanistic_path)
        probes = pd.DataFrame(mechanistic["probe_metrics"])
        tokens, commitments, echo = _mechanistic_tables(mechanistic)
        emergent = _emergent_inventory(tokens)
        _write_csv(probes, ROOT / "summaries" / f"probe_metrics_{phase}.csv")
        _write_csv(commitments, ROOT / "summaries" / f"commitment_depth_{phase}.csv")
        _write_csv(echo, ROOT / "summaries" / f"prompt_echo_{phase}.csv")
        _write_csv(emergent, ROOT / "summaries" / f"emergent_tokens_{phase}.csv")
        _write_csv(tokens, ROOT / "atlas" / f"top_tokens_{phase}.csv")
        _plot_probe(probes, ROOT / "figures" / f"strategy_decoding_{phase}.png")

    report = _report(phase, behavior, summary, probes, commitments)
    (ROOT / f"README_{phase}.md").write_text(report, encoding="utf-8")
    print(json.dumps({"phase": phase, "gate_pass": behavior["summary"]["gate_pass"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("open", "locked"), default="open")
    args = parser.parse_args()
    analyze(args.phase)


if __name__ == "__main__":
    main()
