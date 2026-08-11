from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jspace_policy.analysis import (
    behavior_summary,
    paired_policy_shift,
    paired_readout_trajectories,
)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})


def plot_behavior_gate(frame: pd.DataFrame, output: Path) -> None:
    _style()
    summary = behavior_summary(frame)
    summary["cell"] = summary["world_state"] + " / " + summary["policy"]
    styles = list(summary["policy_style"].drop_duplicates())
    fig, axes = plt.subplots(
        2,
        len(styles),
        figsize=(7 * len(styles), 9),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    axes = np.asarray(axes).reshape(2, len(styles))
    for column, style in enumerate(styles):
        part = summary[summary["policy_style"] == style].reset_index(drop=True)
        x = np.arange(len(part))
        accuracy_error = np.vstack(
            [part["accuracy"] - part["accuracy_low"], part["accuracy_high"] - part["accuracy"]]
        )
        accuracy_axis = axes[0, column]
        accuracy_axis.errorbar(
            x,
            part["accuracy"],
            yerr=accuracy_error,
            fmt="o",
            capsize=4,
            markersize=8,
            color="#176B87",
        )
        accuracy_axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
        accuracy_axis.set_title(f"{style.capitalize()} policy wording")
        accuracy_axis.set_ylim(0, 1.05)
        accuracy_axis.set_ylabel("Policy-following accuracy")

        margin_error = np.vstack(
            [part["margin"] - part["margin_low"], part["margin_high"] - part["margin"]]
        )
        margin_axis = axes[1, column]
        margin_axis.errorbar(
            x,
            part["margin"],
            yerr=margin_error,
            fmt="o",
            capsize=4,
            markersize=8,
            color="#D97745",
        )
        margin_axis.axhline(0, color="0.35", linewidth=1)
        margin_axis.set_ylabel("Correct − alternative log-probability")
        margin_axis.set_xticks(x, part["cell"], rotation=30, ha="right")

    fig.suptitle(
        "Behavioral gate: every cell must be accurate and confidently positive", y=1.01
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_paired_policy_shift(frame: pd.DataFrame, output: Path) -> None:
    _style()
    paired = paired_policy_shift(frame)
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.pointplot(
        data=paired,
        x="world_state",
        y="conceal_minus_reveal_truth_score",
        hue="policy_style",
        errorbar=("ci", 95),
        dodge=0.25,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("World state")
    axis.set_ylabel("Conceal − reveal truth-aligned log-odds")
    axis.set_title("Matched policy change should be negative in both world states")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_intervention_sign_flip(frame: pd.DataFrame, output: Path) -> None:
    required = {"world_state", "delta_literal_logodds", "intervention_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"intervention table missing: {sorted(missing)}")
    _style()
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.pointplot(
        data=frame,
        x="world_state",
        y="delta_literal_logodds",
        hue="intervention_type",
        errorbar=("ci", 95),
        dodge=0.3,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Change in log p(A) − log p(B)")
    axis.set_title("Policy intervention requires opposite literal effects")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_readout_trajectories(frame: pd.DataFrame, output: Path) -> None:
    """Plot scenario-level observational policy and fact contrasts over depth."""
    _style()
    policy, fact = paired_readout_trajectories(frame)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    panels = [
        (
            axes[0],
            policy,
            "Policy-family contrast",
            "Conceal-prompt minus reveal-prompt coordinate",
        ),
        (
            axes[1],
            fact,
            "Fact-family contrast",
            "A-prompt minus B-prompt coordinate",
        ),
    ]
    palette = {"explicit": "#176B87", "indirect": "#D97745"}
    for axis, data, title, ylabel in panels:
        for style, part in data.groupby("policy_style", sort=True):
            scenario = part.pivot(
                index="layer_fraction", columns="scenario_id", values="effect"
            ).sort_index()
            x = scenario.index.to_numpy()
            axis.plot(x, scenario.to_numpy(), color=palette[style], alpha=0.12, linewidth=1)
            axis.plot(
                x,
                scenario.mean(axis=1),
                color=palette[style],
                linewidth=2.8,
                label=f"{style.capitalize()} wording",
            )
        axis.axhline(0, color="0.25", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Normalized layer depth")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=True)
    fig.suptitle(
        "Observational J-lens discovery: paired contrasts across six base prompts", y=1.02
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
