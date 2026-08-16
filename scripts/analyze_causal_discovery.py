from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jspace_policy.analysis import grouped_intervention_summary

ROOT = Path("results/27b_causal_discovery")
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"
FIGURES = ROOT / "figures"
DISCOVERY = RAW / "interventions_discovery.parquet"
STABILITY = RAW / "interventions_stability.parquet"
OBSERVATIONAL = Path(
    "results/27b_observational_pilot/summaries/lens_policy_trajectory_27b.csv"
)

COLORS = {
    43: "#7A5195",
    44: "#536FAE",
    45: "#2F849C",
    46: "#36A071",
    47: "#D17A3B",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 16,
            "legend.fontsize": 9,
        }
    )


def _errorbar(axis, data: pd.DataFrame, x: str, y: str, color: str, label: str) -> None:
    axis.errorbar(
        data[x],
        data[y],
        yerr=np.vstack([data[y] - data[f"{y}_low"], data[f"{y}_high"] - data[y]]),
        marker="o",
        linewidth=2,
        capsize=3,
        color=color,
        label=label,
    )


def main() -> None:
    _style()
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    discovery = pd.read_parquet(DISCOVERY)
    stability = pd.read_parquet(STABILITY)
    for frame in (discovery, stability):
        frame["abs_delta_fact_score_standardized"] = frame[
            "delta_fact_score_standardized"
        ].abs()
        frame["behavior_flip"] = (~frame["intervened_correct"]).astype(float)
        world_sign = np.where(frame["world_state"].eq("A"), 1.0, -1.0)
        frame["delta_true_score"] = world_sign * frame["delta_literal_logodds"]

    discovery_policy = discovery[
        discovery["query_type"].eq("public_report")
        & discovery["intervention_type"].eq("steer")
        & discovery["target_policy"].eq("conceal")
    ]
    dose = grouped_intervention_summary(
        discovery_policy,
        ["layer", "alpha_sd"],
        [
            "delta_policy_score",
            "abs_delta_fact_score_standardized",
            "delta_residual_rms_ratio",
            "output_kl",
            "behavior_flip",
        ],
    )
    dose.to_csv(SUMMARIES / "discovery_policy_dose.csv", index=False)

    stability_public = stability[stability["query_type"].eq("public_report")]
    controls = grouped_intervention_summary(
        stability_public,
        ["intervention_type", "direction_name", "target_policy", "alpha_sd"],
        [
            "delta_policy_score",
            "abs_delta_fact_score_standardized",
            "delta_residual_rms_ratio",
            "output_kl",
            "behavior_flip",
        ],
    )
    controls.to_csv(SUMMARIES / "stability_controls.csv", index=False)

    policy_stability = stability_public[
        stability_public["intervention_type"].eq("steer")
        & stability_public["target_policy"].eq("conceal")
    ]
    sign_flip = grouped_intervention_summary(
        policy_stability,
        ["world_state"],
        ["delta_literal_logodds", "delta_policy_score"],
    )
    sign_flip.to_csv(SUMMARIES / "stability_policy_sign_flip.csv", index=False)

    behavior_groups = stability[
        stability["intervention_type"].eq("steer")
        & stability["target_policy"].eq("conceal")
    ].copy()
    behavior_groups["assay"] = np.select(
        [
            behavior_groups["query_type"].eq("private_use"),
            behavior_groups["policy"].eq("reveal"),
        ],
        ["Private use of fact", "Public reveal prompt"],
        default="Public conceal prompt",
    )
    behavior = grouped_intervention_summary(
        behavior_groups,
        ["assay"],
        ["delta_native_policy_score", "behavior_flip"],
    )
    behavior.to_csv(SUMMARIES / "stability_behavioral_assays.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for layer, part in dose.groupby("layer"):
        _errorbar(
            axes[0],
            part,
            "alpha_sd",
            "delta_policy_score",
            COLORS[int(layer)],
            f"Layer {int(layer)}",
        )
        _errorbar(
            axes[1],
            part,
            "alpha_sd",
            "abs_delta_fact_score_standardized",
            COLORS[int(layer)],
            f"Layer {int(layer)}",
        )
    axes[0].axhline(0.5, color="#A33A3A", linestyle="--", label="0.5 log-odds target")
    axes[0].axhline(0, color="0.25", linewidth=1)
    axes[0].set(
        xlabel="Intervention strength (natural coordinate SDs)",
        ylabel="World-conditioned conceal effect (log-odds)",
        title="Policy effect stays below the 0.5 target",
    )
    axes[1].axhline(0.2, color="#A33A3A", linestyle="--", label="0.2 SD equivalence bound")
    axes[1].set(
        xlabel="Intervention strength (natural coordinate SDs)",
        ylabel="Absolute fact-proxy displacement (SD)",
        title="Fact proxy immediately exceeds the 0.2-SD bound",
    )
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].legend(fontsize=9, ncol=2)
    fig.suptitle("Stage-1 causal discovery dose response (2 base scenarios)")
    fig.savefig(FIGURES / "causal_dose_response.png")
    plt.close(fig)

    selected = stability_public[
        (
            stability_public["direction_name"].eq("policy_conceal_minus_reveal")
            & stability_public["target_policy"].eq("conceal")
        )
        | stability_public["direction_name"].isin(
            ["direct_A", "random_0", "random_1", "empirical_fact_A_minus_B"]
        )
    ].copy()
    selected["intervention"] = selected["direction_name"].replace(
        {
            "policy_conceal_minus_reveal": "Policy family",
            "direct_A": "Direct A",
            "random_0": "Random 0",
            "random_1": "Random 1",
            "empirical_fact_A_minus_B": "Empirical fact",
        }
    )
    selected_sign = grouped_intervention_summary(
        selected,
        ["intervention", "world_state"],
        ["delta_literal_logodds"],
    )
    fig, axis = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    intervention_order = [
        "Policy family",
        "Direct A",
        "Empirical fact",
        "Random 0",
        "Random 1",
    ]
    palette = sns.color_palette("deep", len(intervention_order))
    offsets = np.linspace(-0.24, 0.24, len(intervention_order))
    for intervention, color, offset in zip(
        intervention_order, palette, offsets, strict=True
    ):
        part = (
            selected_sign[selected_sign["intervention"].eq(intervention)]
            .set_index("world_state")
            .loc[["A", "B"]]
        )
        x_positions = np.array([0.0, 1.0]) + offset
        y_values = part["delta_literal_logodds"]
        axis.errorbar(
            x_positions,
            y_values,
            yerr=np.vstack(
                [
                    y_values - part["delta_literal_logodds_low"],
                    part["delta_literal_logodds_high"] - y_values,
                ]
            ),
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=intervention,
        )
    axis.axhline(0, color="0.2", linewidth=1)
    axis.set_xticks([0, 1], ["A", "B"])
    axis.set(
        xlabel="True world state",
        ylabel="Change in log p(A) - log p(B)",
        title="Stability check: the required literal sign flip is tiny",
    )
    axis.legend(title="16-SD intervention", ncol=2)
    fig.savefig(FIGURES / "stability_sign_flip.png")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.7), constrained_layout=True)
    scatter = axis.scatter(
        dose["delta_policy_score"],
        dose["abs_delta_fact_score_standardized"],
        s=55 + 900 * dose["delta_residual_rms_ratio"],
        c=dose["layer"].map(COLORS),
        alpha=0.86,
        edgecolor="white",
        linewidth=0.7,
    )
    del scatter
    for _, row in dose.iterrows():
        if row["alpha_sd"] == 16.0:
            axis.annotate(
                f"L{int(row['layer'])}, {int(row['alpha_sd'])} SD",
                (row["delta_policy_score"], row["abs_delta_fact_score_standardized"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axvline(0.5, color="#A33A3A", linestyle="--")
    axis.axhline(0.2, color="#A33A3A", linestyle="--")
    axis.fill_betweenx([0, 0.2], 0.5, 0.55, color="#4D9B74", alpha=0.15)
    axis.set(
        xlim=(-0.05, 0.55),
        xlabel="World-conditioned conceal effect (log-odds)",
        ylabel="Absolute fact-proxy displacement (SD)",
        title="No tested point enters the causal-separability target region",
    )
    axis.text(0.505, 0.08, "target\nregion", fontsize=9, color="#31704F")
    fig.savefig(FIGURES / "policy_fact_tradeoff.png")
    plt.close(fig)

    order = ["Public reveal prompt", "Public conceal prompt", "Private use of fact"]
    behavior = behavior.set_index("assay").loc[order].reset_index()
    fig, axis = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    x = np.arange(len(behavior))
    y = behavior["delta_native_policy_score"]
    axis.bar(x, y, color=["#4F7CAC", "#D17A3B", "#4D9B74"], width=0.62)
    axis.errorbar(
        x,
        y,
        yerr=np.vstack(
            [
                y - behavior["delta_native_policy_score_low"],
                behavior["delta_native_policy_score_high"] - y,
            ]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    axis.axhline(0, color="0.2", linewidth=1)
    axis.set_xticks(x, behavior["assay"])
    axis.set(
        ylabel="Change in task-aligned answer margin (log-odds)",
        title="No behavioral knowledge-report dissociation was induced",
        ylim=(-0.26, 0.03),
    )
    axis.text(
        0.98,
        0.05,
        "All 72 public/private baselines and interventions remained correct",
        transform=axis.transAxes,
        ha="right",
        fontsize=9,
        color="0.3",
    )
    fig.savefig(FIGURES / "behavioral_assays.png")
    plt.close(fig)

    observational = pd.read_csv(OBSERVATIONAL)
    observational = observational[observational["layer"].isin(sorted(COLORS))]
    observation_summary = (
        observational.groupby(["policy_style", "layer"], as_index=False)["effect"]
        .mean()
        .rename(columns={"effect": "observational_policy_contrast"})
    )
    causal_max = dose[dose["alpha_sd"].eq(16.0)][
        ["layer", "delta_policy_score"]
    ]
    readable_writable = observation_summary.merge(causal_max, on="layer")
    readable_writable.to_csv(SUMMARIES / "readable_writable_comparison.csv", index=False)
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.0), sharex=True, constrained_layout=True
    )
    sns.lineplot(
        data=readable_writable,
        x="layer",
        y="observational_policy_contrast",
        hue="policy_style",
        marker="o",
        linewidth=2.4,
        ax=axes[0],
    )
    axes[0].axhline(0, color="0.2", linewidth=1)
    axes[0].set(
        ylabel="J-lens policy contrast (logit units)",
        title="Readable: observational contrast",
    )
    axes[0].legend(title="Prompt wording", fontsize=9)
    axes[1].plot(
        causal_max["layer"],
        causal_max["delta_policy_score"],
        color="#D17A3B",
        marker="o",
        linewidth=2.4,
    )
    axes[1].axhline(0.5, color="#A33A3A", linestyle="--", label="causal target")
    axes[1].axhline(0, color="0.2", linewidth=1)
    axes[1].set(
        ylabel="16-SD causal effect (log-odds)",
        title="Not writable: negligible causal effect",
    )
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.set_xlabel("Model layer")
        axis.set_xticks(sorted(COLORS))
    fig.suptitle("Middle-layer policy signature: readable, not causally useful")
    fig.savefig(FIGURES / "readable_not_writable.png")
    plt.close(fig)

    print(f"wrote summaries to {SUMMARIES} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
