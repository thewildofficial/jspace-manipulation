from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

REQUIRED_BEHAVIOR_COLUMNS = {
    "scenario_id",
    "family",
    "split",
    "policy_style",
    "world_state",
    "policy",
    "expected_report",
    "logp_A",
    "logp_B",
}


def add_behavior_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_BEHAVIOR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"behavior table missing columns: {sorted(missing)}")
    out = frame.copy()
    out["literal_logodds_A_over_B"] = out["logp_A"] - out["logp_B"]
    out["predicted_report"] = np.where(out["literal_logodds_A_over_B"] >= 0, "A", "B")
    out["correct"] = out["predicted_report"] == out["expected_report"]
    world_sign = np.where(out["world_state"] == "A", 1.0, -1.0)
    out["truth_aligned_score"] = world_sign * out["literal_logodds_A_over_B"]
    policy_sign = np.where(out["policy"] == "reveal", 1.0, -1.0)
    out["policy_following_score"] = policy_sign * out["truth_aligned_score"]
    return out


def _cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster: str = "scenario_id",
    n_boot: int = 1000,
    seed: int = 1729,
) -> tuple[float, float, float]:
    point = float(statistic(frame))
    clusters = frame[cluster].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return point, float("nan"), float("nan")
    grouped = {key: value for key, value in frame.groupby(cluster, sort=False)}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[key] for key in chosen], ignore_index=True)
        samples[i] = statistic(boot)
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def _cluster_mean_bootstrap(
    frame: pd.DataFrame,
    column: str,
    *,
    cluster: str = "scenario_id",
    n_boot: int = 1000,
    seed: int = 1729,
) -> tuple[float, float, float]:
    """Vectorized cluster bootstrap for a mean, including unequal cluster sizes."""
    values = frame.assign(_value=frame[column].astype(float)).groupby(cluster)["_value"]
    sums = values.sum().to_numpy()
    counts = values.count().to_numpy()
    point = float(sums.sum() / counts.sum())
    if len(sums) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    samples = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def cluster_mean_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    cluster: str = "scenario_id",
    n_boot: int = 5000,
    seed: int = 1729,
) -> tuple[float, float, float]:
    """Public scenario-clustered bootstrap interval for a column mean."""
    if column not in frame:
        raise ValueError(f"table missing outcome column: {column}")
    if cluster not in frame:
        raise ValueError(f"table missing cluster column: {cluster}")
    return _cluster_mean_bootstrap(
        frame, column, cluster=cluster, n_boot=n_boot, seed=seed
    )


def grouped_intervention_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    outcome_columns: list[str],
    *,
    n_boot: int = 5000,
    seed: int = 1729,
) -> pd.DataFrame:
    """Summarize intervention outcomes with scenario-clustered intervals."""
    required = set(group_columns) | set(outcome_columns) | {"scenario_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"intervention table missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    grouped = frame.groupby(group_columns, sort=True, dropna=False)
    for group_index, (keys, group) in enumerate(grouped):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys, strict=True))
        record.update(
            n_rows=len(group),
            n_scenarios=group["scenario_id"].nunique(),
        )
        for outcome_index, column in enumerate(outcome_columns):
            point, low, high = cluster_mean_interval(
                group,
                column,
                n_boot=n_boot,
                seed=seed + group_index * 101 + outcome_index,
            )
            record[column] = point
            record[f"{column}_low"] = low
            record[f"{column}_high"] = high
        records.append(record)
    return pd.DataFrame.from_records(records)


def behavior_summary(
    frame: pd.DataFrame, *, n_boot: int = 1000, seed: int = 1729
) -> pd.DataFrame:
    data = add_behavior_metrics(frame)
    records: list[dict[str, object]] = []
    group_cols = ["split", "policy_style", "world_state", "policy"]
    for keys, group in data.groupby(group_cols, sort=True):
        accuracy = _cluster_mean_bootstrap(group, "correct", n_boot=n_boot, seed=seed)
        margin = _cluster_mean_bootstrap(
            group, "policy_following_score", n_boot=n_boot, seed=seed + 1
        )
        record = dict(zip(group_cols, keys, strict=True))
        record.update(
            n_rows=len(group),
            n_scenarios=group["scenario_id"].nunique(),
            accuracy=accuracy[0],
            accuracy_low=accuracy[1],
            accuracy_high=accuracy[2],
            margin=margin[0],
            margin_low=margin[1],
            margin_high=margin[2],
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def paired_policy_shift(frame: pd.DataFrame) -> pd.DataFrame:
    data = add_behavior_metrics(frame)
    index = ["scenario_id", "family", "split", "policy_style", "world_state"]
    wide = data.pivot(index=index, columns="policy", values="truth_aligned_score")
    if not {"reveal", "conceal"}.issubset(wide.columns):
        raise ValueError("both reveal and conceal rows are required for paired shifts")
    out = wide.reset_index()
    out["conceal_minus_reveal_truth_score"] = out["conceal"] - out["reveal"]
    return out


def paired_readout_trajectories(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scenario-level policy and fact contrasts from a balanced lens readout."""
    required = {
        "scenario_id",
        "policy_style",
        "world_state",
        "policy",
        "layer",
        "layer_fraction",
        "policy_coordinate",
        "fact_coordinate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"lens readout table missing columns: {sorted(missing)}")

    policy_index = [
        "scenario_id",
        "policy_style",
        "world_state",
        "layer",
        "layer_fraction",
    ]
    policy_wide = frame.pivot(index=policy_index, columns="policy", values="policy_coordinate")
    if not {"reveal", "conceal"}.issubset(policy_wide.columns):
        raise ValueError("both reveal and conceal rows are required for policy trajectories")
    policy = policy_wide.reset_index()
    policy["effect"] = policy["conceal"] - policy["reveal"]
    policy = (
        policy.groupby(
            ["scenario_id", "policy_style", "layer", "layer_fraction"], as_index=False
        )["effect"]
        .mean()
        .assign(contrast="conceal prompt − reveal prompt")
    )

    fact_index = [
        "scenario_id",
        "policy_style",
        "policy",
        "layer",
        "layer_fraction",
    ]
    fact_wide = frame.pivot(index=fact_index, columns="world_state", values="fact_coordinate")
    if not {"A", "B"}.issubset(fact_wide.columns):
        raise ValueError("both A and B world states are required for fact trajectories")
    fact = fact_wide.reset_index()
    fact["effect"] = fact["A"] - fact["B"]
    fact = (
        fact.groupby(
            ["scenario_id", "policy_style", "layer", "layer_fraction"], as_index=False
        )["effect"]
        .mean()
        .assign(contrast="A prompt − B prompt")
    )
    return policy, fact
