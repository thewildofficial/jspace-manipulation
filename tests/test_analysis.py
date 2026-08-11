import pandas as pd

from jspace_policy.analysis import (
    add_behavior_metrics,
    paired_policy_shift,
    paired_readout_trajectories,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "family": "f",
                "split": "discovery",
                "policy_style": "explicit",
                "world_state": "A",
                "policy": "reveal",
                "expected_report": "A",
                "logp_A": -0.1,
                "logp_B": -2.0,
            },
            {
                "scenario_id": "s1",
                "family": "f",
                "split": "discovery",
                "policy_style": "explicit",
                "world_state": "A",
                "policy": "conceal",
                "expected_report": "B",
                "logp_A": -2.0,
                "logp_B": -0.1,
            },
            {
                "scenario_id": "s1",
                "family": "f",
                "split": "discovery",
                "policy_style": "explicit",
                "world_state": "B",
                "policy": "reveal",
                "expected_report": "B",
                "logp_A": -2.0,
                "logp_B": -0.1,
            },
            {
                "scenario_id": "s1",
                "family": "f",
                "split": "discovery",
                "policy_style": "explicit",
                "world_state": "B",
                "policy": "conceal",
                "expected_report": "A",
                "logp_A": -0.1,
                "logp_B": -2.0,
            },
        ]
    )


def test_behavior_metrics_encode_policy_composition() -> None:
    result = add_behavior_metrics(_frame())
    assert result["correct"].all()
    assert (result["policy_following_score"] > 0).all()


def test_paired_conceal_shift_is_negative_in_both_worlds() -> None:
    shifts = paired_policy_shift(_frame())
    assert (shifts["conceal_minus_reveal_truth_score"] < 0).all()


def test_readout_trajectories_pair_policy_and_fact() -> None:
    rows = []
    for world, fact_coordinate in (("A", 2.0), ("B", -2.0)):
        for policy, policy_coordinate in (("reveal", -3.0), ("conceal", 3.0)):
            rows.append(
                {
                    "scenario_id": "s1",
                    "policy_style": "indirect",
                    "world_state": world,
                    "policy": policy,
                    "layer": 10,
                    "layer_fraction": 0.5,
                    "policy_coordinate": policy_coordinate,
                    "fact_coordinate": fact_coordinate,
                }
            )
    policy, fact = paired_readout_trajectories(pd.DataFrame(rows))
    assert policy.loc[0, "effect"] == 6.0
    assert fact.loc[0, "effect"] == 4.0
