from jspace_policy.dataset import FAMILIES, generate_dataset


def test_factorial_balance_and_grouping() -> None:
    rows = generate_dataset(n_base=12, seed=7, stage="composition")
    assert len(rows) == 12 * 2 * 2 * 2
    assert {row.family for row in rows} == set(FAMILIES)
    for scenario_id in {row.scenario_id for row in rows}:
        group = [row for row in rows if row.scenario_id == scenario_id]
        assert len(group) == 8
        cells = {
            (row.world_state, row.policy, row.policy_style, row.expected_report)
            for row in group
        }
        assert ("A", "reveal", "explicit", "A") in cells
        assert ("A", "conceal", "indirect", "B") in cells
        assert ("B", "conceal", "explicit", "A") in cells


def test_generation_is_deterministic() -> None:
    assert generate_dataset(6, 99) == generate_dataset(6, 99)


def test_toy_stage_omits_story_reasoning() -> None:
    rows = generate_dataset(6, 99, stage="toy")
    assert {row.family for row in rows} == {"toy"}
    assert all("INPUT FACT:" in row.prompt for row in rows)
    assert all("container" not in row.prompt.lower() for row in rows)


def test_indirect_prompt_avoids_policy_lexicon() -> None:
    forbidden = {"truth", "reveal", "hide", "conceal", "lie", "deceive"}
    rows = [row for row in generate_dataset(6) if row.policy_style == "indirect"]
    for row in rows:
        words = set(row.prompt.lower().replace(".", "").split())
        assert forbidden.isdisjoint(words)
