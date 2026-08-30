from pathlib import Path

import pytest

from jspace_policy.budget import (
    RBG5B_EXECUTION_LIMITS,
    RBG5B_INCREMENTAL_COST_LIMIT_USD,
    ExecutionLimit,
    admit_execution_plan,
    admit_run,
    append_ledger,
    estimate_cost,
    execution_plan_cost_usd,
)


def test_estimate_includes_buffer() -> None:
    estimate = estimate_cost("A10", 3600)
    assert estimate.gpu_usd == pytest.approx(1.1016)
    assert estimate.buffered_usd > estimate.subtotal_usd > estimate.gpu_usd


def test_admission_refuses_projected_overspend(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    expensive = estimate_cost("H100", 3600 * 5)
    append_ledger(ledger, expensive, run_id="x", stage="test")
    with pytest.raises(RuntimeError, match="run refused"):
        admit_run(ledger, estimate_cost("A10", 3600 * 10), study_limit_usd=25)


def test_rbg5b_hard_timeouts_fit_incremental_authorization() -> None:
    ceiling = admit_execution_plan(
        RBG5B_EXECUTION_LIMITS,
        limit_usd=RBG5B_INCREMENTAL_COST_LIMIT_USD,
    )

    assert ceiling == execution_plan_cost_usd(RBG5B_EXECUTION_LIMITS)
    assert ceiling < 8.81
    assert RBG5B_EXECUTION_LIMITS["behavior"].timeout_seconds == 1200
    assert RBG5B_EXECUTION_LIMITS["discovery"].timeout_seconds == 3000
    assert RBG5B_EXECUTION_LIMITS["locked"].timeout_seconds == 3600
    assert RBG5B_EXECUTION_LIMITS["jspace"].timeout_seconds == 900


def test_incremental_authorization_refuses_oversized_timeout_plan() -> None:
    oversized = {
        "run": ExecutionLimit(
            timeout_seconds=20_000,
            gpu="A100-80GB",
            cpu_cores=16,
            memory_gib=64,
        )
    }

    with pytest.raises(RuntimeError, match="hard timeout ceiling"):
        admit_execution_plan(oversized, limit_usd=10.0)
