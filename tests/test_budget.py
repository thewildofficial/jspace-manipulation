from pathlib import Path

import pytest

from jspace_policy.budget import admit_run, append_ledger, estimate_cost


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
