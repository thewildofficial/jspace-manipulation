import json

import pytest

from jspace_policy.sprint_runtime import (
    DEFAULT_GLOBAL_CEILING_USD,
    DEFAULT_LEDGER_PATH,
    GHA_GLOBAL_CEILING_USD,
    GHA_LEDGER_PATH,
    digest,
    global_ceiling_usd_for,
    reserve,
    resolve_ledger_path,
    verify_payload,
    write_new,
)


def test_failed_dispatch_cannot_refund_or_retry(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    reserve(ledger, "first", "preflight", 1.2)
    with pytest.raises(ValueError, match="already reserved"):
        reserve(ledger, "first", "preflight", 0.1)
    with pytest.raises(ValueError, match="budget exhausted"):
        reserve(ledger, "second", "preflight", 0.9)
    assert len(ledger.read_text().splitlines()) == 1


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_invalid_reservations(tmp_path, amount):
    with pytest.raises(ValueError):
        reserve(tmp_path / "ledger", "run", "baseline", amount)


def test_artifacts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "raw.json"
    write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"first": False})
    assert json.loads(path.read_text())["first"]


def test_locked_inference_refused_even_with_valid_hash():
    body = {
        "model_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "split": "locked",
        "status": "engineering_pilot",
        "queries": {"one": {}},
    }
    with pytest.raises(ValueError, match="does not authorize mechanistic"):
        verify_payload({**body, "sha256": digest(body)})


def test_locked_confirmation_requires_frozen_contrast():
    base = {
        "model_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "split": "locked",
        "status": "confirmation",
        "queries": {"one": {}},
    }
    with pytest.raises(ValueError, match="frozen contrast"):
        verify_payload({**base, "sha256": digest(base)})
    confirmed = {
        **base,
        "frozen_contrast": {"contrast_id": "auditor-verifiable-minus-unverifiable-v1"},
    }
    verify_payload({**confirmed, "sha256": digest(confirmed)})


def test_tampered_payload_refused():
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_payload({"sha256": "wrong"})


def test_resolve_ledger_path_defaults_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv("REPORT_REACTIVITY_LEDGER", raising=False)
    assert resolve_ledger_path() == DEFAULT_LEDGER_PATH
    custom = tmp_path / "custom.jsonl"
    monkeypatch.setenv("REPORT_REACTIVITY_LEDGER", str(custom))
    assert resolve_ledger_path() == custom


def test_global_ceiling_by_ledger_name_and_override(monkeypatch):
    monkeypatch.delenv("REPORT_REACTIVITY_GLOBAL_CEILING", raising=False)
    assert global_ceiling_usd_for(DEFAULT_LEDGER_PATH) == DEFAULT_GLOBAL_CEILING_USD
    assert global_ceiling_usd_for(GHA_LEDGER_PATH) == GHA_GLOBAL_CEILING_USD
    monkeypatch.setenv("REPORT_REACTIVITY_GLOBAL_CEILING", "12.5")
    assert global_ceiling_usd_for(DEFAULT_LEDGER_PATH) == 12.5


def test_gha_ledger_fresh_preflight_allowed_when_historical_stage_full(tmp_path):
    """C12 failure mode: historical preflight stage full; GHA ledger is independent."""

    historical = tmp_path / "reservations.jsonl"
    gha = tmp_path / "reservations_gha.jsonl"
    reserve(historical, "preflight-qwen38-v1", "preflight", 0.782856)
    reserve(historical, "preflight-qwen38-v2", "preflight", 0.782856)
    with pytest.raises(ValueError, match="budget exhausted"):
        reserve(historical, "gha-preflight-38-v1", "preflight", 0.782856)
    row = reserve(gha, "gha-preflight-38-v1", "preflight", 0.782856)
    assert row["total_reserved_usd"] == pytest.approx(0.782856)
    assert row["global_ceiling_usd"] == GHA_GLOBAL_CEILING_USD
    assert row["ledger_path"] == str(gha)


def test_gha_global_ceiling_binds_at_28(tmp_path):
    ledger = tmp_path / "reservations_gha.jsonl"
    reserve(ledger, "overhead-1", "overhead", 6.0)
    reserve(ledger, "incident-1", "incident", 8.0)
    reserve(ledger, "baseline-1", "baseline", 4.0)
    reserve(ledger, "mech-1", "mechanistic", 6.0)
    reserve(ledger, "repl-1", "replication", 3.9)
    # 6+8+4+6+3.9 = 27.9; another 0.2 preflight would exceed global 28
    with pytest.raises(ValueError, match="budget exhausted"):
        reserve(ledger, "preflight-overflow", "preflight", 0.2)
    reserve(ledger, "preflight-ok", "preflight", 0.05)
