import json

import pytest

from jspace_policy.sprint_runtime import digest, reserve, verify_payload, write_new


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
