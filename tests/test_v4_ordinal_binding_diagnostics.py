from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXTRACT = _module(
    "extract_v4_embedded_ordinal_report",
    "scripts/extract_v4_embedded_ordinal_report.py",
)
DIAGNOSTICS = _module(
    "analyze_v4_ordinal_binding_diagnostics",
    "scripts/analyze_v4_ordinal_binding_diagnostics.py",
)


def test_embedded_report_extraction_is_lossless_and_marks_deviation() -> None:
    payload = {
        "metadata": {"run_id": "behavior-run"},
        "summary": {"gate_pass": True, "self_report": {"n_rows": 2}},
        "self_report_rows": [{"report_id": "left"}, {"report_id": "right"}],
    }
    result = EXTRACT.extract(payload, "abc123")
    assert result["rows"] is payload["self_report_rows"]
    assert result["summary"] == {"n_rows": 2}
    assert result["metadata"]["behavior_run_id"] == "behavior-run"
    assert result["metadata"]["source_behavior_payload_sha256"] == "abc123"
    assert "same Modal invocation" in result["metadata"]["protocol_deviation"]


def test_specificity_is_zero_when_both_false_responses_shift_equally() -> None:
    common = {
        "condition_id": "condition",
        "decision_correct": True,
        "base_game_id": "game",
        "selected_action": "B",
        "correct_surface": "R1",
        "options": {"X": "R1", "Y": "R2", "Z": "R3"},
        "expected_label": "X",
        "legal_choice": "X",
        "legal_choice_correct": True,
        "frame": "strategic",
        "ordinal_binding": {
            "action_presentation_order": [0, 1, 2],
            "response_presentation_order": [0, 1, 2],
        },
    }
    retrospective = {
        **common,
        "access_condition": "retrospective",
        "legal_action_logits": {"X": 3.0, "Y": 2.0, "Z": 2.0},
    }
    answer_only = {
        **common,
        "access_condition": "answer_only",
        "legal_action_logits": {"X": 3.0, "Y": 1.0, "Z": 1.0},
    }
    payload = {"metadata": {}, "rows": [retrospective, answer_only]}
    indexed, conditions = DIAGNOSTICS.paired_rows(payload)
    result = DIAGNOSTICS.specificity(indexed, conditions, "label")
    assert result["cluster_mean"] == 0.0
    assert result["two_sided_exact_cluster_sign_flip_p"] == 1.0
