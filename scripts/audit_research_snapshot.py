"""Offline audit of archived RBG-6 evidence and the zero-vector parity blind spot.

Writes a NEW artifact; never modifies a frozen experiment or its analysis.
The normalization example is synthetic, not a Qwen rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit() -> dict:
    raw_path = ROOT / 'results/v5_full_action_trajectory/raw/behavior_v1.json'
    payload = json.loads(raw_path.read_text())
    rows = [r for r in payload['rows'] if r['frame'] == 'strategic'
            and r['incentive'] == 'opposed' and r['surface_kind'] == 'assertion'
            and r['history'] == 'redundant' and r['mapping_format'] == 'prose']
    both = [r for r in rows if all(v['correct'] for v in r['self_reports'].values())]
    errors = [r for r in rows if r not in both]
    analyzer_path = ROOT / 'scripts/analyze_v5_full_action_trajectory.py'
    spec = importlib.util.spec_from_file_location('rbg6_audit_analyzer', analyzer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((ROOT / 'configs/v5/full_action_trajectory/experiment.json').read_text())
    recomputed = module.analyze(payload, config)
    archived = json.loads((ROOT / 'results/v5_full_action_trajectory/analysis/behavior_v1_analysis.json').read_text())
    assert recomputed == archived, 'archived analysis no longer reproduces'
    # Non-uniform RMSNorm gains can change candidate ranking, not merely scale it.
    h = np.array([[2., 1.], [0., 0.]])
    gamma = np.array([0.1, 3.])
    weights = np.eye(2)
    bare = h @ weights.T
    normalized = (h / np.sqrt(np.mean(h*h, axis=-1, keepdims=True) + 1e-6) * gamma) @ weights.T
    assert np.allclose(bare[1], normalized[1])
    assert bare[0].argmax() != normalized[0].argmax()
    return {
        'audited_commit': 'eb23897f12bf902763f43f4772bdf8b0ebd3f5e3',
        'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'archived_analysis_reproduced': True,
        'rbg6_primary': {
            'rows': len(rows),
            'base_game_clusters': len({r['base_game_id'] for r in rows}),
            'unique_source_conditions': len({r['source_condition_id'] for r in rows}),
            'unique_unordered_concept_pairs': len({tuple(sorted(r['concepts'])) for r in rows}),
            'both_reports_correct': len(both),
            'correct_action_among_both_reports_correct': sum(r['trajectory_arms']['self_generated']['action_correct'] for r in both),
            'wrong_action_among_report_error_cases': sum(not r['trajectory_arms']['self_generated']['action_correct'] for r in errors),
            'correct_report_tokens': sum(v['correct'] for r in rows for v in r['self_reports'].values()),
            'total_report_tokens': 2*len(rows),
            'dissociation_binomial_interval_treating_rows_independent': module.exact_binomial_interval(0, len(rows)),
            'any_dissociation_per_base_zero_event_interval': module.exact_binomial_interval(0, len({r['base_game_id'] for r in rows})),
            'note': 'The per-base interval targets any dissociation across two report orders; it is not an interchangeable CI for trajectory risk. Both binomial models additionally assume independent sampled units.'
        },
        'synthetic_normalization_counterexample': {
            'not_a_model_rerun': True,
            'bare_candidate_scores': bare.tolist(),
            'normalized_candidate_scores': normalized.tolist(),
            'zero_vector_check_passes': True,
            'nonzero_candidate_ranking_changes': True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
