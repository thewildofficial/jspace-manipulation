import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_v4_ordinal_binding.py"
SPEC = importlib.util.spec_from_file_location("analyze_v4_ordinal_binding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
exact_cluster_sign_flip = ANALYSIS.exact_cluster_sign_flip
holm = ANALYSIS.holm


def test_exact_cluster_sign_flip_uses_whole_clusters() -> None:
    assert exact_cluster_sign_flip([1.0, 1.0]) == 0.5
    assert exact_cluster_sign_flip([0.0, 0.0]) == 1.0


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm({"label": 0.01, "position": 0.04})
    assert adjusted == {"label": 0.02, "position": 0.04}
