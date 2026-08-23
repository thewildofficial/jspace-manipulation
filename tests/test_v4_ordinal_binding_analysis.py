from scripts.analyze_v4_ordinal_binding import exact_cluster_sign_flip, holm


def test_exact_cluster_sign_flip_uses_whole_clusters() -> None:
    assert exact_cluster_sign_flip([1.0, 1.0]) == 0.5
    assert exact_cluster_sign_flip([0.0, 0.0]) == 1.0


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm({"label": 0.01, "position": 0.04})
    assert adjusted == {"label": 0.02, "position": 0.04}
