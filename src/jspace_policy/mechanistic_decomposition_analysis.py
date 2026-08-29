"""Reproducible, non-causal summaries for V5-RBG-5B artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .mechanistic_decomposition import analyze_locked_patches


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_remote_artifact(path: str | Path, result_root: Path) -> Path:
    """Resolve a Modal-volume path against the downloaded offline artifact tree.

    Remote manifests intentionally retain their absolute container paths (for
    auditability).  GitHub/offline analysis downloads the corresponding
    ``/artifacts`` subtree under ``result_root/modal_artifacts``; this helper
    bridges those two representations without mutating the immutable manifest.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == "/" and parts[1] == "artifacts":
        relative = Path(*parts[2:])
        return result_root / "modal_artifacts" / relative
    return candidate


def _decode(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.uint16:
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.astype(np.float32, copy=False)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    numerator = float(np.sum(cross * cross))
    denominator = float(
        np.sqrt(np.sum((left.T @ left) ** 2) * np.sum((right.T @ right) ** 2))
    )
    return numerator / denominator if denominator else 0.0


def compute_activation_geometry(
    residual_path: Path,
    metadata: list[dict[str, Any]],
    anchor_names: list[str],
    layers: list[int],
    dataset_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    permutation_seed: int = 829054,
) -> dict[str, Any]:
    """Write matched distances and CKA without exposing a patch-selection signal."""
    residuals = np.load(residual_path, mmap_mode="r")
    feature_index = {row["condition_id"]: index for index, row in enumerate(metadata)}
    metadata_by_id = {row["condition_id"]: row for row in metadata}
    recipients = [
        row for row in dataset_rows
        if row["incentive"] == "opposed"
        and row["surface_kind"] == "assertion"
        and row["history"] == "redundant"
        and row["mapping_format"] == "prose"
        and row["condition_id"] in feature_index
        and not metadata_by_id[row["condition_id"]]["action_correct"]
    ]
    by_key = {
        (
            row["base_game_id"],
            row["frame"],
            row["incentive"],
            row["history"],
            row["surface_kind"],
            row["mapping_format"],
        ): row
        for row in dataset_rows
        if row["condition_id"] in feature_index
    }
    rng = np.random.default_rng(permutation_seed)
    distance_rows = []
    for recipient in recipients:
        key_prefix = (
            recipient["base_game_id"], recipient["frame"], recipient["incentive"], "redundant"
        )
        counterparts = {
            "table": by_key.get(key_prefix + ("assertion", "table")),
            "opaque": by_key.get(key_prefix + ("opaque_token", "prose")),
        }
        candidates = [
            row for row in dataset_rows
            if row["split"] == recipient["split"]
            and row["frame"] == recipient["frame"]
            and row["base_game_id"] != recipient["base_game_id"]
            and row["incentive"] == recipient["incentive"]
            and row["history"] == recipient["history"]
            and row["surface_kind"] == recipient["surface_kind"]
            and row["mapping_format"] == recipient["mapping_format"]
            and row["condition_id"] in feature_index
        ]
        if not candidates:
            continue
        permuted = candidates[int(rng.integers(len(candidates)))]
        for layer in layers:
            for anchor_index, anchor in enumerate(anchor_names):
                recipient_vector = _decode(
                    residuals[feature_index[recipient["condition_id"]], layer, anchor_index]
                )
                for family, donor in counterparts.items():
                    if donor is None:
                        continue
                    donor_vector = _decode(
                        residuals[feature_index[donor["condition_id"]], layer, anchor_index]
                    )
                    distance_rows.append({
                        "split": recipient["split"],
                        "base_game_id": recipient["base_game_id"],
                        "frame": recipient["frame"],
                        "layer": layer,
                        "anchor": anchor,
                        "comparison": family,
                        "cosine": _cosine(recipient_vector, donor_vector),
                        "normalized_l2": float(
                            np.linalg.norm(recipient_vector - donor_vector)
                            / max(np.linalg.norm(recipient_vector), 1e-8)
                        ),
                    })
                permuted_vector = _decode(
                    residuals[feature_index[permuted["condition_id"]], layer, anchor_index]
                )
                distance_rows.append({
                    "split": recipient["split"],
                    "base_game_id": recipient["base_game_id"],
                    "frame": recipient["frame"],
                    "layer": layer,
                    "anchor": anchor,
                    "comparison": "permuted_prose_control",
                    "cosine": _cosine(recipient_vector, permuted_vector),
                    "normalized_l2": float(
                        np.linalg.norm(recipient_vector - permuted_vector)
                        / max(np.linalg.norm(recipient_vector), 1e-8)
                    ),
                })

    condition_specs = {
        "prose_redundant": ("assertion", "redundant", "prose"),
        "table_redundant": ("assertion", "redundant", "table"),
        "opaque_redundant": ("opaque_token", "redundant", "prose"),
        "prose_none": ("assertion", "none", "prose"),
        "aligned_redundant": ("assertion", "redundant", "prose"),
    }
    # CKA uses matched base/frame rows; the aligned condition is kept separate by incentive.
    cka_rows = []
    for layer in layers:
        for anchor_index, anchor in enumerate(anchor_names):
            matrices: dict[str, np.ndarray] = {}
            for name, (surface, history, mapping) in condition_specs.items():
                selected = [
                    row for row in dataset_rows
                    if row["surface_kind"] == surface
                    and row["history"] == history
                    and row["mapping_format"] == mapping
                    and row["condition_id"] in feature_index
                    and (name.startswith("aligned") or row["incentive"] == "opposed")
                ]
                matrices[name] = (
                    np.stack(
                        [
                            _decode(
                                residuals[
                                    feature_index[row["condition_id"]],
                                    layer,
                                    anchor_index,
                                ]
                            )[:128]
                            for row in sorted(
                                selected,
                                key=lambda item: (item["base_game_id"], item["frame"]),
                            )
                        ]
                    )
                    if selected
                    else np.empty((0, 128), dtype=np.float32)
                )
            for left_name, right_name in (
                ("prose_redundant", "table_redundant"),
                ("prose_redundant", "opaque_redundant"),
                ("prose_redundant", "prose_none"),
            ):
                cka = _linear_cka(matrices[left_name], matrices[right_name])
                cka_rows.append({
                    "layer": layer,
                    "anchor": anchor,
                    "left": left_name,
                    "right": right_name,
                    "linear_cka": cka if np.isfinite(cka) else None,
                })
    payload = {
        "schema_version": 1,
        "n_distance_rows": len(distance_rows),
        "n_cka_rows": len(cka_rows),
        "distance_rows": distance_rows,
        "cka_rows": cka_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "n_distance_rows": len(distance_rows),
        "n_cka_rows": len(cka_rows),
    }


def analyze_study(
    config: dict[str, Any], dataset: dict[str, Any], result_root: Path
) -> dict[str, Any]:
    behavior = json.loads((result_root / "raw/behavior.json").read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "dataset_sha256": dataset["content_sha256"],
        "behavior": behavior["analysis"],
    }
    locked_path = result_root / "raw/locked_manifest.json"
    if locked_path.exists():
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        output["locked"] = locked
        patch_path = locked.get("artifact", {}).get("patch_results_path")
        local_patch_path = (
            _resolve_remote_artifact(patch_path, result_root) if patch_path else None
        )
        if local_patch_path and local_patch_path.exists():
            with gzip.open(local_patch_path, "rt", encoding="utf-8") as handle:
                output["locked_patch_analysis"] = analyze_locked_patches(
                    json.load(handle), config
                )
    jspace_path = result_root / "raw/jspace_manifest.json"
    if jspace_path.exists():
        output["jspace"] = json.loads(jspace_path.read_text(encoding="utf-8"))
    output["claim_ladder"] = {
        "behavior": "behavioral conjunction only",
        "probes": "information availability/decodability only",
        "jspace": "observational evidence only",
        "locked_patch": "selective natural residual transport under this frozen contrast"
        if "locked" in output
        else "not established",
        "excluded": [
            "complete circuit",
            "deception",
            "consciousness",
            "model-family generality",
        ],
    }
    output["content_sha256"] = canonical_sha256(output)
    return output
