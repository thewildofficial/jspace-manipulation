from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from jspace_policy.jspace_interventions import canonical_sha256, select_first_feasible_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("configs/v3/jspace_interventions/fresh_candidates.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/v3_jspace_interventions/raw/fresh_baseline_candidates_v3.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/v3/jspace_interventions/fresh_frozen.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite frozen corpus: {args.output}")
    candidates = json.loads(args.candidates.read_text())
    baseline = json.loads(args.baseline.read_text())
    if baseline.get("intervention_opened"):
        raise RuntimeError("baseline artifact says an intervention was opened")
    if baseline["candidate_sha256"] != canonical_sha256(candidates):
        raise RuntimeError("candidate config hash does not match baseline artifact")
    frozen = select_first_feasible_corpus(candidates, baseline["rows"])
    frozen["created_at"] = datetime.now(UTC).isoformat()
    frozen["baseline_artifact_sha256"] = canonical_sha256(baseline)
    without_hash = {key: value for key, value in frozen.items() if key != "content_sha256"}
    frozen["content_sha256"] = canonical_sha256(without_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_sha256": frozen["content_sha256"],
                "categories": {
                    category["name"]: {
                        "arguments": category["arguments"],
                        "functions": [function["name"] for function in category["functions"]],
                    }
                    for category in frozen["categories"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
