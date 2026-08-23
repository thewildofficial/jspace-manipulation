from __future__ import annotations

import json
from pathlib import Path

from jspace_policy.strategic_epistemic_search import (
    dataset_payload,
    verify_dataset_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v4/strategic_epistemic_search/ordinal_binding_permutation.json"
OUTPUT = ROOT / "configs/v4/strategic_epistemic_search/ordinal_binding_dataset.json"


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite prospective artifact: {OUTPUT}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload = dataset_payload(config)
    verify_dataset_payload(payload, config)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(ROOT)),
                "rows": len(payload["rows"]),
                "content_sha256": payload["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
