"""CPU-only exact tokenizer preflight, no scientific endpoint inference."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from jspace_policy.sprint_runtime import MODEL_REVISIONS, digest, prepare_query, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_REVISIONS, default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=MODEL_REVISIONS[args.model],
        cache_dir="artifacts/cache/huggingface",
    )
    messages = [
        [{"role": "user", "content": "Return only A or B. The required answer is A."}],
        [
            {
                "role": "user",
                "content": "An instrument check: return only A or B. The required answer is B.",
            }
        ],
    ]
    payload = {
        "model_id": args.model,
        "revision": MODEL_REVISIONS[args.model],
        "split": "discovery",
        "status": "engineering_pilot",
        "task": "preflight",
        "thinking": False,
        "preserve_thinking": False,
        "tokenizer_template_sha256": digest(tokenizer.chat_template),
        "pad_token_id": tokenizer.pad_token_id,
        "queries": {str(i): prepare_query(tokenizer, m) for i, m in enumerate(messages)},
    }
    payload["sha256"] = digest(payload)
    write_new(args.output, payload)
    print(
        {
            "sha256": payload["sha256"],
            "lengths": [q["length"] for q in payload["queries"].values()],
        }
    )


if __name__ == "__main__":
    main()
