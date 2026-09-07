# Methods note: GHA report-reactivity 16-base discovery baseline (`gha-report16-38-v1`)

Status: **discovery pilot / `engineering_pilot`**. Companion ledger rows: **C14**
(ceiling replication + swapped steering on the GHA path), **C15** (prose vs
opaque swapped split). Not locked confirmation. No novelty or priority claim.

## Run identity

| Field | Value |
|---|---|
| Actions run | [34048123330](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34048123330) |
| `run_id` | `gha-report16-38-v1` |
| Stage | `baseline` |
| Task | `report` |
| Split | `discovery` |
| Bases | 16 |
| Batch size | 4 |
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Payload sha256 | `126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d` (matches CPU dry-run prepare `gha-report16-38-dry-v1`) |
| Parity | passed — `replay_max_abs=0.0`, `batch_single_max_abs=0.125`, `choices_agree=true` |
| Status | `engineering_pilot` |
| Timing | `elapsed_seconds≈297.3`, `load_seconds≈76.3` |
| GPU (artifact) | `NVIDIA A100-SXM4-80GB` |
| Scores / records | 960 scores; 768 records (128 per arm × 6 arms) |

## Immutable artifacts

Committed under `results/report_reactivity/gha-report16-38-v1/`:

- `raw.json` — Modal `score_gpu` JSON return (parity + scores)
- `prepared.json.gz` — exact CPU prepare payload scored by this run (sha256 matches
  `payload_sha256`; recovered from dry-run Actions artifact
  `gha-report16-38-dry-v1` / run 34048067080, not from the scored-run upload
  which originally omitted prepare)
- `input_manifest.json` — reserve metadata (`ledger_path=reservations_gha.jsonl`,
  `global_ceiling_usd=28`, ceiling ≈ `$0.7829`)
- `analysis/arm_accuracy_summary.json` — CPU join of prepared records + scores

GHA spend ledger updated from the Actions artifact only:

- `results/report_reactivity/reservations_gha.jsonl` now includes
  `gha-report16-38-v1` (total reserved ≈ `$1.5657` with prior
  `gha-preflight-38-v2` retention)
- Historical `results/report_reactivity/reservations.jsonl` **unchanged**

## Scoring and analysis

1. CPU prepare (Actions, then locally reproducible):

   ```bash
   uv run --with 'transformers>=5.5' --with 'huggingface-hub>=0.34' \
     --with 'httpx[socks]>=0.28' --with 'jinja2>=3.1' \
     python experiments/report_reactivity/prepare.py --task report \
     --model Qwen/Qwen3.8-27B --bases 16 \
     --output artifacts/prepared/gha-report16-38-v1.json
   # Expect sha256 126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d
   ```

2. GPU scoring was the Actions `dry_run=false` dispatch of
   `modal_report_reactivity.py` with `batch_size=4` (no new GPU from this
   documentation PR).

3. CPU analysis:

   ```bash
   uv run python experiments/report_reactivity/analyze_gha_report16.py \
     --prepared results/report_reactivity/gha-report16-38-v1/prepared.json.gz \
     --raw results/report_reactivity/gha-report16-38-v1/raw.json \
     --output results/report_reactivity/gha-report16-38-v1/analysis/arm_accuracy_summary.json
   ```

Self-report uses chained lookup `report1 → report2[r1] → actions[r1r2]`.
Other arms use the single action query. Correctness is choice vs
`expected_action`. Bases are weighted equally (`sprint_analysis`).

## Base-equal action accuracy (16 bases)

| Arm | Accuracy |
|---|---|
| direct | 1.0 |
| matched_control | 1.0 |
| self_report | 1.0 (both reports correct on 128/128) |
| oracle | 1.0 |
| external_facts | 1.0 |
| swapped | 0.828125 overall (prose 0.71875; opaque 0.9375) |

## Primary contrasts

- `self_report − matched_control = 0.0`
- `self_report − direct = 0.0`

Primary gate does not pass (ceiling / zero gain); status remains descriptive
pilot. Selection must not treat this as confirmatory.

## What is NOT claimed

- Not locked confirmation; not a paper conclusion
- Not a novelty or priority claim over prior Modal C2/C3
- Ceiling on Direct / self / control → rewrite-vs-reveal rescue is
  **unidentifiable** here (stop rule for mechanistic rescue /
  reports-fix-failures)
- Swapped accuracy below ceiling shows report content can steer action
  (stronger drop on prose than opaque); instruction privileges reports —
  transcript dependence, not spontaneous self-trust
- Qualitative replication of C2/C3 on a fresh GHA path + nonce corpus only;
  do not overclaim statistical independence beyond that
- No new GPU spend from documenting this result
