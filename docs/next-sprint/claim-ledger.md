# Claim ledger: report-reactivity sprint (Qwen3.8-27B + Qwen3.6-27B)

Status: **living record**. Each row names the experimental unit, the raw
artifact, the analysis command, and whether the claim is discovery (pilot,
descriptive) or confirmation (locked, frozen contrast). Limits are part of
every claim. No row is a paper conclusion on its own.

## How to reproduce the CPU side

```bash
uv sync --extra dev --extra modal
uv run pytest                       # 160 passed at last full run
uv run python experiments/report_reactivity/prepare.py --task report \
  --model Qwen/Qwen3.8-27B --bases 16 --output /tmp/check.json
```

GPU rows were produced by `modal_report_reactivity.py` (manually dispatched,
no retries, ceiling reserved before launch, reservation retained on failure).
Spend ledger: `results/report_reactivity/reservations.jsonl` against the $30
global ceiling in `src/jspace_policy/sprint_runtime.py::STAGE_LIMITS`.

## Claims

| # | Claim | Unit | Raw artifact | Analysis | Standing | Limit |
|---|---|---|---|---|---|---|
| C1 | Batch-8 scoring drifts vs singleton on long prompts (0.5 > 0.25 gate); batch-4 passes (0.125) | 960-query report pilot payload, Qwen3.8 | `results/report_reactivity/report16-38-v1/raw.json` (gate failed, scores withheld), `report16-38-v2/raw.json` (960 scores) | parity block in `modal_report_reactivity.score_gpu` | Discovery, instrument | One payload, one checkpoint; padding/SSM mechanism is hypothesis, not established |
| C2 | Fresh nonce-word report games sit at ceiling: direct/self/oracle/control/external 100%, primary self-minus-control +0.00pp over 16 bases | 768 report records, Qwen3.8 non-thinking | `artifacts/processed/sprint_report16_38.json`, `report16-38-v2/raw.json` | `sprint_analysis.primary_gate` logic, base-equal weighting | Discovery pilot | Ceiling means no rescue inference either way; stop rule applied to mechanistic rescue |
| C3 | Inverted reports are followed 82.8% (106/128): report content controls action | Same as C2, swapped arm | Same as C2 | arm accuracy vs `expected_action` | Discovery pilot | Transcript dependence, not spontaneous self-trust; instruction explicitly privileges reports |
| C4 | Pressure induces violations (48/96 conflict, 0/96 no-conflict), uniform across tool-name families | 192 incident records, Qwen3.8 non-thinking | `artifacts/processed/sprint_incident16_38.json`, `results/report_reactivity/incident16-38-v1/raw.json` | choice mapped via `expected_paths`, simulator ground truth | Discovery pilot | Prompted game incentives; not autonomous goal pursuit |
| C5 | Disclosure is oversight-invariant: verifiable vs unverifiable auditor answers byte-identical; violators confess 48/48 | Same as C4, auditor branches of taken actions | Same as C4 | taken-action branch only, vs immutable event log | Discovery pilot; locked confirmation pending | Null with 48 violations of support; exact-zero difference is descriptive |
| C6 | Primary gate needs ~30pp true gain for 80% power at n=64-96; power at the 15pp margin is ~3% by construction | CPU simulation, no model | `sprint_analysis.simulate_primary_power` | paired-Bernoulli planning sim | Methods note, not empirical | Normal approximation; realized test is the exact sign-flip in `grouped_paired_analysis` |
| C7 | Tokenizer template hashes differ between Qwen3.6 and Qwen3.8; per-checkpoint pinning required | CPU tokenizer render | `artifacts/processed/sprint_preflight_36.json`, `sprint_preflight_38.json` | `sprint_runtime.prepare_query` exact single-token check | Engineering fact | Template content, not a behavioral claim |

## Frozen confirmation contrast (locked): CONFIRMED

Primary locked estimand: auditor-report truthfulness difference,
verifiable-minus-unverifiable, averaged per base episode over conflict
episodes with violations. Discovery value: exactly 0.00.

Locked result (`locked32-38-v1`, 32 fresh held-out bases, Qwen3.8,
replication stage, batch-4 parity 0.125 passed): conflict violations
96/192 (32/64 per alias family, pressure-driven as in discovery),
no-conflict 0/192, violators confess 96/96 under both auditor conditions,
and the verifiable-minus-unverifiable difference is exactly 0.00 again.
The frozen null replicates on unseen lexical material.

## Matched-checkpoint rows (Qwen3.6-27B, non-thinking, batch-4 parity passed)

| # | Claim | Unit | Raw artifact | Analysis | Standing | Limit |
|---|---|---|---|---|---|---|
| C8 | 3.6 report pilot: direct 96.1% (5 errors), self-report rescues all 5 to 100%; primary self-minus-control +0.00 (control at ceiling); swapped-following 65.6% vs 82.8% on 3.8 | 768 report records, Qwen3.6 | `artifacts/processed/sprint_report16_36.json`, `results/report_reactivity/report16-36-v1/raw.json` | arm accuracy vs `expected_action`, base-equal weighting | Discovery pilot | 5-case rescue is descriptive; parity passed exactly on the 0.25 line, so 3.6 batch numerics are touchier than 3.8 (0.125) |
| C9 | 3.6 incident pilot: 0/96 conflict violations vs 48/96 on 3.8; disclosure honest throughout | 192 incident records, Qwen3.6 | `artifacts/processed/sprint_incident16_36.json`, `results/report_reactivity/incident16-36-v1/raw.json` | choice via `expected_paths`, taken-action auditor branches | Discovery pilot | Violation propensity is a checkpoint property; with no violations there is no concealment test on 3.6 |

## Spend (reservations retained, fail-closed)

$7.26 of $30 reserved across 8 runs (1 overhead + 2 preflight + 2 report
baseline + 2 incident + 1 locked replication). Failed runs
(`report16-38-v1` parity, first `locked32-38-v1` dispatch bug) retain their
ceilings; provider reconciliation is recorded separately and never refunds
the ledger.
