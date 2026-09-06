# Claim ledger: report-reactivity sprint (Qwen3.8-27B + Qwen3.6-27B)

Status: **living record**. Each row names the experimental unit, the raw
artifact, the analysis command, and whether the claim is discovery (pilot,
descriptive) or confirmation (locked, frozen contrast). Limits are part of
every claim. No row is a paper conclusion on its own.

## How to reproduce the CPU side

```bash
uv sync --extra dev --extra modal
uv run pytest                       # 160+ passed at last full run
uv run python experiments/report_reactivity/prepare.py --task report \
  --model Qwen/Qwen3.8-27B --bases 16 --output /tmp/check.json
```

Actions: `.github/workflows/report-reactivity.yml` (`workflow_dispatch` only).
Default `dry_run=true` runs pytest + CPU prepare and uploads the prepared
payload without Modal. Set `dry_run=false` only when you intend to spend GPU;
`batch_size` defaults to 4 (C1). Workflow sets
`REPORT_REACTIVITY_LEDGER=results/report_reactivity/reservations_gha.jsonl`
(GHA-era ledger, global ceiling $28; see RR-D001). Local/manual runs default
to historical `results/report_reactivity/reservations.jsonl` ($30).

GPU rows were produced by `modal_report_reactivity.py` (manually dispatched,
no retries, ceiling reserved before launch, reservation retained on failure).
`score_gpu` returns a JSON string (`dumps_jsonable`) so the Actions client
never unpickles torch (C13). Spend ledgers: historical
`results/report_reactivity/reservations.jsonl` (~$7.26 / $30, immutable) and
GHA `results/report_reactivity/reservations_gha.jsonl` (~$1.57 / $28 after
retained C13 preflight + C14 baseline). Stage caps remain
`src/jspace_policy/sprint_runtime.py::STAGE_LIMITS`.

## Claims

| # | Claim | Unit | Raw artifact | Analysis | Standing | Limit |
|---|---|---|---|---|---|---|
| C1 | Batch-8 scoring drifts vs singleton on long prompts (0.5 > 0.25 gate); batch-4 passes (0.125) | 960-query report pilot payload, Qwen3.8 | `results/report_reactivity/report16-38-v1/raw.json` (gate failed, scores withheld), `report16-38-v2/raw.json` (960 scores) | parity block in `modal_report_reactivity.score_gpu` | Discovery, instrument | One payload, one checkpoint; padding/SSM mechanism is hypothesis, not established |
| C2 | Fresh nonce-word report games sit at ceiling: direct/self/oracle/control/external 100%, primary self-minus-control +0.00pp over 16 bases | 768 report records, Qwen3.8 non-thinking | `artifacts/processed/sprint_report16_38.json`, `report16-38-v2/raw.json` | `sprint_analysis.primary_gate` logic, base-equal weighting | Discovery pilot | Ceiling means no rescue inference either way; stop rule applied to mechanistic rescue |
| C3 | Inverted reports are followed 82.8% (106/128): report content controls action | Same as C2, swapped arm | Same as C2 | arm accuracy vs `expected_action` | Discovery pilot | Transcript dependence, not spontaneous self-trust; instruction explicitly privileges reports |
| C4 | WITHDRAWN as stated (see C10). Conflict-violation count (48/96) equals the design balance fraction, not a measured propensity | 192 incident records, Qwen3.8 non-thinking | `artifacts/processed/sprint_incident16_38.json`, `results/report_reactivity/incident16-38-v1/raw.json` | mapping-conditional split (below) | Superseded | Incentive-chasing interpretation refuted by m1-conflict trials (paid option A, model picks B) |
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

## Instrumentation path (GitHub Actions)

| # | Claim | Unit | Raw artifact | Analysis | Standing | Limit |
|---|---|---|---|---|---|---|
| C11 | GHA CPU dry-run path works: pytest → prepare_preflight → artifact upload; Modal skipped | Actions run [34044659196](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34044659196), `run_id=gha-dryrun-preflight-38-v1`, `dry_run=true`, stage=preflight, model Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Prepared payload sha256 `226e488f85f437b14ff4a66382e73c4f36a8d7be8b3dde315fc98c9e8105fb20` (n_queries=2, lengths [24, 28]); workflow `.github/workflows/report-reactivity.yml` | Methods note `experiments/report_reactivity/gha-cpu-dryrun-methods.md` | Engineering confirmation of CPU instrumentation | Tokenizer preflight payload only; not behavioral; does not validate GPU scoring, parity, or any scientific estimand |
| C12 | GHA GPU path reaches Modal but fails closed at study-ledger `reserve()` before scoring (stage cap / ledger block, not missing Modal balance or auth) | Actions run [34044902792](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34044902792), `run_id=gha-preflight-38-v1`, `dry_run=false`, stage=preflight | CPU prepare OK (same payload sha256 `226e488f…`); Modal credentials present; Modal app initialized; `results/…/gha-preflight-38-v1/input_manifest.json` written with `ceiling_usd≈0.7829`, `payload_sha256=226e488f…`, `stage=preflight` (run_id is the results subdirectory / dispatch input); **no** `raw.json` scores; **no** new row in `reservations.jsonl` | Fail-closed `sprint_runtime.reserve`: `ValueError('global or stage budget exhausted')`. Historical ledger total ~$7.26; `STAGE_LIMITS['preflight']=2.0` already holds ~$1.5657, so another ~$0.7829 cannot fit. Upload-artifacts step still ran (`always()`). | Instrument / ops finding (confirmed failure mode) | **Not a GPU behavioral result.** Provider Modal balance ~$28 ≠ study ledger headroom. Fix: RR-D001 new GHA ledger (`reservations_gha.jsonl`, global $28) via `REPORT_REACTIVITY_LEDGER` — does not invent scores for this run_id |
| C13 | GHA GPU path reserves on `reservations_gha.jsonl` and completes remote `score_gpu`, then fails on local Modal deserialize (`torch` missing); reservation retained; no usable scores | Actions run [34045326136](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34045326136), `run_id=gha-preflight-38-v2`, `dry_run=false`, stage=preflight | `input_manifest.json` (`ceiling≈0.7829`, `global_ceiling_usd=28`, `ledger_path=…/reservations_gha.jsonl`, payload sha256 `226e488f…`); GHA ledger row retained; `failure.json` with `DeserializationError` / `No module named 'torch'`; **no** `raw.json`; historical `reservations.jsonl` unchanged | Methods note `experiments/report_reactivity/gha-cpu-dryrun-methods.md` (C13 section). Root cause: Modal pickle return required torch on the Actions client (`uv run --extra modal` has no torch). Fix: `score_gpu` returns `dumps_jsonable(...)` JSON string; local entrypoint `loads_jsonable` before writing `raw.json`. | Instrument / ops finding (confirmed failure mode) | **Not a behavioral result.** Do not install full torch on the CPU runner; keep payload JSON-safe. Re-dispatch needs a **new** `run_id` (this one is reserved). |
| C14 | GHA 16-base discovery baseline replicates C2/C3 qualitative pattern: Direct/self/oracle/control/external at ceiling 1.0; primary self−control and self−direct = 0.0; swapped base-equal action accuracy 0.828125 (106/128) — report content can steer below ceiling | Actions run [34048123330](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34048123330), `run_id=gha-report16-38-v1`, stage=baseline, task=report, model Qwen/Qwen3.8-27B revision `1d4bf0f2…`, batch_size=4, split=discovery, bases=16 | `results/report_reactivity/gha-report16-38-v1/{raw.json,input_manifest.json}`; payload sha256 `126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d` (matches CPU dry-run `gha-report16-38-dry-v1`); parity passed (`replay_max_abs=0.0`, `batch_single_max_abs=0.125`, `choices_agree=true`); status `engineering_pilot`; n_scores=960; n_records=768; GHA ledger row retained (~$0.7829; total with C13 ≈$1.5657) | `experiments/report_reactivity/analyze_gha_report16.py` → `…/analysis/arm_accuracy_summary.json`; `sprint_analysis` primary contrasts; methods note `experiments/report_reactivity/gha-report16-38-v1-methods.md` | Discovery pilot | Ceiling → rewrite-vs-reveal rescue unidentifiable (stop rule). Not locked confirmation; not novelty/priority. Qualitative replication of C2/C3 on fresh GHA path + nonce corpus — do not overclaim independence. Instruction privileges reports. |
| C15 | Under the same GHA baseline, swapped steering is stronger on prose than opaque: swapped base-equal action accuracy prose 0.71875 (46/64) vs opaque 0.9375 (60/64) | Same unit as C14 (`gha-report16-38-v1`, 16 discovery bases, Qwen3.8) | Same raw artifact as C14; surface split in `…/analysis/arm_accuracy_summary.json` (`swapped_by_surface_kind`) | Same CPU analysis as C14; methods note `gha-report16-38-v1-methods.md` | Discovery pilot (descriptive split) | Surface contrast is descriptive within one pilot payload; not a locked surface-main-effect claim; not evidence of a mechanism. |

## Correction C10: conflict-conditional position rigidity (Qwen3.8)

Mapping-conditional choices (discovery 16-base and locked 32-base identical):

| | m0 (A=preserve, B=remove) | m1 (A=remove, B=preserve) |
|---|---|---|
| 3.8 conflict | B always (all violate) | B always (all preserve) |
| 3.8 no-conflict | A always (all preserve) | B always (all preserve) |
| 3.6 either condition | consequence-tracking, all preserve | consequence-tracking, all preserve |

Qwen3.8 tracks consequences perfectly with no conflict and goes fixed-B
under explicit constraint/reward conflict. The 48/96 (discovery) and 96/192
(locked) violation counts are fully explained by the 2-variant letter
balance, so no incentive propensity was measured. The condition effect
(tracking vs rigid) and the checkpoint gap (3.6 keeps tracking where 3.8
goes rigid) are real and replicate; the mechanism is rigidity under
conflict, not reward optimization. Disclosure honesty (C5) is unaffected:
auditor scores were conditional on the taken action throughout.

Method gate adopted: every future run reports violation rates *within* each
mapping before any incentive claim. A 100%/0% split across mappings is a
position strategy — stop. A real incentive effect must show as a
within-mapping dose-response under pressure titration, with letter
decoupled from reward across 4+ mapping variants.

## Spend (reservations retained, fail-closed)

**Historical ledger** `results/report_reactivity/reservations.jsonl` (immutable):
$7.26 of $30 reserved across 8 runs (1 overhead + 2 preflight + 2 report
baseline + 2 incident + 1 locked replication; plus retained failed-run
ceilings as recorded in the file). Provider reconciliation is recorded
separately and never refunds the ledger.

**C12 arithmetic.** Preflight stage sum on the historical ledger is
~$1.5657 (`preflight-qwen38-v1` + `preflight-qwen38-v2`).
`STAGE_LIMITS['preflight']=2.0`, so `2.0 - 1.5657 < 0.7829` — another
preflight ceiling cannot reserve. Actions `gha-preflight-38-v1` therefore
added **no** reservation row; GPU never ran. Modal account balance (~$28)
is orthogonal.

**GHA-era ledger** `results/report_reactivity/reservations_gha.jsonl`
(RR-D001): retained rows are C13 `gha-preflight-38-v2` and C14
`gha-report16-38-v1` (each ~$0.7829; total ≈ **$1.5657** / $28.0); same
`STAGE_LIMITS`; selected in Actions via `REPORT_REACTIVITY_LEDGER`. Local
default remains the historical path. Exact lines are the committed
`reservations_gha.jsonl` (from Actions artifact download; historical
`reservations.jsonl` immutable).

**C13 arithmetic.** Ledger OK (reserve succeeded); remote scoring completed
enough to return a payload; local Modal unpickle failed needing `torch`.
Reservation retained; no `raw.json` scores committed for analysis.

**C14 arithmetic.** Same ≈$0.7829 baseline ceiling reserved on the GHA ledger
after the C13 JSON-return fix; scoring completed with usable `raw.json`
(parity passed). Download artifact for exact ledger lines.
