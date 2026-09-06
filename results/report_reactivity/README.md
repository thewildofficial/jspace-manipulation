# Report-reactivity — when words steer the next press

## The question

On fresh nonce-word consequence games, does asking the model to report what an
action will cause change (or merely reveal) what it then chooses? And when the
transcript already contains a lied report, does the model follow the words?

This chapter is the discovery pilot trail on the GitHub Actions scoring path
(Qwen3.8-27B, non-thinking, batch size 4). It is not locked confirmation.

## What was held constant

- Same consequence-game family (A/B actions, factorial frame × surface × policy).
- Frozen checkpoint revision and CPU prepare → Modal GPU score pipeline.
- Parity gate before scoring; study spend on the GHA ledger
  (`reservations_gha.jsonl`), separate from the older local ledger.
- Prepared inputs for scored runs are committed as `prepared.json` next to
  `raw.json` (Actions artifacts alone expire in ~14 days).

## What happened

**Ask-first ceiling (GHA baseline).** On 16 discovery bases, Direct, self-report,
matched control, oracle, and external-facts all sat at perfect action accuracy.
Primary self-minus-control and self-minus-direct were zero. Ask-first “rescue”
cannot be measured when Direct is already at the ceiling.

**Swapped / poisoned self-talk.** With inverted reports in the transcript, action
accuracy fell below ceiling (~0.83 overall). The drop was larger on prose than
opaque, and the worst cell was wordy opposed strategic framing (~0.375). Words in
the transcript can steer the press; that is descriptive in this pilot, not a
mechanism claim.

**Harder-games null.** Extra correct demonstration history was meant to break the
Direct ceiling so rescue would become identifiable. It did not: Direct and the
other “honest” arms stayed at 1.0. Swapped still leaked (~0.79). Ask-first rescue
remains blocked this way.

**Mid-trajectory ask (scores landed; story not yet claimed).** A follow-up
protocol inserts a consequence question between two forced choices. The first
GPU score pack is committed under `ask-mid-traj-qwen38-n16-v1/` (parity passed;
864 scores). Human-facing arm analysis and claim-ledger rows are **not** written
here yet — do not treat the raw file as a paper result.

## Where the evidence lives

| Story | Results dir | Key files |
|---|---|---|
| Ask-first GHA baseline | [`gha-report16-38-v1/`](gha-report16-38-v1/) | `raw.json`, `prepared.json`, `analysis/arm_accuracy_summary.json` |
| Harder games (redundant demos) | [`harder-games-qwen38-n16-v1/`](harder-games-qwen38-n16-v1/) | `raw.json`, `prepared.json`, `analysis/arm_accuracy_summary.json` |
| Mid-trajectory ask (raw only) | [`ask-mid-traj-qwen38-n16-v1/`](ask-mid-traj-qwen38-n16-v1/) | `raw.json`, `prepared.json` (no analysis JSON yet) |

Methods and cell maps live under
[`../../experiments/report_reactivity/`](../../experiments/report_reactivity/).
CPU joins use `analyze_gha_report16.py` and `analyze_harder_games.py` (immutable
`write_new`; existing analysis paths fail closed).

Earlier Modal-local pilots (`report16-38-v2`, incident, locked, preflights) keep
their historical `raw.json` / manifests; only the GHA-era scored runs above have
recovered permanent `prepared.json` copies. Local prepare scratch under
`artifacts/prepared/` stays disposable.

[Results map](../README.md) · [Project README](../../README.md)
