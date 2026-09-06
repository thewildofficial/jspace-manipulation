# Poisoned self-talk: swapped-arm cell map

**Story.** When the model is given a *lied* (swapped) report transcript before
acting, action accuracy falls below ceiling — and the drop concentrates in
**wordy opposed strategic** cells. Opaque surfaces mostly hold.

Machine ids (Actions run, `run_id`, claim footnote **C16**) are secondary; see
the JSON sibling and claim ledger.

## Source (no new GPU)

Re-analysis of the existing GHA 16-base discovery baseline:

| Field | Value |
|---|---|
| Story run | GHA report baseline, 16 discovery bases, Qwen3.8 |
| Actions | [34048123330](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34048123330) |
| Artifact pattern | `gha-report16-38-v1` (prepared dry twin `gha-report16-38-dry-v1`) |
| Arm | `swapped` (poisoned / lied report tokens before the common action query) |

CPU join only: prepared records + committed `raw.json`. No Modal dispatch from
this note.

## Overall swapped

- Accuracy **≈ 0.828** (106/128)
- Failures: **prose 18**, **opaque 4**

## Cell map (n=16 each)

| Surface | Policy | Frame | Accuracy |
|---|---|---|---|
| prose | opposed_assertion | strategic | **0.375** |
| prose | opposed_assertion | nonagentic | **0.6875** |
| prose | direct_policy | strategic | **0.8125** |
| prose | direct_policy | nonagentic | **1.0** |
| opaque | (most cells) | | **0.875–1.0** |

Exact opaque cells: opposed/strategic 0.875, opposed/nonagentic 1.0,
direct/strategic 0.875, direct/nonagentic 1.0.

## Methods note

1. Load prepared report payload (sha256
   `126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d`) and
   `results/report_reactivity/gha-report16-38-v1/raw.json`.
2. Score the swapped arm: choice vs `expected_action`.
3. Stratify by `surface_kind × policy_kind × frame` (factorial cells, 16 bases
   each).

Companion machine-readable table:
[`poisoned-self-talk-cell-map.json`](poisoned-self-talk-cell-map.json).

## Deployment implication

**Lied transcripts hurt most under wordy opposed strategic framing.** Monitoring
or HITL that trusts verbose self-talk in opposed-assertion tool settings is the
highest-risk cell in this pilot; opaque / direct-policy surfaces are relatively
protected here.

## Redundant-history replication (harder games)

Same factorial cell map under `history_mode=redundant`
(`harder-games-qwen38-n16-v1`, Actions
[34051102729](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34051102729);
claim **C18**):

- Overall swapped ≈ **0.789** (was ~0.828 under minimal)
- **Worst cell still prose + opposed + strategic**, now **0.5** (was 0.375)
- Strategic cells remain worse than nonagentic; Direct still ceiling (C17)

This is a replication under a harder demo dial, not a locked three-way
interaction claim. See
[`harder-games-qwen38-n16-v1-methods.md`](harder-games-qwen38-n16-v1-methods.md).

## Limits

Discovery pilot only; instruction privileges reports; not a locked surface ×
policy × frame interaction claim.
