# Methods note: Harder games null (`harder-games-qwen38-n16-v1`)

Status: **discovery pilot / substantive null**. Companion ledger rows: **C17**
(Direct still ceiling under redundant demos → ask-first rescue still
unidentified), **C18** (swapped still leaks under redundant; strategic cells
worst). Not locked confirmation.

## Story first

We turned the difficulty dial: same nonce games, same six arms, but
**redundant correct prior-trial demos** instead of balanced minimal history —
hoping Direct would fall below ceiling so ask-first rescue becomes measurable.
It did not. Direct (and self / control / oracle / external) stayed at **1.0**.
Ask-first rescue remains unidentifiable this way. Swapped still leaks
(~**0.789**); the worst cell is still prose + opposed + strategic (now **0.5**).

## Run identity

| Field | Value |
|---|---|
| Story run name | Harder games (redundant demos), 16 discovery bases, Qwen3.8 |
| Actions run | [34051102729](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34051102729) |
| `run_id` / results dir | `harder-games-qwen38-n16-v1` |
| Stage | `baseline` |
| Task | `report` |
| Split | `discovery` |
| Bases | 16 |
| Batch size | 4 |
| `history_mode` | **redundant** |
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Payload sha256 | `a8b43df94451398b4edfecddb1a3f1c821c014e1d4212d6cc207be2dd077964c` |
| Parity | passed — `replay_max_abs=0.0`, `batch_single_max_abs=0.25`, `choices_agree=true` |
| Status | `engineering_pilot` |
| Timing | `elapsed_seconds≈340.4`, `load_seconds≈36.6` |
| GPU (artifact) | `NVIDIA A100 80GB PCIe` |
| Scores / records | 960 scores; 768 records |

## Immutable artifacts

Committed under `results/report_reactivity/harder-games-qwen38-n16-v1/`:

- `raw.json` — Modal `score_gpu` JSON return (parity + scores); full raw lives
  here and in Actions artifact for run 34051102729
- `prepared.json.gz` — exact CPU prepare payload for this run (sha256 matches
  `payload_sha256`; recovered from dry-run Actions artifact
  `harder-games-qwen38-n16-dry-v1` / run 34051045417)
- `input_manifest.json` — reserve metadata (`ledger_path=reservations_gha.jsonl`,
  `global_ceiling_usd=28`, ceiling ≈ `$0.7829`, `stage=baseline`)
- `analysis/arm_accuracy_summary.json` — CPU join of prepared records + scores

GHA spend ledger updated from the Actions artifact only:

- `results/report_reactivity/reservations_gha.jsonl` now includes
  `harder-games-qwen38-n16-v1` (total reserved ≈ `$2.3486` with prior C13
  preflight + C14 baseline retentions)
- Historical `results/report_reactivity/reservations.jsonl` **unchanged**

## Scoring and analysis

1. CPU prepare (Actions, then locally reproducible):

   ```bash
   uv run --with 'transformers>=5.5' --with 'huggingface-hub>=0.34' \
     --with 'httpx[socks]>=0.28' --with 'jinja2>=3.1' \
     python experiments/report_reactivity/prepare.py --task report \
     --model Qwen/Qwen3.8-27B --bases 16 --history-mode redundant \
     --output artifacts/prepared/harder-games-qwen38-n16-v1.json
   # Expect sha256 a8b43df94451398b4edfecddb1a3f1c821c014e1d4212d6cc207be2dd077964c
   ```

2. GPU scoring was the Actions `dry_run=false` dispatch (no new GPU from this
   documentation PR).

3. CPU analysis:

   ```bash
   uv run python experiments/report_reactivity/analyze_harder_games.py \
     --prepared results/report_reactivity/harder-games-qwen38-n16-v1/prepared.json.gz \
     --raw results/report_reactivity/harder-games-qwen38-n16-v1/raw.json \
     --output results/report_reactivity/harder-games-qwen38-n16-v1/analysis/arm_accuracy_summary.json
   ```

## Base-equal action accuracy (16 bases)

| Arm | Accuracy |
|---|---|
| direct | 1.0 |
| matched_control | 1.0 |
| self_report | 1.0 |
| oracle | 1.0 |
| external_facts | 1.0 |
| swapped | **0.7890625** (~0.789) |

## Swapped cells (n=16 each)

| Surface | Policy | Frame | Accuracy |
|---|---|---|---|
| prose | opposed_assertion | strategic | **0.5** |
| opaque | opposed_assertion | strategic | **0.625** |
| opaque | direct_policy | strategic | **0.625** |
| prose | direct_policy | strategic | **0.75** |
| prose | opposed_assertion | nonagentic | **0.8125** |
| prose | direct_policy | nonagentic | **1.0** |
| opaque | opposed_assertion | nonagentic | **1.0** |
| opaque | direct_policy | nonagentic | **1.0** |

## Contrast vs prior ask-first baseline (minimal history)

| | Minimal (`gha-report16-38-v1`, Actions 34048123330) | Redundant (this run) |
|---|---|---|
| Direct | 1.0 | 1.0 |
| Swapped overall | ~0.828 | ~0.789 |
| Worst cell (prose/opposed/strategic) | 0.375 | 0.5 |

Primary contrasts self−control and self−direct remain **0.0** (ceiling).

## What is NOT claimed

- That demos make games harder in a useful way
- Locked confirmation
- That ask-first rescue was measured (still blocked by Direct ceiling)
- A mechanism for the swapped leak
- Novelty/priority over prior C14–C16 beyond the controlled history dial
