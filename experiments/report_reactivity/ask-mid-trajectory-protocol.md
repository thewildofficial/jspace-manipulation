# Protocol: mid-trajectory ask as intervention

Status: **scored discovery pilot**. Canonical protocol ID:
**`ask_mid_trajectory`** (also `prepare --task`, Actions `task`, and
`study_id`). First GPU scores live under `run_id`
`ask-mid-traj-qwen38-n16-v1` →
`results/report_reactivity/ask-mid-traj-qwen38-n16-v1/` (Actions 34052527423,
parity passed). Arm analysis and claim-ledger rows **C19–C21** are written
(ledger indices, not alternate experiment names). Lead finding: asking
mid-trajectory does nothing; a lied mid-answer rewrites ~60% of second presses.
CPU prepare + Actions dry-run remain wired. See [NAMING.md](NAMING.md).

Companion context: harder-games substantive null (Direct still ceiling under
redundant demos → ask-first rescue still blocked); claim ledger C14–C18;
literature wedge on belief–action / CoT rationalization rarely treating the
*ask itself* as the intervention (HITL-shaped).

## Scientific question

Does inserting a consequence question **between** two forced action choices
change the second press (**intervention**), or only elicit a report
(**measurement / reveal**)?

| Pattern | Counts as |
|---|---|
| Choice2 accuracy or flip rate moves under consequence mid-ask vs matched non-consequence chatter | **Intervention** (ask changes the press) |
| Mid answer is accurate but choice2 matches the no-ask control | **Reveal / measurement** only |
| Mid answer is wrong / swapped and choice2 tracks the lied report | Report-content steering (secondary; maps to swapped arm of ask-first) |

## Minimal trial sequence

Same scenario family as report-reactivity (consequence games, A/B; factorial
frame × surface × policy). Per trial:

1. **Scenario** + **forced choice #1** — same final instruction as Direct:
   `Choose the action that achieves the stated target. Return only A or B.`
2. **Mid turn** — either a short consequence ask (“which outcome follows for
   action …?”) or a matched formatting/control check of similar length.
3. **Answer** — model-generated (`mid_ask_self`) or frozen oracle / swapped /
   control token.
4. **Forced choice #2** — *same* final instruction and options again.

## Arms

| Arm | Mid turn | What it identifies |
|---|---|---|
| `mid_no_ask_control` | Neutral formatting check + frozen expected marker | Extra turns / engagement without consequence content |
| `mid_ask_self` | Consequence check; model answers X/Y | Total mid-ask elicitation effect |
| `mid_ask_oracle` | Same check with correct answer injected | Effect with mid accuracy controlled |
| `mid_ask_swapped` | Same check with inverted answer injected | Sensitivity of choice2 to mid report content |

Default `history_mode=minimal` (backward-compatible with six-arm report
prepares). Redundant history remains available as a difficulty dial but is
**not** required for the first mid-trajectory dry-run.

## Primary estimands

Within each base (then base-equal):

1. **Persistence:** \(P(\text{choice2} = \text{choice1})\).
2. **Choice2 accuracy:** \(P(\text{choice2} = \text{expected\_action})\).
3. **Intervention contrast:** `mid_ask_self − mid_no_ask_control` on choice2
   accuracy and on flip rate \(P(\text{choice2} \neq \text{choice1})\).

Secondary / descriptive: oracle vs swapped mid content; agreement of mid token
with `expected_mid_token`; optional later “ask after act” (ask *after* choice2)
vs “ask before second act” (this protocol).

## Stop rules

1. If choice1 is already at ceiling **and** choice2 never flips under any mid
   arm, the intervention estimand is unsupported — report descriptive null;
   do not spend locked budget on mechanistic follow-ups.
2. If `mid_ask_self − mid_no_ask_control` ≈ 0 while `mid_ask_swapped` moves
   choice2, treat as **content sensitivity without self-ask intervention**.
3. If only total length / recency explains flips (control ≈ ask), stop the
   “ask-as-intervention” claim; report a turn-effect.
4. Discovery-only until a preregistered locked contrast is frozen; not a paper
   conclusion from the first pilot.

## CPU prepare / Actions dispatch

```bash
uv run python experiments/report_reactivity/prepare.py \
  --task ask_mid_trajectory \
  --model Qwen/Qwen3.8-27B --bases 16 \
  --history-mode minimal \
  --output artifacts/prepared/ask-mid-traj-dry-v1.json
```

Actions (`.github/workflows/report-reactivity.yml`), CPU dry-run first:

- `task=ask_mid_trajectory`
- `stage=baseline`
- `dry_run=true`
- `run_id` e.g. `ask-mid-traj-dry-v1` (results folder name only)
- `history_mode=minimal` (default)
- `bases=16`, model Qwen3.8-27B

GPU scoring later: same workflow with `dry_run=false` and
`pinned_payload_sha256` equal to the reviewed prepare sha256. Uses existing
`modal_report_reactivity.py` (scores all prepared queries). **No separate local
GPU path.**

## Legacy / path aliases (not IDs)

| Kind | String | Status |
|---|---|---|
| Historical study label | `ASK-MID-TRAJECTORY-1` | synonym of `ask_mid_trajectory`; do not invent new aliases |
| Markdown paths | `ask-mid-trajectory-*.md` | kebab display paths only |
| Results folder | `ask-mid-traj-qwen38-n16-v1` | `run_id` only — do not rename |

## What this protocol does *not* claim yet

- Discovery pilot only (C19–C21); not locked confirmation.
- Not a replacement for ask-first (self-report before any press).
- Not evidence that HITL questions always intervene.
- Does not invent a second Modal entrypoint.
