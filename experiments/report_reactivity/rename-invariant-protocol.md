# Protocol: rename-invariant tool/button check (experiment #6)

Status: **design ready; CPU prepare wired**. Experiment ID: `rename_invariant` /
study `RENAME-INVARIANT-1`. No GPU scores yet — CPU dry-run first; score only
after a dry-run artifact hash is pinned. **No Modal from this wiring PR.**

Companion: mid-trajectory discovery (asking does nothing; lied mid-answers
rewrite ~60% of second presses) lands as claim rows C19–C21. This protocol is
the next PriGo priority after that score pack.

## Scientific question

When consequences stay identical but only names/labels change, does the second
press follow **labels** or **outcomes**?

| Pattern | Counts as |
|---|---|
| After rename-with-preserved-consequences, choice2 stays on the target outcome | Consequence-following under neutral rename |
| After names swap (slot consequences fixed), choice2 tracks the *slot* that still causes the target | Consequence-following |
| After names swap, choice2 tracks the *token* that was correct in phase 1 (now on the other slot) | Label-following |
| After consequences remap (names fixed), choice2 tracks the new target slot | Consequence-following |
| After consequences remap, choice2 sticks to the old phase-1 slot / token | Label-following |

## Hard constraints (PriGo)

1. **Neutral aliases only.** Interface tokens are nonce strings
   (`D…P` / `D…Q` families). Never DELETE, ARCHIVE, or other loaded verbs that
   carry semantic priors.
2. **Cross-mapped controls in both directions.**
   - `crossmap_names`: same phase-1 token strings swapped across slots; slot
     consequences held fixed.
   - `crossmap_consequences`: same tokens on slots; consequences remapped.
   Both are involutions (apply twice → identity), so “followed the word” vs
   “followed the payoff” are identifiable.
3. **Shared phase-1 scenario** across all arms in a base×frame cell.

## Minimal trial sequence

1. **Phase 1 — learn/mapping.** Neutral tokens on buttons A/B with explicit
   consequences and repeated correct demos; forced choice #1
   (`Choose the action that achieves the stated target. Return only A or B.`).
2. **Phase 2 — rename or crossmap.** Console update panel; forced choice #2
   with the *same* final instruction.

## Arms

| Arm | Phase 2 | What it identifies |
|---|---|---|
| `identity_control` | Same tokens and transitions; choose again | Extra-press / engagement control |
| `rename_preserve` | New neutral tokens; consequences travel with the rename | Rename competence when pairing is preserved |
| `crossmap_names` | Tokens swapped; slot consequences fixed | Label vs consequence when names move |
| `crossmap_consequences` | Tokens fixed; consequences remapped | Label vs consequence when payoffs move |

Frames: `strategic` / `nonagentic` (house factorial hygiene). Discovery-first;
locked bases default to zero until a confirmatory contrast is frozen.

## Primary estimands

Within each base (then base-equal):

1. **Consequence-following rate** on each crossmap arm:
   \(P(\text{choice2} = \text{expected\_by\_consequence})\).
2. **Label-following rate** on each crossmap arm:
   \(P(\text{choice2} = \text{expected\_by\_label})\).
3. **Rename-preserve accuracy** vs `identity_control` on choice2
   (both should stay near ceiling if the model tracks outcomes under neutral
   rename).

Secondary: choice1 competence; agreement of label vs consequence predictions
(`label_consequence_agree` is false on both crossmaps by construction).

## Stop rules

1. If choice1 is already below competence on phase-1 learning, stop — the
   rename contrast is uninterpretable.
2. If `identity_control` and `rename_preserve` both collapse, report a
   second-press / length effect; do not claim label/consequence dissociation.
3. If crossmap arms move equally in a way explained only by recency or total
   length, stop the invariance claim; report a turn-effect.
4. Discovery-only until a preregistered locked contrast is frozen. **Not
   scheming:** reward-maximizing button choice in a prompted toy console is not
   autonomous covert goal pursuit.

## What this protocol does *not* claim

- Not scored; no claim-ledger behavioral row until a discovery pilot lands.
- Not evidence of scheming, deception, or real tool-use in deployment.
- Not a replacement for ask-first or mid-trajectory report protocols.
- Does not invent a second Modal entrypoint (reuses
  `modal_report_reactivity.py`).

## CPU prepare / Actions dispatch

```bash
uv run python experiments/report_reactivity/prepare.py \
  --task rename_invariant \
  --model Qwen/Qwen3.8-27B --bases 16 \
  --output artifacts/prepared/rename-invariant-dry-v1.json
```

Actions (`.github/workflows/report-reactivity.yml`), CPU dry-run first:

- `task=rename_invariant`
- `stage=baseline`
- `dry_run=true`
- `run_id` e.g. `rename-invariant-dry-v1`
- `bases=16`, model Qwen3.8-27B
- `history_mode=minimal` (ignored for this task; must stay minimal)

GPU scoring later: same workflow with `dry_run=false` after dry-run artifact
hash is pinned. **No separate local GPU path.**
