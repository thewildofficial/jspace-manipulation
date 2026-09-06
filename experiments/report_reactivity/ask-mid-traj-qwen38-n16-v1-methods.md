# Methods note: Mid-trajectory ask discovery (`ask-mid-traj-qwen38-n16-v1`)

Status: **scored discovery pilot**. Companion ledger rows: **C19** (self mid-ask
is not an intervention), **C20** (swapped mid content steers choice2), **C21**
(descriptive opaque>prose flip asymmetry — opposite direction from ask-first).
Not locked confirmation.

## Story first

**Asking mid-trajectory does nothing.** On this 16-base pilot, a consequence
self-ask between two forced presses leaves flip and choice2 accuracy unchanged
versus matched non-consequence chatter (intervention contrast **0.0**). Control,
self, and oracle arms sit at ceiling: choice1 = choice2 = persist = **1.0**.

**A lied mid-answer rewrites ~60% of second presses.** With an inverted mid
token injected (`mid_ask_swapped`), base-equal flip is **0.59375** and choice2
accuracy falls to **0.40625**. Every flip lands on the wrong action; every
persist stays correct. Content sensitivity without self-ask intervention —
stop rule 2 of the protocol.

## Run identity

| Field | Value |
|---|---|
| Canonical protocol ID | `ask_mid_trajectory` |
| Story run name | Mid-trajectory ask, 16 discovery bases, Qwen3.8 |
| Actions run | [34052527423](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34052527423) |
| `run_id` / results dir | `ask-mid-traj-qwen38-n16-v1` |
| Stage | `baseline` |
| Task / study_id | `ask_mid_trajectory` |
| Split | `discovery` |
| Bases | 16 |
| Batch size | 4 |
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Payload sha256 | `50b854ae18ab584a25234e62687de4d4f6de8c467d5656ec9dda89a0b1d7a5d6` |
| Parity | passed — `replay_max_abs=0.0`, `batch_single_max_abs=0.125`, `choices_agree=true` |
| Status | `engineering_pilot` |
| Scores / records | 864 scores; 512 records |

## Immutable artifacts

Committed under `results/report_reactivity/ask-mid-traj-qwen38-n16-v1/`:

- `raw.json` — Modal `score_gpu` JSON return (parity + scores)
- `prepared.json` — exact CPU prepare payload (sha256 matches `payload_sha256`)
- `input_manifest.json` — reserve metadata (`ledger_path=reservations_gha.jsonl`)
- `analysis/arm_accuracy_summary.json` — CPU join (immutable `write_new`)

## Scoring and analysis

```bash
uv run python experiments/report_reactivity/analyze_ask_mid_trajectory.py \
  --prepared results/report_reactivity/ask-mid-traj-qwen38-n16-v1/prepared.json \
  --raw results/report_reactivity/ask-mid-traj-qwen38-n16-v1/raw.json \
  --output results/report_reactivity/ask-mid-traj-qwen38-n16-v1/analysis/arm_accuracy_summary.json
```

Fails closed if the analysis path already exists (no unlink).

## Authoritative arm metrics (base-equal, 16 bases)

| Arm | choice1 | choice2 | persist | flip | mid correct |
|---|---:|---:|---:|---:|---:|
| `mid_no_ask_control` | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| `mid_ask_self` | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| `mid_ask_oracle` | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| `mid_ask_swapped` | 1.0 | **0.40625** | **0.40625** | **0.59375** | 0.0 |

Intervention contrast `mid_ask_self − mid_no_ask_control`: flip **0.0**, choice2
accuracy **0.0**.

### Swapped flip cells (n=16 each)

| Cell | Flip |
|---|---:|
| opaque \| opposed \| strategic | 0.75 |
| opaque \| direct \| strategic | 0.75 |
| prose \| opposed \| strategic | 0.625 |
| prose \| direct \| strategic | 0.5625 |
| several nonagentic | ~0.5625 |
| prose \| direct \| nonagentic | 0.375 |

Strategic overall flip ~**0.67** vs nonagentic ~**0.52**.

### Descriptive opaque/prose asymmetry (not locked)

Base-equal swapped flip: opaque **0.65625** vs prose **0.53125**
(opaque−prose **+0.125**). Across 16 bases: opaque>prose on 7, prose>opaque on 2,
ties 7. Within base×policy×frame pairs (n=64): opaque-only flips 11 vs prose-only
3. Mean c1→c2 length delta ≈93.25 prose vs 93.69 opaque — not an obvious length
confound. **Opposite direction** from ask-first swapped (C15/C16), where prose
failed more than opaque.

## Limits

- Discovery pilot only; not locked confirmation.
- Self mid-ask is not an intervention here; do not spend locked budget on
  mechanistic follow-ups of “ask-as-intervention” without a new design.
- Opaque>prose flip split is descriptive — not a locked interaction.
- Not scheming; not HITL-general.
