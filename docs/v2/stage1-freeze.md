# Stage 1 locked-test freeze

This document freezes the once-opened Stage 1 locked test after all permitted
discovery and validation inspection and before any locked-family behavior.

## Immutable identifiers

- Dataset SHA-256: `6cb715641f0bcdccc9c7248b4806f81ac170123f3d93d534363ee2bdd9532d89`
- Stage 1 config SHA-256: `cfac817c1d6ab9924570899ac6ac111cf992936a554c8d678a0e16040fa40489`
- Probe artifact SHA-256: `5cb49d7d6adfd0d71a39fd09893292fc556bed4ce71d525708b8ad7b7edfbddb`
- Development run ID: `b4f53930df69466a8af312d65a231f8e`
- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Lens revision: `qwen-n1000`
- Lens code commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Runtime: PyTorch `2.13.0+cu130`, Transformers `5.15.0`, bfloat16,
  NVIDIA A100-SXM4-80GB
- Primary position/band: final pre-output prompt token, layers 36–43
- Bootstrap: 2,000 base-scenario draws, seed 981723

## Locked families

No behavior or mechanistic output has been opened for these families:

- Stage 1A: `marine_buoy`, `clinic_alias`, `robot_memory`
- Stage 1B: `largest_inventory`, `only_eligible`, `checksum_match`

Each family has ten base scenarios and every scenario retains all eight
state-by-policy rows.

## Frozen probes

All coefficients, intercepts, discovery-only scalers, class orders, and full
validation selection tables are stored in
`configs/v2/stage1_probe_freeze.json`.

| Substage | Probe | Layer | C |
|---|---|---:|---:|
| 1A | state, all policies | 36 | 1.0 |
| 1A | report, all policies | 36 | 0.01 |
| 1A | cross-policy state | 43 | 10.0 |
| 1B | state, all policies | 43 | 10.0 |
| 1B | report, all policies | 43 | 10.0 |
| 1B | cross-policy state | 43 | 0.1 |

Cross-policy artifacts contain separate truthful-trained and transformed-
trained models under the shared selected layer and C.

## Frozen outputs and claim ladder

The locked behavior command is run first. If either substage misses its 90%
overall, 85% cell, or 80% family threshold, its mechanistic test is invalid and
unopened.

For each behaviorally valid substage:

1. `transformed_band_K` must have a scenario-bootstrap 95% interval above zero.
2. At least two of three locked families must have positive mean transformed K.
3. The all-policy state probe's locked accuracy interval must be above 0.25.
4. All three conditions must hold for the substage to meet the frozen
   dissociation criterion.
5. Stage 1A alone licenses only persistence of explicitly supplied state.
6. Language about inferred latent state requires Stage 1B to pass separately.
7. Comparisons against output logits and the conventional logit lens are
   quantitative context, not additional conjunctive success criteria.

The report-preparation effect, state-policy effect, interaction, full layer
trajectory, cross-policy probe metrics, calibration, and AUROC remain frozen
reported outcomes. No exploratory development result changes the confirmatory
definitions.

