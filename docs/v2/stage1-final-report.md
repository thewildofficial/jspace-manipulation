# Stage 1 final report: latent state–report dissociation

This report is generated deterministically from the immutable Stage 1 raw results.

## Execution and behavioral validity

The locked run used model revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` and the pinned Jacobian Lens over layers 36–43 at the final pre-output prompt position.

| Phase | Substage | Accuracy | Minimum cell | Minimum family | Gate |
|---|---:|---:|---:|---:|---:|
| dev | 1A | 1.000 | 1.000 | 1.000 | pass |
| dev | 1B | 0.954 | 0.889 | 0.887 | pass |
| locked | 1A | 1.000 | 1.000 | 1.000 | pass |
| locked | 1B | 0.988 | 0.967 | 0.963 | pass |

## Confirmatory results

| Substage | Transformed K (95% CI) | Positive locked families | State probe accuracy (95% CI) | Criterion |
|---|---:|---:|---:|---:|
| 1A | -0.108 [-0.142, -0.073] | 0/3 | 1.000 [1.000, 1.000] | fail |
| 1B | -0.017 [-0.043, 0.008] | 2/3 | 0.992 [0.979, 1.000] | fail |

## Policy decomposition

| Substage | Report preparation ΔM (95% CI) | State change ΔK (95% CI) | Interaction (95% CI) |
|---|---:|---:|---:|
| 1A | 0.673 [0.618, 0.729] | -0.495 [-0.550, -0.446] | 1.167 [1.067, 1.267] |
| 1B | 0.275 [0.207, 0.351] | -0.087 [-0.127, -0.046] | 0.362 [0.256, 0.474] |

Both interactions are positive: report-target evidence increases under the transformed policy while true-state J-space evidence decreases. This policy sensitivity does not rescue the failed positive-K endpoint.

## Independent residual probes

| Substage | All-policy state | Truth→transformed | Transformed→truth | All-policy report |
|---|---:|---:|---:|---:|
| 1A | 1.000 [1.000, 1.000] | 0.475 [0.425, 0.525] | 0.250 [0.250, 0.250] | 1.000 [1.000, 1.000] |
| 1B | 0.992 [0.979, 1.000] | 0.700 [0.608, 0.792] | 0.492 [0.417, 0.567] | 0.933 [0.892, 0.971] |

Four-way chance is 0.25. Full AUROC, negative log-likelihood, and calibration metrics are in `probe_metrics_locked.csv`.

## Output-facing baselines

| Substage | J-space transformed K | Logit-lens transformed K | Output-logit transformed K |
|---|---:|---:|---:|
| 1A | -0.108 [-0.142, -0.073] | -0.016 [-0.041, 0.009] | -1.663 [-1.999, -1.353] |
| 1B | -0.017 [-0.043, 0.008] | 0.004 [-0.011, 0.017] | -0.616 [-0.945, -0.295] |

## Preregistration, deviations, and status

The model/lens revisions, four-state factorial structure, family-level splits, behavior gates, final pre-output position, layers 36–43, K/Q/M/D signs, probe grid, scenario bootstrap, success conjunction, and claim ladder were preregistered before mechanistic output.

Before any activation was opened, behavior-only evidence led to logged positive-control redesigns: a tokenizer-invalid label was replaced; an invalid raw-prompt format was switched to Qwen's pinned non-thinking chat rendering; and behaviorally weak inferred-state families were simplified or replaced. All failed behavior runs remain committed. Two no-output service/client interruptions were also logged. No family was changed after the passing development behavior gate, and no mechanistic definition changed after development inspection.

Development/validation results are exploratory or selection-only. The once-opened locked results and frozen pass/fail decision are confirmatory. The append-only decision log contains the complete information history.

## Interpretation

Neither substage meets the frozen J-space criterion. Stage 1A true-state J-space evidence is significantly negative, and Stage 1B is not distinguishable from zero. Stage 2 is therefore not licensed.

The independent residual probes nevertheless recover state well above chance on structurally unseen locked families. The licensed secondary conclusion is that state information is present in the residual stream but is not reliably surfaced through the pinned vocabulary-grounded J-space score.

This observational result does not establish deception, deceptive intent, consciousness, scheming, a general-purpose lie detector, or causal use of the decoded state. No Stage 0 or H0R failure is reinterpreted.

## Reproducibility

- Dataset, configuration, probe, model, lens, and code identifiers are recorded in `results/v2_stage1/run_manifest.json` and the raw metadata.
- All uncertainty uses the frozen 2,000-draw base-scenario bootstrap.
- Run `uv run python scripts/analyze_stage1.py --phase locked` to regenerate this report, tables, and figure.
