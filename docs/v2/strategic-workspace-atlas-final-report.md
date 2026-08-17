# V2-E1 final report: Strategic Workspace Atlas

Status: **completed observational atlas with mixed prospective replication**.

## Executive result

V2-E1 supports a narrow selective-workspace result, not a strategic-intent
monitoring result.

Across unseen locked renderings of Kuhn poker and signaling, the pinned
Jacobian Lens surfaced a generic optimization family (`optimal`, `optimize`,
and `optimization`) in all 24 cases at layer 43. None of those tokens occurred
in the prompts. Their median best ranks were 27.5 in Kuhn and 11 in signaling,
and they preceded conventional-logit-lens action commitment in every locked
row.

However, the stronger strategy-routing hypotheses failed. Residual strategy
decoding averaged 0.719 balanced accuracy, but its mean advantage over the
stronger J-space/output baseline was only 0.090 versus the frozen 0.15
threshold, and it won only three of five games versus the required four. When
Kuhn was restricted to the same external action (`A=BET`), bluff, thin-bet, and
value-bet decoding was at chance in both residual and J-space readouts.

The most defensible interpretation is therefore:

> On these controlled tasks, J-space prospectively exposed a task-general
> optimization/choice abstraction before action commitment, but did not expose
> a reliably generalizable representation of the specific strategic meaning
> behind that action.

This is observational and does not reopen the failed causal H0/H0R gate.

## Execution record

The study used pinned `Qwen/Qwen3.6-27B` in non-thinking chat mode and the
existing 63-layer Qwen Jacobian Lens. The corpus contained six games, three
rendering splits, 12 rows per game/split, 216 rows total, and 108 matched
bootstrap groups.

Before model execution, eight atlas-specific tests independently recomputed
every payoff system, checked action/margin consistency, verified rendering-
split parameter invariance, required action and strategy dissociations, and
validated matched-group isolation. The complete local suite passed with 27
tests and 2 infrastructure-dependent skips.

Corpus v1 was rejected before mechanistic execution: pooled exact-game behavior
reached its 70% threshold, but Kuhn reached only 41.7% versus the 50% family
floor. A logged behavior-only redesign removed tied optima, widened small Kuhn
value margins, required literal use of supplied opponent policies, and made the
inspection/disclosure payoff rows explicit. No behavioral threshold or
mechanistic definition changed.

Corpus v2 passed both behavioral gates:

| Phase | Rows | Formatting | Exact-game accuracy | Minimum exact family |
|---|---:|---:|---:|---:|
| Open | 144 | 100% | 78.3% | 62.5% |
| Locked | 72 | 100% | 85.0% | 66.7% |

The open mechanistic run stored top-50 J-lens tokens at all 63 layers for 144
rows, plus 12 deterministic 32-position deep traces and 4,572 decoder
evaluations. Four endpoints were then frozen before locked behavior. The locked
run trained frozen classifiers on both open renderings and evaluated the third
rendering once.

## Frozen endpoints

### H1 — Generic optimization workspace: pass in both games

| Game | Locked presence | Median best rank at L43 | Before commitment | Result |
|---|---:|---:|---:|---|
| Kuhn | 100% | 27.5 | 100% | pass |
| Signaling | 100% | 11.0 | 100% | pass |

The token family consisted only of the frozen normalized strings `optimal`,
`optimize`, `optimally`, and `optimization`. In both games, `optimal`,
`optimize`, and `optimization` each appeared for every locked row. Prompt-echo
count was zero.

This is the cleanest positive result. It generalizes across a new rendering and
two distinct games, appears before the output action is conventionally
committed, and names a decision process rather than a legal action.

It does not identify *which* strategy is being used.

### H2 — Five-game strategy-routing dissociation: fail

| Game | Residual BA | J-space BA | Output BA |
|---|---:|---:|---:|
| Inspection | 1.000 | 0.500 | 1.000 |
| Kuhn | 0.375 | 0.250 | 0.312 |
| Cheap talk | 0.833 | 0.500 | 0.500 |
| Signaling | 0.389 | 0.333 | 0.333 |
| Disclosure | 1.000 | 0.500 | 1.000 |

Mean residual balanced accuracy exceeded the frozen 0.70 floor at 0.719.
Nevertheless, residual advantage over the stronger baseline was only 0.090,
and residual won three games. Both remaining conjuncts failed.

Inspection and disclosure illustrate why the output control matters: their
strategy labels are largely action-determined, so final action probabilities
decode them perfectly. Cheap talk produced the only large locked incremental
residual advantage. Kuhn and signaling strategy transfer was weak in every
representation.

The open residual result therefore did not justify a cross-game claim that
strategic policy is systematically retained outside J-space. Part of the open
signal was action, task, or rendering structure.

### H3 — Same-action Kuhn strategy: fail

Restricting to solver-optimal `A=BET` cases removes the most important action
confound. The three classes were bluff, thin bet, and value bet.

| Representation | Locked balanced accuracy |
|---|---:|
| Residual, L42 | 0.333 |
| J-space top-50, L42 | 0.333 |
| Final legal-action logits | 0.417 |

Residual accuracy missed the 0.65 floor and did not exceed the stronger
baseline. This is direct prospective evidence against the proposed headline
that the same output action carries a readily decodable, rendering-general
strategic meaning in this experiment.

It does not show that the model never represents bluff/value distinctions. The
task is small, behavioral poker accuracy is only 66.7%, and the tested decoder
is linear.

### H4 — Late action commitment: pass in both games

| Game | Locked censored median | Uncommitted through L62 | Result |
|---|---:|---:|---|
| Kuhn | 63 | 58.3% | pass |
| Signaling | 60 | 8.3% | pass |

Commitment is defined relative to the model's final legal action, not solver
correctness. An uncommitted row is censored at layer 63. The generic L43
optimization representation therefore has substantial depth lead time over
the stable conventional-logit-lens action in these two families.

That lead time is about *decision optimization*, not demonstrated strategic
intent.

## Descriptive findings not advanced as endpoints

- The controlled Chameleon task fell to 25% locked behavior. Its open
  `safe`/`risk` tokens were template-specific and were correctly excluded from
  replication.
- Top J-lens readouts in early layers were noisy and often dominated by
  multilingual or lexical artifacts. Intermediate layers around 41--46 shifted
  toward generic decision terms such as `choice`, `closest`, and `optimal`.
  Final layers increasingly contained answer-letter and formatting variants.
- `closest` was a prominent cheap-talk readout, consistent with its explicit
  squared-distance decision rule. It was not frozen as a strategy concept.

## What the study establishes

Within one pinned model/lens and these finite games:

1. a vocabulary-grounded generic optimization representation can appear well
   before stable output-action preparation;
2. that representation generalizes across unseen renderings in two games;
3. it does not entail that J-space distinguishes bluff, value, pooling,
   influence, or disclosure policy; and
4. apparent residual strategy decodability must be compared against action
   logits and same-action controls before receiving a strategic interpretation.

This sharpens the Stage 1 boundary. The negative result is no longer merely
that a privileged truth token was absent. Here, the workspace did surface a
meaningful intermediate abstraction—but at a more generic level than the
specific strategic variable the monitoring hypothesis would need.

## Limitations

- Locked sample size is 12 information sets per game, using an unseen rendering
  but the same parameter inventory.
- Inspection and disclosure v2 prompts include computed payoff rows, making
  them calibration games rather than clean tests of internally computed value.
- J-space decoders use top-50 reciprocal-rank features. They are not the formal
  sparse nonnegative gradient-pursuit decomposition from the workspace paper.
- Only linear residual probes were tested.
- The primary model is one 27B Qwen revision under one pretrained lens.
- The work is observational. It establishes neither causal use nor monitor
  robustness under adversarial optimization.

## Reproducibility and artifacts

- Protocol: `docs/v2/strategic-workspace-atlas-spec.md`
- Replication freeze: `docs/v2/strategic-workspace-atlas-freeze.md`
- Machine configs and immutable corpus: `configs/v2/workspace_atlas/`
- Raw behavior/mechanistic returns: `results/v2_workspace_atlas/raw/`
- Full top-token atlases: `results/v2_workspace_atlas/atlas/`
- Endpoint and decoder tables: `results/v2_workspace_atlas/summaries/`
- Figures: `results/v2_workspace_atlas/figures/`
- Preserved failed behavior corpus: files containing `v1_failed_gate`

Regenerate deterministic analysis with:

```bash
uv run python scripts/analyze_workspace_atlas.py --phase open
uv run python scripts/analyze_workspace_atlas.py --phase locked
```

Total measured buffered compute in the V2-E1 ledger is approximately $1.00,
well below the frozen $25 ceiling.
