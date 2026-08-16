# Causal manipulation of reporting policies in J-space

Can a language model continue to represent what is true while a causal
intervention changes the policy it uses to report that truth?

This repository tests that narrow mechanistic question in
`Qwen/Qwen3.6-27B` with the pinned Anthropic Jacobian Lens. It does not equate
instructed concealment with real-world deception, model intent, or
consciousness.

> **Current status: the causal V2 branch is blocked.** The original H0
> instrumentation gate failed, and the preregistered H0R recovery study later
> failed prospective H0R-C. H0R-D and all strategic-reporting experiments were
> therefore not run. No validated Qwen intervention protocol exists.

## What happened

The early V1 work established useful but observational prerequisites:

- the initial 4B candidate failed its balanced behavioral screen and was
  retired;
- Qwen3.6-27B passed the toy behavioral gate (192/192 rows);
- the released 27B lens passed structural/reference checks;
- a balanced discovery pilot found a middle-layer policy-family signal under
  named and unlabeled rules.

V2 then required a causal instrumentation gate before any strategic task:

1. **Original H0 failed.** Numerical transport, direction, and analytic-swap
   checks passed, but the preregistered workspace-band/all-position topology
   produced 0/90 target-answer top-1 outcomes. An exploratory one-layer,
   all-position topology produced 25/90 overall and 24/33 on countries. The
   original verdict remains: **no strategic-reporting experiment is licensed
   or was run**.
2. **H0R diagnosed the topology.** Three burned-country GPU diagnostics found
   exact fp32 coordinate exchange and a dose-responsive lower-strength regime.
   The frozen selection rule chose layers 36--42, argument-through-end
   positions, and alpha 0.5. On burned data it produced mean target-vs-source
   log-odds gain +4.34 with low KL and displacement.
3. **The first fresh H0R-C corpus was invalid.** Baseline accuracy was 60.8%,
   below the frozen 80% gate, so its runner stopped before intervention.
4. **Replacement H0R-C failed prospectively.** The simplified fresh corpus
   passed baseline at 125/130 (96.2%). The frozen semantic intervention had
   mean log-odds gain +3.53 (95% scenario-bootstrap interval +2.68 to +4.29),
   91.2% positive direction, and a +2.82 advantage over random/unrelated
   controls, while remaining inside all validity limits. However, only 4/125
   eligible trials made the target answer top-1 (3.2%), below the
   preregistered 20% minimum. Because the gate was conjunctive, H0R-C failed.

The strong directional effect is a useful mechanistic diagnostic, but it does
not satisfy the locked criterion for reliable counterfactual control. H0R-D
was prohibited by design. Continuing the causal study now requires a new
preregistration, model/lens change, or redesigned instrument—not further tuning
on the strategic task.

## Evidence map

| Stage | Result | Evidence |
|---|---|---|
| 4B behavioral screen | Failed and retired | [`results/4b_behavioral_screen/`](results/4b_behavioral_screen/README.md) |
| 27B behavioral gate | Passed, 192/192 | [`results/27b_behavioral_gate/`](results/27b_behavioral_gate/README.md) |
| 27B lens integrity | Passed structural/reference checks | [`results/27b_lens_integrity/`](results/27b_lens_integrity/README.md) |
| Observational pilot | Candidate signal; no causal claim | [`results/27b_observational_pilot/`](results/27b_observational_pilot/README.md) |
| V2 H0 | Failed causal topology | [`results/v2_smoke_tests/`](results/v2_smoke_tests/README.md) |
| H0R-B diagnostic | Candidate selected on burned controls | [`results/v2_h0r_diagnostic/`](results/v2_h0r_diagnostic/) |
| First H0R-C corpus | Invalid 60.8% baseline; intervention unopened | [`results/v2_h0r_argument_validation/`](results/v2_h0r_argument_validation/README.md) |
| Replacement H0R-C | Failed 20% target-top-1 criterion | [`results/v2_h0r_argument_validation_v2/`](results/v2_h0r_argument_validation_v2/README.md) |
| H0R-D / strategic V2 | Not run | [`docs/v2/h0r-final-report.md`](docs/v2/h0r-final-report.md) |

The append-only methodological record is in
[`docs/v2/decision-log.md`](docs/v2/decision-log.md). The original H0 and H0R
preregistrations, analysis rules, diagnostic interpretation, and terminal
report are under [`docs/v2/`](docs/v2/).

## Claim boundary

A readable J-space feature without a selective causal effect is observational,
not a mechanism. A directional log-odds change without reliable target-answer
conversion is evidence of partial causal access, not a validated instrument.
The project required both unseen argument substitution (H0R-C) and a computed
intermediate (H0R-D) to pass prospectively before testing latent truth and
strategic reporting. That conjunction was not met.

The repository therefore supports three narrow conclusions:

- the pinned J-lens transport and coordinate-write implementation are
  numerically sound;
- Qwen has a measurable, selective, strength-dependent response to the frozen
  J-space write;
- that write did not achieve the preregistered level of reliable
  counterfactual control needed for the strategic study.

## Repository map

```text
src/jspace_policy/
  dataset.py              deterministic factorial prompt generation
  analysis.py             paired estimates and bootstrap intervals
  interventions.py        J-lens directions and residual hooks
  h0r.py                   locked positive-control generators
  h0r_diagnostics.py       coordinate, conditioning, and reconstruction checks
scripts/
  analyze_results.py       V1 summaries and figures
  analyze_v2_h0.py         original H0 summaries and figures
  analyze_h0r.py           H0R-B selection tables and nine diagnostic figures
modal_app.py               V1 benchmark/readout entry points
modal_v2.py                original bounded H0 GPU runner
modal_h0r.py               H0R freeze, diagnostic, and prospective runners
configs/v2/                preregistered gates and immutable H0R controls
docs/v2/                   V2/H0/H0R plans, decisions, and reports
results/                   committed raw evidence, summaries, and figures
```

## Local verification and analysis

Set up the project and run the non-GPU checks:

```bash
uv sync --extra dev --extra modal
uv run pytest
```

Regenerate committed analyses from their immutable raw rows:

```bash
uv run python scripts/analyze_v2_h0.py
uv run python scripts/analyze_h0r.py
```

The H0R analysis emits the full layer curve, top-1 curve, position and strength
sweeps, cumulative-layer comparison, coordinate trajectory, reconstruction
curve, effect-versus-distance plot, and semantic-versus-control comparison in
[`results/v2_h0r_diagnostic/figures/`](results/v2_h0r_diagnostic/figures/).

The Modal entry points preserve the exact GPU workflow used for the committed
runs. Prospective entry points intentionally refuse to overwrite existing
result directories. Do not rerun or retune them as though the existing locked
controls were fresh.

## Interpretation and novelty

Activation probes and steering directions for truthfulness or deception already
exist, and the Jacobian Lens work already demonstrates causal concept swaps.
The proposed novelty here was the stronger combined test of a reporting policy:
a vocabulary-grounded J-space intervention, factorial world-state controls, a
required compositional output change, and independent fact preservation.

That full claim was not reached. The retained result is a transparent negative
and diagnostic study: readable and directionally causal structure did not
become a sufficiently reliable instrument for the planned latent-knowledge and
strategic-reporting experiment.

## Primary references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) and the [reference implementation](https://github.com/anthropics/jacobian-lens).
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Burns et al., [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).
- Levinstein and Herrmann, [Still No Lie Detector for Language Models](https://arxiv.org/abs/2307.00175).
- Wang, Zhang, and Sun, [When Thinking LLMs Lie](https://arxiv.org/abs/2506.04909).
