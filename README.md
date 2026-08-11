# Causal manipulation of reporting policies in J-space

Can a language model continue to represent what is true while a causal intervention changes the policy it uses to report that truth?

This repository is the reproducible experiment scaffold for answering that narrow question in `Qwen/Qwen3.6-27B` with Anthropic's released Jacobian Lens. It deliberately does **not** equate instructed concealment with real-world deception, model intent, or consciousness. The target claim is mechanistic: under controlled binary tasks, a fact representation and a context-dependent reporting computation may be empirically dissociable.

> **Status:** the 27B model passed the toy behavioral gate (192/192 balanced rows), its released J-lens passed structural/reference validation, and a small discovery pilot found a middle-layer policy-family signal under both named and unlabeled rules. This is observational evidence, not a mechanism. Held-out causal tests remain before any mechanistic claim.

The initial 4B candidate failed its balanced behavioral gate and was retired before mechanistic analysis. The complete negative screen is preserved in [`results/4b_behavioral_screen/`](results/4b_behavioral_screen/README.md). The successful replacement gate, including raw scores and an interpretable figure, is in [`results/27b_behavioral_gate/`](results/27b_behavioral_gate/README.md).

The released 27B lens now also passes structural integrity and a reference qualitative readout. Its complete top-token records are preserved in [`results/27b_lens_integrity/`](results/27b_lens_integrity/README.md). No causal intervention has yet been interpreted as a result.

The first balanced observational pilot is in [`results/27b_observational_pilot/`](results/27b_observational_pilot/README.md). It finds a candidate policy-family signal under both semantic and unlabeled rule wording, while also showing why the literal A/B J-lens score is inadequate as the independent fact-preservation metric.

## Decision rule

The main result is positive only if all of the following hold on a locked test split:

1. Unmodified behavior follows both reveal and conceal policies in every world-state cell.
2. A policy readout generalizes across world states and held-out scenario families.
3. The same intervention has a compositional sign flip: it favors literal `B` when the fact is `A`, and literal `A` when the fact is `B`.
4. The intervention outperforms norm-matched random, unrelated-semantic, lexical, and direct-answer controls.
5. An independently specified fact readout changes substantially less than reporting behavior.
6. The layer and strength are chosen on development data; the reported effect is estimated once on the locked test data with scenario-level uncertainty.

A readable policy token without a causal effect is an observational finding, not a mechanism. A behavioral flip that also flips the fact readout is answer/belief steering, not policy dissociation.

## Repository map

```text
src/jspace_policy/
  dataset.py          deterministic factorial prompt generation
  analysis.py         paired estimates and scenario bootstrap intervals
  plotting.py         publication-oriented figures from raw tables
  interventions.py    lens-vector construction and residual hooks
scripts/
  generate_dataset.py
  analyze_results.py
modal_app.py           metered GPU entry points for pilot/readout experiments
configs/
  experiment.json      locked defaults and artifact identifiers
  concepts.json        discovery/validation concept-family registry
docs/
  research-spec.md     improved design and preregistered claim boundary
  literature-review.md prior results, novelty boundary, and claim ladder
  analysis-plan.md     schemas, estimands, figures, and interpretation rules
artifacts/
  raw/                 immutable machine outputs (not committed by default)
  processed/           derived summary tables
  figures/             regenerated figures
```

## Reproduce locally

```bash
uv sync --extra dev --extra modal
uv run python scripts/generate_dataset.py --n-base 120 --seed 1729
uv run pytest
```

Run the small GPU benchmark and behavioral pilot on Modal:

```bash
modal run modal_app.py::benchmark
modal run modal_app.py::pilot_27b --n-base 24
```

Then generate all available summaries and plots:

```bash
uv run python scripts/analyze_results.py --behavior-file behavior_27b.jsonl
```

Modal outputs are written to a persistent volume and can be downloaded with the command printed by each run. The raw result schema and figure definitions are in [the analysis plan](docs/analysis-plan.md).

## Planned figures

1. Four-cell behavioral validation with scenario-bootstrap intervals.
2. Fact, policy, and answer J-lens trajectories across normalized depth.
3. Representative layer-by-position token/rank maps.
4. Causal layer sweep against random and direct-answer controls.
5. Dose-response curves with an activation-distance overlay.
6. The critical two-by-two sign-flip matrix.
7. Reporting change versus independently measured fact-readout change.
8. Discovery-to-held-out-family transfer.

The plotting code emits figures only when their required raw table exists. It never fills missing experiments with synthetic values.

## Interpretation and novelty

Activation probes and steering directions for truthfulness or deception already exist, and the Jacobian Lens paper already shows causal concept swaps and alignment-audit examples. The potentially novel contribution here is therefore **not** “a deception vector.” It is the combined causal test of a *reporting policy* using a vocabulary-grounded J-space intervention, factorial world-state controls, a required literal-output sign flip, and quantitative preservation of an independently measured fact variable.

That is an interesting result if it transfers to held-out task families and survives the controls. If it only works on A/B prompts, explicit policy wording, or answer-token directions, it is a useful but limited negative/diagnostic result.

## Primary references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/) and the [reference implementation](https://github.com/anthropics/jacobian-lens).
- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Burns et al., [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).
- Levinstein and Herrmann, [Still No Lie Detector for Language Models](https://arxiv.org/abs/2307.00175).
- Wang, Zhang, and Sun, [When Thinking LLMs Lie](https://arxiv.org/abs/2506.04909).
