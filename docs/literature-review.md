# Literature review and novelty boundary

## Bottom line

The toy experiment is not, by itself, evidence of deception. Its useful contribution is narrower: a causal assay for whether a model can keep a fact available while a separable reporting function changes what it says. If that assay generalizes through controlled composition and then prospectively predicts an instrumentally chosen report in a naturalistic task, it becomes relevant evidence about deceptive or unaligned behavior.

The most defensible potentially novel result is:

> A vocabulary-grounded J-space reporting-policy intervention changes the mapping from latent fact to report, produces opposite literal effects for opposite facts, preserves an independently frozen fact readout, and transfers across policy wording, output symbols, and held-out task families.

That conjunction matters. Any one component already has close precedents.

## What prior work already establishes

| Prior result | What it means for this project |
|---|---|
| The Jacobian Lens produces token-indexed causal read/write directions; the J-space paper already demonstrates concept swaps, redirection of intermediate reasoning, alignment-audit readouts, evaluation-awareness ablations, and signatures in trained model organisms. | “J-space can reveal or steer deception-related concepts” is not novel. We must test a more specific computation and stronger controls. |
| Representation engineering and truth-probe work finds directions or classifiers associated with honesty, truth, and deception. | A linear probe or a generic “deception vector” is not enough. |
| Truth probes can disagree with model outputs, but calibration, task heterogeneity, and poor generalization can mimic hidden knowledge or lying. | We need paired causal intervention, held-out transfer, and an independently specified fact measure—not probe/output disagreement alone. |
| Alignment-faking and sleeper-agent studies construct settings with strategically different behavior across contexts. | Those are better target domains for the final naturalistic transfer, but jumping directly to them would make mechanism identification difficult. |

## Claim ladder

### Where the repository currently stands

The original mechanistic ladder is still inside **Research Stage 1**. The 4B
screen, 27B scale-up, released-lens check, observational pilot, and V2
causal-instrument attempt all test the toy mechanism. **Stage 2** and **Stage 3**
are planned extensions, not completed mechanistic experiments.

V4 and V5 later ran behavioral follow-ons about rationale interference and
contrarian games. They are recorded as protocol generations, not as completed
Stage 2 or Stage 3 mechanistic studies, because the V2 causal-instrument gate
that would have licensed that ladder was not passed.

### Stage 1: toy mechanism

Prompt: an explicit `A` or `B` fact, a symmetric COPY/FLIP function, and an exact one-token report.

Observe: normal behavioral competence; J-lens fact and policy trajectories; then whether a policy intervention causes a world-state-conditioned output sign flip while the fact readout remains within a frozen equivalence margin.

Permitted claim: knowledge and an instructed reporting computation are causally dissociable under controlled conditions.

Not permitted: the model intended to deceive, has contradictory goals, or possesses a universal deception representation.

### Stage 2: controlled composition

Prompt: a short story implies the fact; a separately supplied policy determines the report (`story → fact → policy → report`).

Observe: whether the Stage-1 direction and layer band transfer without rediscovery; whether the latent fact emerges before policy application; whether lexical and output-symbol changes preserve the effect.

Permitted claim: the mechanism composes with an inferred world state and is not merely an A/B lookup or prompt echo.

### Stage 3: naturalistic behavior

Prompt: a richer setting in which the model must infer both a latent fact and whether concealing it advances an objective, without an explicit FLIP instruction.

Observe: whether the frozen Stage-1/2 reporting-policy measure prospectively predicts concealment before output; whether causal intervention changes reporting while preserving task knowledge and unrelated capability; whether it generalizes to unseen cue families.

Permitted claim, if controls pass: evidence for a reusable internal reporting-policy mechanism relevant to strategic concealment or unaligned behavior.

Even here, “deceptive intent” requires operational evidence for knowledge, a target whose belief is being manipulated, an objective served by the false report, and context-sensitive selection—not merely a false answer.

## Findings that would be genuinely interesting

1. **Compositional sign flip:** the same conceal intervention pushes literal output toward `B` when fact=`A`, but toward `A` when fact=`B`.
2. **Fact preservation:** behavior moves substantially while two independently frozen fact measurements remain inside an equivalence band.
3. **Policy abstraction:** the effect survives removal of words such as *truth*, *hide*, *copy*, and *flip*.
4. **Symbol transfer:** a direction selected using A/B works when reports use X/Y or semantically meaningful alternatives.
5. **Task transfer:** the frozen direction and layer band work on held-out story families, not merely shuffled instances of one template.
6. **Temporal ordering:** fact information appears before the reporting-policy signal, which then predicts the imminent report.
7. **Naturalistic prospective validity:** the frozen measure predicts or causally changes instrumentally useful concealment without being rediscovered on the naturalistic task.

The strongest paper-worthy outcome would combine 1–5 in the controlled ladder and then show at least preliminary 7. A clean failure is also informative if the behavioral task is competent and the controls show that J-space supports answer steering but not a fact-preserving reporting operation.

## Results that are not novel enough alone

- A J-lens token such as `lie`, `truth`, `secret`, or `conceal` ranks highly.
- A classifier distinguishes reveal from conceal prompts.
- Steering changes the final A/B answer.
- Internal truth scores disagree with an output.
- A direction works only with explicit policy labels or only on discovery templates.
- A false report appears after an instruction to report falsely.

## Experimental upgrades required for soundness

- Freeze the fact readout, candidate concept family, layer band, intervention strength, and equivalence margin before locked testing.
- Keep all conditions for a base scenario in one split and one bootstrap unit.
- Require all fact × policy × wording behavioral cells to pass; never accept aggregate accuracy alone.
- Compare against direct-answer, direct-fact, lexical, unrelated-semantic, and norm-matched random interventions.
- Report activation distance, downstream KL/entropy, and unrelated-task degradation so an off-distribution perturbation cannot masquerade as a mechanism.
- Redo the decisive test with new output symbols and held-out task families.
- In the naturalistic stage, distinguish target-belief manipulation from generic compliance, role-play, uncertainty, and reward-conditioned answer selection.
- Treat all concept search and layer sweeps as discovery. Use only the frozen aggregate estimand for the final claim.

## Primary references

- Gurnee et al. (2026), [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
- Zou et al. (2023), [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405).
- Burns et al. (2022), [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827).
- Levinstein and Herrmann (2023), [Still No Lie Detector for Language Models](https://arxiv.org/abs/2307.00175).
- Liu et al. (2023), [Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness?](https://arxiv.org/abs/2312.03729).
- Hubinger et al. (2024), [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566).
- Greenblatt et al. (2024), [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093).
- Wang, Zhang, and Sun (2025), [When Thinking LLMs Lie](https://arxiv.org/abs/2506.04909).
