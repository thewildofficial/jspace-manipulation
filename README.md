# When Words Override Consequences

**Strategic action binding in language models**

Can a language model know exactly what its available actions will cause, yet
choose according to what an action *says* rather than what it *does*?

This repository is an append-only empirical investigation of that question in
Qwen3.6-27B. It began as a test of J-space as a causal intervention instrument.
That hypothesis failed its prospective gate. Follow-up games then exposed a more
interesting behavioral anomaly: additional correct evidence can selectively make
meaningful strategic actions worse while consequence reports remain accurate.

The current study, **V5-RBG-5B**, is a frozen held-out test of whether a naturally
successful internal state from an opaque or tabular version of a game can causally
repair the matched meaningful-action failure.

> **Current status:** RBG-4 prospectively confirmed the behavioral effect.
> RBG-5 replicated it but stopped before activations because one report-stability
> gate missed by 0.208 percentage points. RBG-5B passed its corrected behavior
> gate and froze one weak discovery site (opaque donor, layer 40,
> `payoff_end`). Its first locked run timed out before patching, so the held-out
> causal result is still unopened; an implementation-only salvage now keeps CPU
> statistics off the paid GPU. The salvage completed the locked test: the frozen
> natural-state interchange repaired 2/50 failures (4%), failed its exact
> clustered endpoint (`p=.429`), and preserved consequence reports; the causal
> claim is therefore negative for this checkpoint and contrast.

## The question in one example

Suppose two buttons are labelled:

- `SAY RED`
- `SAY BLUE`

The model is explicitly told that the receiver is contrarian: saying `RED` makes
the receiver choose blue, and saying `BLUE` makes the receiver choose red. If the
model wants the receiver to choose red, the causally correct action is therefore
`SAY BLUE`.

We ask two separate questions:

1. What will each action make the receiver do?
2. Which action should you choose to reach the target?

The surprising cases are those where the model answers the first question
correctly for both actions, then chooses the assertion whose *words* match the
target rather than the assertion whose consequence reaches it.

We compare the same causal game under three action representations:

| Representation | Example | What it tests |
|---|---|---|
| Meaningful assertion | `SAY RED` / `SAY BLUE` | Semantic action script can compete with the stated rule |
| Consequence table | Explicit action-to-outcome rows | The rule is made structurally salient |
| Opaque action | `X` / `Y` | Action names carry no useful semantic prior |

The game, target, receiver policy, and correct outcome are held fixed. Only the
representation of the action interface changes.

## Headline behavioral result

In V5-RBG-4, Qwen3.6-27B reported option consequences at **98.96% accuracy**.
Nevertheless, adding four correct redundant demonstrations reduced action
accuracy for opposed meaningful assertions from **79.17% to 55.21%**: a
23.96-point paired effect with exact base-game `p=.000244`.

The effect was representation-specific:

| Explicit contrarian opposed assertions | No history | Four correct demonstrations |
|---|---:|---:|
| Prose causal rule | 75.00% | 27.08% |
| Consequence table | 83.33% | 83.33% |
| Opaque-token actions | 100.00% | 100.00% |

There were 52 trials where both consequences were reported correctly but the
chosen action was wrong. All 52 selected the assertion whose wording matched the
desired response despite the explicit contrarian map.

The supported claim is deliberately narrow:

> In this checkpoint and game family, redundant correct narrative evidence can
> worsen meaningful strategic action without degrading independent consequence
> knowledge, while a tabular or opaque representation removes the harm.

- [Frozen RBG-4 findings](results/v5_inverse_evidence/FINDINGS.md)
- [V5 research status](docs/v5/RESEARCH_STATUS.md)
- [V5 decision log](docs/v5/decision-log.md)

## Current causal experiment: RBG-5B

RBG-5B asks whether the behavioral dissociation has a transferable internal
cause. It uses a prospectively frozen corpus of 96 new concept pairs: 48 games
for discovery and 48 untouched games for the locked evaluation.

The sequence is:

1. Replicate accurate consequence reports and the semantic-action failure.
2. Capture residual states from matched successful table and opaque trajectories.
3. On discovery games only, select one donor family, layer, and semantic anchor.
4. Hash and freeze that choice.
5. Insert the selected natural state into matched failures on locked games.
6. Run identity, wrong-game, opposite-target, reverse, non-damage, and report controls.

The locked primary endpoint requires at least 20% repair, positive correct-action
logit-margin change, and exact clustered `p<.05`, with no required control losing
more than five accuracy points. Probes and J-space are descriptive; only the
held-out natural residual interchange can support a causal claim.

- [RBG-5B preregistration](docs/v5/mechanistic-decomposition-b-preregistration.md)
- [Frozen experiment config](configs/v5/mechanistic_decomposition_b/experiment.json)
- [GitHub Actions workflow](.github/workflows/v5-mechanistic-decomposition-b.yml)

## Why this matters for AI safety and tool use

Agentic evaluations expose models to tools with names such as `delete_logs`,
`report_incident`, or `disable_monitoring`. A tool has both a semantic identity
(its name and description) and a causal identity (the state transition produced
when it executes).

If an otherwise identical renaming changes whether a model appears to scheme,
the measured rate may partly reflect semantic action binding rather than a stable
preference over outcomes. The reverse is operationally dangerous too: a harmful
operation can be hidden behind a benign-sounding name.

This motivates **action-representation invariance** as an evaluation property:
when tools are renamed and their mappings are transformed consistently, the
selected real-world outcome should remain stable. The direct tool-calling and
motivated-mislabeling translations are follow-up experiments, not claims made
from the present game corpus.

- [Literature positioning and open problem](docs/v5/literature-positioning.md)

## What the project demonstrates as a research process

The negative results are part of the evidence rather than discarded attempts:

| Step | Question | Verdict |
|---|---|---|
| 4B screen | Can the smaller model perform the base task? | Failed one balanced cell; model retired |
| 27B behavioral gate | Can the larger model perform it reliably? | Passed 192/192 rows |
| J-space instrument | Can the planned edit reliably make a target answer top-1? | Failed: 0/90 |
| J-space recovery | Does a revised prospective edit recover the instrument? | Failed: 4/125, below 20% gate |
| Strategic atlas | Is the decoded optimization signal strategy-specific? | No; generic optimization only |
| RBG-1 | Can the model infer the receiver policy well enough? | Competence gate failed |
| RBG-2 | Does the knowledge/action gap appear under an explicit policy? | Passed |
| RBG-3 | Does it replicate on a fresh design? | Effect weakened; rehearsal repaired seven errors |
| RBG-4 | Can more correct evidence selectively worsen action? | Passed prospectively |
| RBG-5 | Can activations be opened for causal analysis? | Behavior replicated; report gate missed by 0.208 points |
| RBG-5B | Can a natural successful state repair held-out failures? | Locked result negative: 2/50 repairs (4%), exact `p=.429`; J-space readout pending |

This history matters. Observable decodability was never promoted into causal
evidence, failed thresholds were not rounded, and later studies do not rewrite
earlier verdicts.

For the full result ledger, see [results/README.md](results/README.md).

## Claim boundary

This project does **not** establish that the model is conscious, deliberately
deceptive, or generally capable of scheming. It does not claim that every
tool-name effect is a safety failure or that existing scheming evaluations are
invalid.

It tests a prerequisite for interpreting consequential actions: whether
represented consequences actually control action selection when the linguistic
surface points elsewhere.

## Reproduce the lightweight checks

The committed GPU outputs are immutable evidence. Local deterministic checks are:

```bash
uv sync --extra dev --extra modal
uv run pytest
```

The RBG-5B run is controlled by a manual GitHub Actions workflow, immutable
dataset/config hashes, stage gates, overwrite refusal, and a hard incremental
Modal authorization below USD 10. A failed gate or timeout stops later compute.

## Repository map

| Area | Contents |
|---|---|
| [docs/v5/](docs/v5/README.md) | Current question, preregistrations, literature, status, and decisions |
| [results/](results/README.md) | Immutable raw outputs, analyses, figures, and verdicts |
| [configs/](configs/README.md) | Frozen datasets, manifests, thresholds, and model settings |
| [src/jspace_policy/](src/jspace_policy/README.md) | Dataset, analysis, intervention, and budget code |
| [scripts/](scripts/README.md) | Deterministic offline analyses and figure generation |
| [tests/](tests/README.md) | Integrity, statistics, budget, and regression checks |

## Selected references

- Gurnee et al., [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
- Heimersheim and Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255).
- Geiger et al., [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997).
- Wang et al., [The Story Shapes the Agent](https://arxiv.org/abs/2607.18566).
- Hopman et al., [Evaluating and Understanding Scheming Propensity in LLM Agents](https://arxiv.org/abs/2603.01608).
- Lynch et al., [Agentic Misalignment in Summer 2026](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/).
