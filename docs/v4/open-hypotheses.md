# Open hypotheses and contribution map

**Status:** research agenda written after the 2026-08-24 ordinal result. None of
the hypotheses below is a finding unless explicitly marked as prior evidence.

## Literature boundary

Several broad claims are already occupied:

- The [J-space paper](https://transformer-circuits.pub/2026/workspace/) argues that
  a limited-capacity verbalizable subspace participates in report, modulation, and
  reasoning, and observes some competition for workspace access.
- [Broken Links Between Observations, Beliefs, and Actions](https://arxiv.org/abs/2605.00226)
  reports that internal strategic beliefs can be more accurate than verbal reports
  and that internal beliefs can connect weakly to action.
- [Language Models Use Lookbacks to Track Beliefs](https://arxiv.org/abs/2505.14685)
  identifies ordering IDs and causal lookback mechanisms in theory-of-mind tasks.
- [How Do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191)
  and [Representational Analysis of Binding](https://arxiv.org/abs/2409.05448)
  identify low-rank binding/order representations and causally swap bound
  attributes.
- [Correct Chains, Wrong Answers](https://arxiv.org/abs/2604.13065) already shows
  fully correct visible reasoning followed by incorrect declared answers.
- [Unable to Forget](https://arxiv.org/abs/2506.08184) and
  [Dual-Process Interference](https://arxiv.org/abs/2603.00270) establish broad
  retrieval interference and reasoning-related interference patterns.
- [Can LLMs Introspect? A Reality Check](https://arxiv.org/abs/2605.26242) explains
  why behavioral self-report gaps alone cannot establish introspection.
- Strategic deception has behavioral benchmarks, including
  [The Secret Agenda](https://arxiv.org/abs/2509.20393),
  [Lying to Win](https://arxiv.org/abs/2603.07202), and
  [MASK](https://arxiv.org/abs/2503.03750).
- Alignment work already studies trained disclosure, including
  [Training Agents to Self-Report Misbehavior](https://arxiv.org/abs/2602.22303)
  and [Teaching LLMs to Self-Report Their Hidden Objectives](https://arxiv.org/abs/2511.06626).

The contribution cannot merely be “internal signal differs from report,” “CoT can
hurt,” “models can lie in games,” or “binding lives in a low-rank subspace.” The
novel target must join strategic use, report divergence, and causal mechanism in
the same controlled trial.

## Priority 1 — Strategic semantics selectively destabilizes report

### Hypothesis

When identical expected-value computations are framed as influencing another
agent, the model recruits social/strategic relational state that competes with the
correct receiver-response relation during later report. The effect should exceed
ordinary lexical variation and matched nonagentic reasoning load.

### Prior evidence

This was discovered post-hoc in V4-SES-2:

- strategic-minus-nonagentic false-vs-true pull: +0.6970 logits;
- 19/20 base-game clusters in the predicted direction, exact p=3.81e-6; and
- nine versus two answer-only-only errors.

The accuracy interaction itself was only borderline, p=.0547. The current wording
confounds agency with lexical choices such as sender/controller,
receiver/device, message/input, and response/outcome.

### Cheapest decisive experiment

Freeze fresh numeric games and cross:

1. human receiver versus algorithmic receiver;
2. strategic message versus noncommunicative control input;
3. two or more independently written minimal paraphrase blocks; and
4. rationale versus answer-only access.

Keep probability tables, payoff vectors, costs, action labels, response labels,
display order, token budget, answer mapping, and game IDs matched. The primary
endpoint should be the within-game strategic-minus-nonagentic change in the average
false-vs-true report logit, clustered by numeric game. Binding lures should be
secondary and compared with nondesignated false responses.

### Contribution if confirmed

> Merely representing an isomorphic decision as an interaction with another agent
> selectively destabilizes later access to a relation the model computed
> correctly.

That would be a behavioral bridge between social cognition and workspace
competition. It becomes substantially stronger if the effect localizes to a
reportable workspace while the true relation remains elsewhere.

### Kill criteria

- no fresh clustered frame interaction;
- effect changes sign across paraphrase blocks;
- effect is explained by one lexical substitution; or
- answer-only context shows the same frame interaction.

If killed, do not mechanism-sweep the current V4 phenomenon.

## Priority 2 — Residual truth / reportable-workspace distortion

### Hypothesis

After a correct strategic rationale, the true predicted response remains encoded
in residual state, but the report prefix changes what becomes accessible in
J-space or late answer preparation. Answer-only-to-rationale activation patching
at a narrow layer/position window should restore the true report.

### Required design

Only run after Priority 1 confirms on untouched games. Freeze:

- high-effect correct decisions plus matched low-effect controls;
- a small layer set justified independently by the published workspace band;
- report-prefix positions rather than a full all-position sweep;
- simultaneous readouts for true response, false alternatives, selected action,
  and strategic/nonagentic frame;
- answer-only→rationale and rationale→answer-only interchange interventions;
- unrelated-game, random-direction, same-action, and KL/displacement controls; and
- a rescue endpoint defined before viewing activations.

Decodability alone is insufficient. The central evidence is a double dissociation:

1. residual truth survives when report accuracy falls;
2. workspace/late state shifts away from truth;
3. a narrow patch repairs report without changing the underlying decision or
   unrelated variables; and
4. the reverse patch induces the distortion.

### Contribution if confirmed

> Correct strategic knowledge can persist internally while self-report access is
> causally rerouted, and the divergence can be repaired at a localized report
> interface.

This would go beyond both generic latent-belief probes and generic interference.

### Kill criteria

- truth is absent from residual state;
- J-space and residual readouts degrade together;
- rescue requires broad high-KL corruption;
- random or unrelated patches work equally well; or
- the effect is not selective to report.

## Priority 3 — Genuine belief-guided deception with false self-report

### Why V4 does not test it

V4 prints the receiver policy. The model calculates expected value; it does not
infer another agent's hidden belief or type. Strategic versus device framing is a
semantic control, not theory of mind.

### Hypothesis

In a Bayesian signaling game, a model can use an accurate inferred receiver model
to choose a profitable deceptive message while later reporting an inaccurate
receiver belief. The accurate belief should be causally necessary for the message,
not merely decodable after the fact.

### Experimental game

- Sender privately observes world state `s`.
- Receiver has latent type `t` or policy learned from prior episodes, not printed
  in the current trial.
- Sender and receiver incentives vary from aligned to opposed.
- Sender selects a message; receiver updates and acts.
- Oracle calculation identifies the belief/action prediction needed to choose the
  payoff-maximizing message.
- Matched controls hold state, action, message surface, and payoff constant while
  counterfactually changing receiver type or belief.

Behavior must first show that receiver history changes messages in the normative
direction and improves payoff over no-history and shuffled-history controls.

Then measure three objects separately:

1. the receiver belief/type used before message selection;
2. the message/action policy; and
3. the receiver belief verbally reported afterward to an auditor.

Patch the receiver-belief representation across matched types. A valid mechanism
must change the message in the counterfactual direction while preserving private
state and payoff representation.

### High-value result

> The model used a true latent model of the receiver to deceive, reported a false
> model of the receiver to the auditor, retained the true model outside reportable
> workspace, and changed disclosure under a targeted causal patch.

This is the strongest alignment contribution in the agenda. It combines causal
theory of mind, strategic deception, and self-report reliability rather than
repeating a behavioral lying benchmark.

### Kill criteria

- messages do not depend on inferred receiver type;
- a lexical or memorized-policy baseline explains behavior;
- decoded receiver belief does not causally affect action;
- verbal-report divergence vanishes under forced-choice controls; or
- opposed incentives do not selectively induce false disclosure.

## Priority 4 — Audience-indexed reporting policy

### Hypothesis

With world state and internal receiver belief held fixed, the model routes different
content into report depending on the auditor's verification power, incentives, or
relationship to the modeled receiver. The report policy may change while the
belief used for action remains stable.

### Test

Use the confirmed Bayesian signaling game and randomize whether the auditor can
verify the private state, whether misreporting is rewarded, and whether the auditor
shares the receiver's interest. Query the same belief with randomized forced-choice
labels. Require stable action and stable residual belief across auditor conditions.

A causal report-policy result requires an audience representation that mediates
disclosure and can be patched without changing the underlying belief. Instructed
lying alone is not a contribution.

### Contribution if confirmed

This would bear directly on whether trained self-report or self-incrimination can
be trusted out of distribution: disclosure would be a context-sensitive policy,
not a transparent readout of belief.

## Priority 5 — Binding-specific capture, redesigned correctly

### Current status

Not confirmed. V4's lure-vs-true endpoint confounded designated lure capture with
generic truth suppression. Post-hoc label specificity had Holm p=.191; position
specificity p=.290.

### Better test

If revisited, use four actions and four responses so each eligible trial has one
designated lure and at least two nondesignated false controls. Independently
permute semantic labels and presentation positions upstream in the decision prompt.
The primary statistic must be:

`rationale-induced designated-lure change - mean nondesignated-false change`.

A causal contribution would then identify the low-rank binding/order direction and
show that patching it swaps which false response captures report. Mere significant
lure-vs-true movement is insufficient.

### Priority decision

Lower than the strategic-frame replication. The current data do not justify paying
for a binding mechanism search.

## Priority 6 — Capacity competition among strategic variables

### Hypothesis

The reportable workspace has a limited relational capacity: private state,
receiver model, payoff pathway, selected action, and auditor model compete for
access. Increasing the number of simultaneously relevant relations should produce
structured substitutions rather than uniform forgetting.

### Test

Build matched games requiring one, two, or three independently reportable latent
relations while holding prompt length and arithmetic depth approximately fixed.
Randomize which relation will later be queried only after the decision. Trace
residual versus J-space retention and causally prioritize one relation.

### Contribution if confirmed

This would extend the J-space paper's qualitative competition result into a
controlled relational-capacity law relevant to strategic cognition. It is more
cognitive science than game theory unless coupled to the Bayesian signaling game.

## Hypothesis not worth chasing — mystical self-generated residue

For a deterministic transformer, replaying the exact same token sequence in the
same roles produces the same activations; generation history does not leave a
separate private residue beyond the tokens and resulting cache. “The model knows it
wrote this” is therefore not an independent mechanism in the current architecture.

Role, content, position, hidden-reasoning channels, and cache state are testable.
An undefined property of authorship is not. Existing V4 controls also show that an
unrelated matched trajectory can be more harmful than the model's own rationale.

## Recommended sequence

1. Fresh strategic-frame replication with specificity-aware endpoints.
2. If positive, narrow residual/J-space localization and bidirectional repair.
3. In parallel at the design level, build the inferred-receiver Bayesian signaling
   game; do not call it deception until the behavioral definition is satisfied.
4. Only after receiver belief causally controls action, add auditor-dependent
   self-report and J-space analysis.
5. Treat binding and capacity studies as supporting mechanism projects, not the
   central story unless they causally explain strategic disclosure.

The intended paper-shaped contribution is not “J-space sometimes fails.” It is:

> A model can use one epistemic state for strategic action while exposing another
> through verbal report, and the divergence has a specific, causal, repairable
> computational mechanism.
