# V5 literature positioning: inverse evidence and semantic action capture

## Closest open literature

**A Mechanistic Lens on Semantic Conflicts: Using Activation Patching to
Understand LLM Behavior** is the closest mechanistic precedent found after
RBG-4. It uses token-aligned triplets, bidirectional residual patching, raw and
normalized output-preference changes, and denominator safeguards to localize
competition between meaningful cues and executable code. RBG-5 adopts those
methodological safeguards but studies redundant correct evidence, independent
consequence reports, and naturally successful table/opaque donors in a strategic
action task.

- https://arxiv.org/abs/2607.05587

**Towards Best Practices of Activation Patching in Language Models** and **How
to use and interpret activation patching** show that localization depends on the
contrast, corruption, metric, and patch granularity. RBG-5 therefore freezes its
raw logit-margin endpoint, source/destination pairing, semantic anchors, reverse
patches, and denominator handling before locked execution.

- https://arxiv.org/abs/2309.16042
- https://arxiv.org/abs/2404.15255

**Causal Abstractions of Neural Networks** motivates using interchange
interventions to test whether a neural state has the counterfactual role of a
high-level variable. RBG-5 makes the narrower sufficiency claim licensed by a
single natural residual replacement rather than claiming a full abstraction.

- https://arxiv.org/abs/2106.02997

**Function Vectors in Large Language Models** provides a candidate account on
which demonstrations write a compact task state into middle-layer residual
processing. This motivates—but does not establish—the persistent-policy-state
hypothesis for RBG-5.

- https://arxiv.org/abs/2310.15213

Probe methodology supplies an important negative constraint. **Designing and
Interpreting Probes with Control Tasks** and **Amnesic Probing** distinguish
decodable information from information the model behaviorally uses. RBG-5 uses
group-held-out and shuffled-label controls and assigns all causal weight to the
locked interchange rather than probe accuracy.

- https://aclanthology.org/D19-1275/
- https://aclanthology.org/2021.tacl-1.10/

**Broken Links Between Observations, Beliefs, and Actions** reports that internal
game-state beliefs can exceed verbal reports and that beliefs often weakly control
best-response actions. It establishes the broad strategic belief/action gap and
first-item bias. RBG-4 instead manipulates redundant correct evidence and
representation format while holding the causal game fixed, and finds action harm
with unchanged consequence reports.

- https://arxiv.org/abs/2605.00226

**Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning** shows
that natural-language semantic priors can override counterfactual in-context rules
in `Baba Is You`, and that code grounding mitigates the effect. RBG-4 is consistent
with semantic inertia but isolates an assertion-based strategic decision and an
inverse-evidence effect: four additional correct demonstrations worsen action, and
a table rather than executable code removes the harm.

- https://arxiv.org/abs/2601.18352

**The Story Shapes the Agent: Narrative Priors in LLM Behavior** holds decision
structure fixed and shows that narrative framing drives action patterns, sometimes
against task success. RBG-4 similarly uses structural isomorphs, but additionally
measures both option consequences and intervenes on redundant evidence and policy
format within the same base games.

- https://arxiv.org/abs/2607.18566

**Larger Language Models Do In-Context Learning Differently** studies flipped and
semantically unrelated labels, showing competition between semantic priors and
in-context mappings. RBG-4 extends this tension from classification labels to
causal strategic action under false-belief goals.

- https://arxiv.org/abs/2303.03846

## Distinctive empirical conjunction

The literature reviewed so far contains each neighboring idea—semantic inertia,
narrative action priors, in-context label conflict, and strategic belief/action
gaps—but we have not found a paper reporting this exact conjunction:

1. structurally matched strategic actions with explicit consequences;
2. independent reports showing that both consequences remain available;
3. a randomized addition of correct, redundant evidence that worsens action;
4. semantic specificity relative to opaque actions; and
5. elimination of the harm by tabularizing the same causal rule.

This is a defensible novelty target, not proof of priority. A broader systematic
search and author-level comparison are required before a paper claims “first.”

## Contribution framing

The strongest current framing is not “models secretly know the answer” or “models
scheme.” It is:

> Strategic failures can be caused by how correct causal knowledge is bound to
> meaningful actions. More evidence can strengthen a misleading semantic action
> script without degrading explicit consequence knowledge, while a representation
> change restores policy use.

This links cognitive-control/Stroop-style interference to alignment evaluation:
evaluators may add demonstrations expecting greater task understanding, yet make
meaningful strategic action less reliable even as verbal checks remain excellent.
