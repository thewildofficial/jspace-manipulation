# Literature comparison and novelty boundary

This is a targeted primary-source review, not a systematic proof of priority.
Full-text methods/results were inspected for the workspace paper, strategic
belief/action paper, semantic-conflict patching paper, Monitoring Monitorability,
latent-CoT monitorability, probe trajectories and the original deception probes.
Abstract-only entries are marked. Sources were retrieved in September 2026.
The question is what a new experiment would add, rather than how many neighboring
papers can be cited.

## Closest work and its consequences for this project

| Primary source | Existing contribution | Consequence for our design |
|---|---|---|
| [Gurnee et al., Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/) (2026), Methods; Limitations | Averaged Jacobian readout and workspace-related interventions; explicitly identifies limitations of a bag of concepts | Distinguish token ranking from sparse J-space; test relational information rather than naming a thought |
| [Why Do LLMs Struggle in Strategic Play?](https://arxiv.org/html/2605.00226v1) (2026), internal probes and interventions | Internal beliefs can exceed verbal reports and weakly control actions; positional biases | A generic belief/action gap is not novel; isolate an experimentally manipulated action mapping and report intervention |
| [A Mechanistic Lens on Semantic Conflicts](https://arxiv.org/html/2607.05587v1) (2026), Methods and Results | Token-aligned, bidirectional patching for meaningful cue versus executable-code conflict | A semantic-conflict patching heatmap alone adds little; require report reactivity or safety-relevant transfer |
| [Monitoring Monitorability](https://arxiv.org/html/2512.18311v1) (2025), Section 7 | Follow-up CoT can improve monitoring of completed behavior; some follow-ups degrade it | 'Ask another question' is occupied; separate post-action diagnosis from pre-action behavior modification |
| [Does Out-of-Sight Equal Out-of-Mind in CoT Monitorability?](https://arxiv.org/html/2608.04928v1) (2026), Sections 3–5 | Compares explicit/latent reasoning and internal/text access; internal probes often help; transfer is task-dependent | Comparing a probe with text is occupied; require matched information budgets and held-out mappings |
| [Monitoring the Internal Monologue](https://arxiv.org/html/2605.18549v1) (2026), Methods and Evaluation | Probe trajectories can predict future behavior better than static scores | 'We trace a probe through reasoning' is occupied; test counterfactual relational changes and causal specificity |
| [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/html/2502.03407v1) (2025), Methods and limitations | Strong detection on selected deception datasets, but no robust-defense claim | A high in-distribution AUROC is insufficient; measure hard negatives, transfer and added value over external observations |
| [Did you lie?](https://arxiv.org/abs/2606.12618) (2026), abstract reviewed | Detectors degrade on trained versus prompted lying; belief verification can privilege readable CoT | Do not label a behavior deceptive just because it contradicts a neutral-prompt answer; report selection bias |
| [Chain-of-Thought Monitoring Can Be Unreliable in Implicit-Influence Settings](https://arxiv.org/abs/2608.04735) (2026), abstract reviewed; HTML retrieval failed | Implicit influences are harder to detect than explicit concealment instructions | Include semantic influence without 'lie/hide' language; a conceal-instruction classifier is a weak baseline |
| [Evaluating and Understanding Scheming Propensity](https://arxiv.org/html/2603.01608v1) (2026), Section 4.3 | Scheming varies with prompt/scaffold and tool access | A renamed-tool result must hold available transitions constant; tool availability is a different intervention |
| [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191) (2023), abstract reviewed | Binding representations and interventions linking entities and attributes | 'A binding representation exists' is occupied; test its relationship to report and outcome control |
| [Patchscopes](https://research.google/blog/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/) (2024), authors' technical account | Uses a model to decode patched hidden representations into text | Optional relational decoder baseline; generated explanations remain a measurement instrument needing validation |

## Three distinct scientific quantities

**Availability:** can a frozen readout recover a specified variable from a state?
A residual probe, candidate-token J-lens readout, and Patchscope are different
instruments. A full-rank Jacobian transform followed by a learned probe need not
be an information bottleneck; that comparison cannot automatically establish
that formal J-space contains or loses information.

**Use:** changing the relevant state changes the model's action in the predicted
counterfactual direction. Decodability does not imply use, and a harmful broad edit
does not demonstrate a selective mechanism. A positive interchange is evidence
about that contrast/site, not a complete computational abstraction.

**Disclosure:** a query causes the model to output information. The query can
recompute the answer and alter subsequent behavior. Accurate disclosure under a
new prefix does not establish that the same information controlled the original
action. Post-action detection is valuable but cannot retroactively change that
action; pre-action elicitation can be both measurement and intervention.

Our current results distinguish some of these quantities behaviorally but do not
establish a general internal separation or deliberate concealment.

## Strongest research opportunity

The candidate contribution is a controlled relation-level study of **elicitation
reactivity**:

1. Reproduce action failure and report-conditioned repair on fresh games.
2. Hold final instructions fixed to identify content versus instruction effects.
3. Compare passive pre-query measurements with branched report readouts.
4. Change action–outcome binding while balancing individual words and labels.
5. Test a narrowly frozen intervention on the relation and held-out formats.

This conjunction appears more distinctive than a new token atlas. The review did
not find an identical experiment; that is a bounded search result, not a 'first'
claim. Further citation/author searches are required before claiming priority.

The strongest result would be: a relation recoverable before a report becomes
behaviorally influential after elicitation, and a selective intervention explains
part of that change. A useful negative result would be: only a decoder that can
recompute from context succeeds, so this lens cannot support the proposed passive
monitor under those conditions.

## Safety translation and limits

Renamed toy actions can expose a construct-validity problem in a safety proxy.
They cannot establish that real scheming evaluations are invalid. The tool task
must execute the same state transition under every alias and score the resulting
state, not the function's moral-sounding name. Report errors, action errors,
reward-maximizing deception and unauthorized goal pursuit are distinct labels.

The ambitious downstream question is whether an audience/oversight-dependent
report changes while a causally used belief remains stable. It requires competent
strategic behavior and outcome-grounded labels first. With $30, a new frontier
scheming benchmark or full model-organism training campaign is a poor first bet.
A focused mechanistic precursor can still be alignment-relevant.

## Corrections to the prior positioning

The existing V5 conjunction remains a reasonable behavioral novelty candidate.
However, 'more evidence strengthens a misleading semantic script' is a proposed
mechanism, not an established result. Independent consequence reports do not
certify an unchanged internal knowledge state on the action trajectory. RBG-6
supports a report-conditioned rescue, not transparent access to that trajectory.
RBG-5B does not provide a successful causal-localization component, and its mixed
normalized/unnormalized readout requires correction before a lens conclusion.

## Existing V5 reference recheck

The title/abstract mappings for Code over Words (2601.18352),
Formal Verification of Agentic Systems over Operational Data (2608.03609), and
ToolRobustBench (2608.23635) were rechecked against arXiv. They support semantic
interference, identifier equivariance and stage-wise tool failure diagnosis,
respectively. Identifier renaming in operational data is not identical to tool
function renaming; our experiment must establish its own effect.

The primary Anthropic [summer agentic-misalignment report](https://alignment.anthropic.com/2026/agentic-misalignment-summer-2026/)
was also inspected. Its simulation case studies motivate our safety task; they do
not establish a result for Qwen or the proposed invariant tool interface.

Direct retrieval of the existing The Story Shapes the Agent (2607.18566) and
INTENT-AS-A-TOOL (2608.27348) references failed in this pass. Their prior entries
remain historical references, not newly verified support for our novelty claim.
The current recommendation does not depend on those two sources.
