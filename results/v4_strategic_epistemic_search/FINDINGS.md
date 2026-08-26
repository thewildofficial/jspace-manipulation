# V4-SES-1 findings: correct rationales can impair later relational report

## Bottom line

V4 established a reproducible behavioral phenomenon in one pinned model:
retaining Qwen3.6-27B's own correct decision rationale makes a later forced-choice
report about the selected action's most likely response worse than retaining only
the final action. The effect survived a held-out confirmation and a fresh,
prospectively counterbalanced 360-decision study.

The current evidence does **not** establish a binding-ID collision, privileged
introspection, a J-space mechanism, deception, scheming, or a uniquely
self-generated interference effect. The strongest new lead is an exploratory
interaction: the rationale-induced report distortion was substantially larger in
sender/receiver language than in an isomorphic controller/device frame. That
interaction was noticed after inspecting the prospective result and requires a
fresh confirmation.

## Discovery and held-out confirmation

Qwen3.6-27B solved the original expected-payoff decisions at 97.5% accuracy
(234/240). After a correct decision, it was asked which response had the highest
probability under its selected signal. The report used disjoint, randomized X/Y/Z
labels.

| Stage | Frozen population | Rationale | Answer only | Paired discordance | Exact p |
|---|---:|---:|---:|---:|---:|
| Discovery | 78 correct decisions | 70/78 (89.7%) | 77/78 (98.7%) | 0 vs 7 | .015625 |
| Held-out confirmation | 54 correct C-decisions | 47/54 (87.0%) | 53/54 (98.1%) | 0 vs 6 | .03125 |

The discovery contrast was one of four preregistered predicted-response controls;
its Holm-adjusted p was .0625. Every discovery discordance occurred after selecting
signal C, so the confirmation protocol froze that stratum and used only unseen
validation and locked splits. The held-out result passed its preregistered
two-sided exact McNemar test.

Manual inspection found that the key held-out rationales contained correct
arithmetic and explicitly stated the correct high-probability response. This rules
out a simple "the rationale never computed the fact" explanation. It does not show
that every hidden reasoning step was correct.

## Alias experiment and corrected interpretation

The post-confirmation 2x2 alias experiment produced:

| Report condition | Indexed R1/R2/R3 | Arbitrary aliases |
|---|---:|---:|
| Answer only | 87/88 (98.9%) | 85/88 (96.6%) |
| Full rationale | 75/88 (85.2%) | 69/88 (78.4%) |

Aliases did not rescue the rationale condition. However, this experiment renamed
only the report outputs. The original decision prompt and rationale still used
A/B/C and R1/R2/R3. A cross-role ordinal association formed during the decision
could therefore survive the report-only aliasing. The earlier statement that this
experiment localized the effect away from a C-to-R3 collision was too strong.

The pooled alias statistics also reuse discovery and confirmation examples and
remain descriptive. In discovery, a matched unrelated trajectory was worse than
the model's own rationale (85.9% versus 89.7%). V4 therefore has not shown that
self-generated text is uniquely harmful; additional relational text is sufficient
to produce interference.

## Prospective ordinal permutation

V4-SES-2 independently counterbalanced action labels, action presentation order,
response labels, and response presentation order with an OA(9,4,3,2). Twenty new
base games, nine variants, and two matched frames yielded 360 decisions.

The behavior gate passed:

- parseability: 360/360;
- decision accuracy: 345/360 (95.83%);
- minimum preregistered cell accuracy: 83.33%; and
- no token-ceiling hits.

On the 345 correct decisions, report accuracy was 332/345 (96.23%) with the
rationale and 342/345 (99.13%) answer-only. There were 11 answer-only-only
correct pairs, one rationale-only correct pair, 331 both-correct pairs, and two
both-wrong pairs. The unclustered secondary exact McNemar p was .00635.

Both preregistered lure-vs-true logit tests were positive:

| Frozen primary test | Eligible | Cluster mean pull | Raw exact p | Holm p |
|---|---:|---:|---:|---:|
| Action-label lure | 229 | +1.2177 | 1.91e-6 | 3.81e-6 |
| Action-position lure | 229 | +1.2190 | 1.91e-6 | 3.81e-6 |

Under the frozen interpretation table, support for both tests maps to a "mixed
binding code." That label is not mechanistically secure because both tests compare
their designated lure with the true response. A generic rationale-induced
suppression of the true response makes both tests positive even when neither lure
is selectively targeted.

## Post-hoc specificity diagnostics

After observing the nearly identical primary effects, a separate analysis compared
each designated lure with the *other false response*. These checks are explicitly
post-hoc and cannot confirm a new mechanism.

| Exploratory contrast | Cluster mean | Raw exact p | Exploratory Holm p |
|---|---:|---:|---:|
| Label lure minus nondesignated false response | +0.1766 | .0954 | .1908 |
| Position lure minus nondesignated false response | +0.1459 | .2902 | .2902 |

The generic average pull of both false responses relative to the truth was +1.1097
logits and had the minimum attainable 20-cluster two-sided sign-flip p of 1.91e-6.
The current result therefore supports broad true-response suppression or relational
interference, not a confirmed label or ordering-pointer collision.

The most interesting exploratory interaction was frame-specific:

- nine of the 11 answer-only-only errors occurred in the strategic frame;
- the strategic-minus-nonagentic generic false-vs-true logit interaction was
  +0.6970, with 19/20 game clusters positive and exact p=3.81e-6; and
- the corresponding accuracy-penalty interaction was -4.44 percentage points,
  exact p=.0547.

The numeric games and counterbalancing were matched across frames, but their
wording necessarily differed. This could reflect social/strategic semantics,
ordinary lexical differences, or both. It is a fresh hypothesis, not a finding to
generalize.

## Protocol deviation

The `ordinal_behavior` entrypoint called a shared `behavior_remote` implementation
that also generated the 720 preregistered report rows in the same Modal invocation.
The named `ordinal-report` workflow step was skipped, but its outcomes were already
inside the immutable behavior artifact before inspection. The behavior gate passed,
the report configuration was frozen, and no result-dependent prompt selection
occurred, so the embedded report is retained as the prospective outcome.

The stage-separation bug is fixed for future runs. The raw behavior artifact is
preserved unchanged, and the report file is a lossless, hash-linked extraction.
This deviation is also recorded in `docs/v4/decision-log.md`.

## Supported and unsupported claims

Supported:

> In one pinned model, retaining a correct decision rationale causally reduced
> later forced-choice access to the correct predicted response relative to an
> answer-only context. On fresh counterbalanced games, the main logit effect was a
> broad shift from the true response toward false alternatives.

Not supported:

- a label-binding or position-binding mechanism;
- truth preserved outside J-space while a false report occupies J-space;
- causal J-space mediation or a repair intervention;
- uniquely self-generated interference;
- genuine receiver-belief inference, because the receiver policy was printed;
- deception, scheming, intent, or introspective access; or
- cross-model generality.

## Current decision

Do not launch a broad mechanistic sweep from the two preregistered lure p-values.
First run a cheap, fresh confirmation of the strategic-frame interaction with
minimal lexical controls and a specificity-aware primary outcome. If that fails,
archive V4 as a clean rationale-interference result. If it passes, test the stronger
causal hypothesis: the correct relation remains in residual state while strategic
rationale context changes what enters the reportable workspace, and a narrow
answer-only-to-rationale patch repairs the report.
