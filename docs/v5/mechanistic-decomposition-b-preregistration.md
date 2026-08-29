# V5-RBG-5B preregistration

## Status and question

This is a fresh, prospective follow-up to RBG-5. RBG-5 behavior may inform
engineering and power calculations, but no RBG-5 behavior row, activation,
probe, patch site, or locked result is reused as evidence. The pinned model is
Qwen/Qwen3.6-27B at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

RBG-5B asks whether a naturally occurring residual state from a matched
successful table or opaque-token trajectory selectively repairs the redundant
prose/assertion action failure. Probes and J-space are availability/readout
diagnostics; only held-out natural residual interchange is causal evidence.

## Fresh corpus and behavioral gate

The corpus contains 96 new concept pairs, 48 discovery and 48 locked base
games, two frames, seven focused cells, randomized action and concept order,
two consequence reports per action row, and exact one-token A/B and X/Y
continuations. The expected dataset hash is recorded in the committed manifest.
The five captured cells are the opposed prose assertion with and without history,
opposed redundant table, opposed redundant opaque token, and aligned redundant
prose assertion. Behavior still evaluates all seven cells.

No activation output opens unless all of the following pass: overall reports and
critical prose report cells meet their floors; opaque and aligned actions meet
their floors; opposed prose history harm and assertion-minus-opaque harm are at
least 20 points with exact clustered `p<.05`; table history harm is at most five
points; and both donor families have at least 24 locked and discovery eligible
recipients spanning at least 18 bases.

Report harm is defined as `accuracy(no history) - accuracy(redundant history)`.
The gate is one-sided: its 95% base-cluster bootstrap upper bound must be at most
five points. An improvement in reports therefore cannot fail the non-inferiority
criterion.

## Capture, probes, and geometry

All 64 residual-block outputs are captured at the four history endings when
present, history end, mapping end, actions end, payoff end, and final answer.
States are stored as exact BF16 bit patterns in uint16 containers. Semantic
anchors are resolved from tokenizer offsets and paired anchor-token IDs are
checked before execution.

Activation geometry reports matched cosine and normalized-L2 distances, linear
CKA, and deterministic permuted-base controls with base-cluster bootstrap
intervals. It cannot select a patch site.

Standardized balanced L2 logistic probes use fixed `C=0.1`. Rule, target, and
correct-action probes train on discovery table-plus-opaque formats; chosen-action
probes train on all captured discovery formats. Core anchors are probed at all
layers; history episode anchors use the frozen checkpoint layer list. Locked data
are evaluated once, with metadata baselines and ten label permutations at four
sentinel sites. All probe results are descriptive.

## Natural interchange and J-space

Discovery searches only full residual replacement at layers 36–43 and the four
pre-answer anchors. Same-base donors preserve frame, target, response map, and
correct action. One donor family/layer/anchor is selected by base-cluster mean
correct-action logit-margin improvement, with deterministic table/layer/anchor
tie breaks, then hashed before locked capture.

The locked primary requires at least 20% repair, positive raw margin change, and
two-sided exact clustered `p<.05`. Identity, cross-base, opposite-target,
reverse, aligned/table/opaque non-damage, and consequence-report controls are
mandatory; control accuracy may lose at most five points. Normalized recovery is
reported only when its denominator is at least 0.10.

Required J-space runs after locked patching. Candidate-token projections cover
all captured rows at layers 36–43 and available anchors; full-vocabulary top-20
summaries cover eight seeded bases per split at the five core anchors. J-space
cannot select a patch site. Optimized candidate projections are checked against
full-unembedding slices.

## Provenance and claim ladder

The $12 cumulative V5 ceiling reserves approximately $1.253 behavior, $3.764
discovery, $4.517 locked, and $1.016 J-space buffered cost, in addition to the
current approximately $1.306 V5 ledger. Entrypoints refuse overwrite, failed
gates, missing hashes, pre-freeze locked access, and over-budget execution.

The claim ladder is: behavioral conjunction; probe decodability; observational
activation/J-space evidence; and, only if the locked endpoint passes, selective
causal transport under this contrast. The study does not establish a complete
circuit, deception, consciousness, or model-family generality.

The literature appendix retains the comparisons to token-aligned semantic
conflict patching, activation-patching metric sensitivity, interchange
interventions, function vectors, semantic-prior competition, and probe
selectivity.
