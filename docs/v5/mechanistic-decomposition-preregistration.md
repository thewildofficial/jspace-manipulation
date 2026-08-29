# V5-RBG-5 preregistration: natural activation interchange

**Status:** prospectively frozen after RBG-4 and before RBG-5 dataset
materialization, model execution, activation inspection, or patch-site selection.

The deterministic expected dataset SHA-256 is
`238ac7d6a36c49851ef2ebabce201aca3ad3c33c2d79251ab67bf2ee16090c8e`.

## Question and claim boundary

RBG-4 showed that four correct prose demonstrations selectively harmed
assertion-based action while consequence reports stayed accurate, opaque actions
remained perfect, and a table representation removed the harm. RBG-5 asks whether
a naturally occurring state from a matched successful table or opaque trajectory
can causally repair a failed prose/assertion trajectory on held-out games.

Probe decoding and Jacobian-lens readouts establish only that information is
available. A passing locked interchange establishes selective causal transport
under this contrast, not a complete circuit, deception, consciousness, or a
model-family-general mechanism.

## Fresh focused corpus and behavioral gate

Forty-eight new base games are lexically disjoint from RBG-4 and from one another
across a 24-base discovery split and 24-base locked split. Each base crosses the
strategic/device frames with seven focused cells: the opposed assertion prose,
assertion table, and opaque prose conditions with and without redundant history,
plus an aligned assertion/redundant/prose control. Each action row receives two
independently rendered consequence-report queries.

No activation output may open unless the fresh behavior run has:

1. at least 95% consequence-report and opaque-action accuracy;
2. at least 20 points of opposed prose/assertion history harm, no more than five
   points of absolute report harm, and exact base-cluster sign-flip `p < .05`;
3. at least 20 points of assertion-minus-opaque history harm with `p < .05`;
4. no more than five points of table history harm; and
5. at least 12 locked redundant-prose failures with both reports correct for
   each natural donor family, so discovery cannot select a family without a
   sufficiently sized locked population.

Failure closes the mechanistic phase without opening activations.

## Residual capture and probes

All transformer-block residual outputs are saved in float16 at semantic anchors:
the four history episode endings when present, end of history, end of mapping,
end of actions, end of payoff, and final `Answer:`. Anchor character offsets are
frozen in the dataset and resolved through tokenizer offsets; absolute sequence
positions are not treated as semantic matches.

Standardized L2 logistic probes are trained only on discovery bases. Six-fold
grouped cross-validation chooses `C` from `{.01, .1, 1}` separately per target,
layer, and anchor. Locked bases are evaluated once. Targets are the caused
response under option A, desired response, correct action, and eventual unpatched
choice. Full trajectories, balanced accuracy, base-bootstrap intervals,
shuffled-label controls, and cross-format transfer are descriptive secondary
outcomes.

## Natural interchange

Recipients are opposed assertion/redundant/prose rows. Same-base donors preserve
the frame, target, randomized option labels, and underlying consequences while
changing only to assertion/redundant/table or opaque/redundant/prose. Donors must
be behaviorally correct.

Discovery patching is limited to full residual outputs at layers 36–43 and the
pre-answer history, mapping, actions, and payoff anchors. The donor/layer/anchor
with the largest base-cluster mean improvement in raw correct-action logit margin
is frozen. Ties prefer table, then the lower layer, then the earlier anchor. The
answer anchor is a positive control and cannot win selection. If no discovery
candidate has positive mean improvement, no locked patch runs.

The locked primary passes only if the frozen patch repairs at least 20% of
eligible failures and produces a positive paired correct-action margin change
with exact base-cluster sign-flip `p < .05`. Raw margin change is primary. Repair
rate and normalized recovery are secondary; normalized recovery is omitted when
the absolute source/destination margin gap is below 0.10 logits.

Identity, same-condition cross-base, opposite-target/action, reverse-direction,
aligned, table, opaque, and consequence-report controls are mandatory. Control
accuracy may fall by no more than five points, and an identity patch may change
the correct-action margin by at most `1e-4`. Arbitrary steering vectors,
head/MLP/path searches, and locked-site tuning are prohibited.

## Optional J-space analysis

The pinned Jacobian Lens may read the saved states only if the remaining buffered
V5 budget permits. On the first four bases of each split it reports action
margins, target/consequence evidence, and top tokens at layers 36–43 and every
anchor, and cannot select the patch site. If
the cost gate refuses it, the recorded outcome is `not_run_budget` regardless of
the core result.

## Provenance and stopping rules

The config, dataset, behavior, activation manifests, probe freeze, patch freeze,
raw patches, analysis, environment, model/tokenizer/lens revisions, seeds, GPU,
runtime, costs, and git commit are immutable or content-addressed. Entrypoints
refuse overwrites, locked capture before a valid patch freeze, execution after a
failed gate, and cumulative buffered V5 cost above USD 6.
