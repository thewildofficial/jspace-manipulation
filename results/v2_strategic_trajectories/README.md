# V2-E2 MVP — Strategic J-Space Trajectories

Implementation, behavior gating, the cheap mechanistic pass, and initial open
inspection are complete. Patchscope was not run.

## Frozen v3 corpus

- Source: `configs/v2/strategic_trajectories/dataset_source.json`
- Tokenized corpus: `configs/v2/strategic_trajectories/dataset.json`
- Rows: 48 rollouts from 12 numerical instances
- Matched pairs: 2 receiver-causal, 2 payoff-causal, 2 action-change controls
- Source SHA-256: `8da3baa00d46ec393bc96f4d0f507ce81bbc74c8581c4f40d8eebb5b260c93dc`
- Tokenized SHA-256: `1a4a81f14454d8dd1d2e6bca83eba62dbf20ab723cf8d1397c4924df2ee0c187`
- Probability grid: `{0.05, 0.15, 0.80}`
- Payoff range: `[-2, 10]`
- Minimum top-two margin after behavior-only redesign: `3.0`
- Action-label counts across numerical instances: A=8, B=1, C=3

All four same-action pairs keep the winner, runner-up, and label permutation fixed
while changing the unique decisive response pathway. The decisive response contributes
at least 50% of total positive support in every numerical instance.

## Behavioral development record

- V1 failed: 25/48 parseable and 23/48 correct. Direct mode usually began an
  explanation and exhausted its 24-token budget.
- V2 failed only the conjunctive format gate: 47/48 parseable and correct. One
  strategic short-CoT row used all 96 tokens before writing `FINAL:`.
- V3 passed: 48/48 parseable and correct, with 24/24 in strategic,
  non-strategic, direct, and short-CoT marginals.

Failed corpora and behavior outputs are retained with `_v1_failed_gate` and
`_v2_failed_gate` suffixes. No mechanistic result was opened for either failed version.

## Mechanistic artifacts

- Run ID: `3007f6b56637442d953b4a6f6804c10e`
- Compressed payload: `raw/mechanistic.json.gz`
- Payload SHA-256: `320ae20c2f74daed5cc74116389e8eb84badf54955166965af1070e1d459f883`
- Human-readable transcripts: `transcripts/transcripts/`
- Residual audit files: 48 files on the `jspace-v2-e2-trajectory-artifacts`
  Modal volume; they were not used for a separate analysis endpoint.

The pass records top-20 J-lens trajectories at layers 34, 42, 46, 54, and 60;
an all-fitted-layer top-50 final-prompt diagnostic; contextual A/B/C logits; and FP16
residual audit artifacts. The downloaded payload passed its checksum and schema checks.

## Initial open inspection

The initial classification is:

> Generic task semantics and action preparation without a decisive-pathway readout.

Within the stored selected-layer top-20 readouts:

- Direct runs contain no exact response/value/probability/payoff terms.
- No rollout contains an exact `R1`, `R2`, or `R3` token.
- Agent nouns do not favor the strategic condition; the sole `responder` occurrence
  is in a non-strategic rollout.
- The expected action label is absent at the final prompt in all 48 rollouts but is
  present at layer 60 immediately before the answer in all 48.
- Legal-action logits already select the answer in 23/24 direct and 22/24 short-CoT
  final-prompt states, reaching 24/24 in both modes immediately before the answer.
- Short-CoT trajectories anticipate generic words such as `probability`, `expected`,
  `payoff`, and `value`, but these do not identify the changing decisive response.

This is an exploratory negative for the human-readable monitoring objective. It does
not establish that decisive variables are absent from residual activations or causally
unused. No residual probe, causal intervention, or formal onset statistic was added.

See `summaries/exploratory_summary.json` for the reproducible checks.

## Cost and scope

The measured subtotal across three behavior attempts and the mechanistic pass was
`$0.87`; the buffered ledger total was `$1.05`, below the `$5` ceiling. Patchscope
remains disabled, and no causal intervention entrypoint exists.
