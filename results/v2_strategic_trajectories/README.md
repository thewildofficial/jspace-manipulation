# V2-ST-1 — strategic J-Lens trajectories

Historical study label: `V2-E2 MVP`.

**Project state: COMPLETE — EXPLORATORY NEGATIVE.**

## The question

Can a time-resolved top-k lens readout show which changing internal pathway
produced a strategic action?

## What happened

The model solved the final 48-row corpus perfectly, but the readout mostly showed
generic task meaning followed by late answer preparation. It did not give us a
clear, human-readable signal for the decisive strategy, and the strategic and
non-agentic versions did not show a compelling difference.

That is a limitation of this readout method, not proof that the relevant
variables are absent from the model.

V2-E2 measured a pinned top-k Jacobian-Lens readout proxy. It did not implement
the formal sparse nonnegative J-space decomposition, so the result should not be
described as a failure of J-space. Implementation, behavior gating, the cheap
mechanistic pass, and open inspection are complete. Patchscope was disabled and
was not run.

The frozen execution config retains its original `study_name` string so its
recorded config SHA-256 continues to match the archived run. That string is a
historical label, not the final name of the measured object.

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

All four same-action pairs pass the strengthened certificate: winner signal,
runner-up signal, expected label, runner-up label, and signal-to-label mapping are
fixed; the decisive response changes; each decisive fraction is at least 50%; and
each margin is at least 3.0. Receiver-causal pairs change only the probability
matrix among payoffs, costs, and probabilities; payoff-causal pairs change only
payoffs among those quantities.

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

## Open inspection

The initial classification is:

> Generic task semantics and action preparation without a decisive-pathway readout.

Within the stored selected-layer top-20 readouts:

- Direct runs contain no exact response/value/probability/payoff terms.
- The single-token exact response-identifier count is retained only as a tokenizer-
  dependent diagnostic. An identifier can split (for example, ` R` plus `2`), so
  this statistic is not evidence that a response representation is absent.
- Inspection did not reveal an early, coherent, human-readable distinction between
  the changing R1/R2/R3 decisive pathways.
- Strategic and isomorphic non-agentic framings show no compelling agent-specific
  difference in this top-k readout. Agent-noun counts are descriptive only and do
  not imply that opponent representations are absent.
- The expected action label is absent at the final prompt in all 48 rollouts but is
  present at layer 60 immediately before the answer in all 48.
- Legal-action logits already select the answer in 23/24 direct and 22/24 short-CoT
  final-prompt states, reaching 24/24 in both modes immediately before the answer.
- Short-CoT trajectories contain generic expected-value vocabulary shortly before
  its first surface occurrence. This exact lexical statistic imposes no minimum
  lead time and is not evidence that the readout anticipated reasoning.

The clean action-readout dissociation is narrow: output-facing action preference
was usually aligned before the literal answer label became top-k J-lens-visible.
It does not establish when the decision was made. A compact description of the
observed trajectory is generic task semantics, then answer-format preparation,
then literal action-token preparation.

## Final conclusion

> **V2-E2 was a controlled exploratory test of whether temporally resolved top-k
> Jacobian-Lens readouts could expose the response-conditioned computation
> underlying simple strategic choices. Behavior was perfect on the final corpus,
> and four same-action matched pairs changed the quantitatively decisive response
> pathway while preserving the winning and runner-up actions. The pinned J-lens
> trajectories showed generic calculation/choice semantics and reliable late
> answer preparation, but did not provide a coherent human-readable readout
> distinguishing those decisive pathways. Strategic and isomorphic non-agentic
> framings also showed no compelling agent-specific readout. The result constrains
> this top-k J-lens monitoring method; it does not establish absence of the relevant
> variables from residual activations or causal computation.**

Secondary observation:

> **Legal-action logits were usually action-aligned at the final prompt before the
> literal answer label became top-k J-lens-visible, suggesting that answer-token
> visibility in this readout can lag an output-facing preference for the eventual
> action.**

The `optim*` family is retired as substantive evidence: model-generated short CoT
uses `highest expected payoff`, `optimal`, and `maximize` even after those terms
were removed from prompts. `optim*` should not be reused as a substantive
strategic-monitoring token family without a much stronger matched control.

Several correct short-CoT decisions also state inaccurate expected values. This is
an unquantified exploratory lead for later offline work, not a CoT-faithfulness
claim. The absolute action-preparation result may also reflect the imbalanced
numerical-instance labels (A=8, B=1, C=3); no post hoc redesign was performed.

Residual artifacts remain archival only. No probe, sparse pursuit, Patchscope run,
causal intervention, or formal onset statistic was added. H0/H0R are not overridden:
the result makes no claim about causal use or absence, a strategic/opponent circuit,
deceptive intent, or CoT faithfulness.

See `summaries/exploratory_summary.json` for the reproducible checks.

## Cost and scope

The measured subtotal across three behavior attempts and the mechanistic pass was
`$0.87`; the buffered ledger total was `$1.05`, below the `$5` ceiling. Patchscope
remains disabled, and no causal intervention entrypoint exists.
