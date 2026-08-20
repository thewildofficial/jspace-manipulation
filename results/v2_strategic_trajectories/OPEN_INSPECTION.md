# V2-E2 Strategic J-Lens open inspection

The pinned top-k Jacobian-Lens readout proxy does not expose the experiment's
decisive response pathway in a coherent human-readable way. This experiment did
not implement formal sparse nonnegative J-space, so the negative result applies to
this J-lens monitor rather than to J-space generally.

All four same-action matched pairs are valid: each pair keeps the winning action and
runner-up fixed while changing the unique response pathway carrying the positive
decision margin. Behavior is perfect on the final v3 corpus. The negative result is
therefore not explained by solver failure or malformed contrasts.

In direct mode, the five selected-layer top-20 trajectories contain no exact tokens
for response, probability, payoff, value, expected value, likelihood, responder, or
receiver. The single-token exact `R1`/`R2`/`R3` count is only a tokenizer-dependent
diagnostic: identifiers may split across tokens, so the count cannot support a
representational-absence claim. Inspection of same-action transcripts did not reveal
an early, coherent, human-readable R1-vs-R2-vs-R3 distinction. Instead, it shows
generic calculation/choice vocabulary followed by answer-format and literal-label
preparation.

Short-CoT trajectories contain generic `probability`, `expected`, `payoff`, and
`value` tokens shortly before their first exact surface occurrence. The statistic
has no minimum lead time and does not establish anticipation of reasoning. These
terms occur broadly across rows and framings and do not identify the decisive
response. Strategic and isomorphic non-agentic framings provide no positive evidence
of a distinct human-readable agent model in this top-k readout. Agent-noun counts are
descriptive only; they do not establish absence of opponent representations.

The cleanest transition observed is an action-readout dissociation. The expected
label is absent from the selected top-20 at the final prompt for all 48 runs and
present at layer 60 at the contextual pre-`FINAL` point for all 48. Meanwhile,
ordinary legal-action logits already select the correct action in 23/24 direct and
22/24 short-CoT final-prompt states, then 24/24 in both modes before the answer. The
narrow interpretation is that output-facing action preference was usually aligned
before literal answer-token visibility in this readout; this does not establish when
the decision was made. Label imbalance (A=8, B=1, C=3 across numerical instances)
may affect absolute action-preparation results.

The supported conclusion is limited to the instrument:

> In these controlled decisions, the pinned top-k J-lens trajectory showed generic
> task semantics and reliable late answer preparation, but did not provide a
> coherent human-readable readout of the response-conditioned pathway distinguishing
> matched cases with the same final action.

This does not show that the variables are absent from the residual stream, that they
are causally unused, or that another monitor could not recover them. Residual artifacts
are archival only. Patchscope was disabled and was not run; no probe, sparse pursuit,
or causal intervention will be added to rescue V2-E2. The experiment closes as
**COMPLETE — EXPLORATORY NEGATIVE**.
