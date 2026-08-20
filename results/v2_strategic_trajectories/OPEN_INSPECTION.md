# V2-E2 initial open inspection

The current J-lens MVP does not expose the experiment's decisive response pathway in
a human-readable way.

All four same-action matched pairs are valid: each pair keeps the winning action and
runner-up fixed while changing the unique response pathway carrying the positive
decision margin. Behavior is perfect on the final v3 corpus. The negative result is
therefore not explained by solver failure or malformed contrasts.

In direct mode, the five selected-layer top-20 trajectories contain no exact tokens
for response, probability, payoff, value, expected value, likelihood, responder, or
receiver. There are also no exact `R1`, `R2`, or `R3` tokens in any selected-layer
trajectory, direct or short-CoT. Inspection of the same-action direct transcripts shows
generic calculation/choice vocabulary followed by `FINAL` and the answer label; it does
not show the R1-versus-R2 or R2-versus-R3 distinctions that define the matched causes.

Short-CoT trajectories do contain generic `probability`, `expected`, `payoff`, `value`,
and occasional response-like tokens before those words appear on the surface. This is
not enough for a positive: those terms occur broadly across rows and framings, and none
identifies which response is decisive. The strategic condition also shows no agent-noun
advantage over the isomorphic mechanism control.

The cleanest transition observed is answer preparation. The expected label is absent
from the selected top-20 at the final prompt for all 48 runs and present at layer 60 at
the contextual pre-`FINAL` point for all 48. Meanwhile, ordinary legal-action logits
already select the correct action in 23/24 direct and 22/24 short-CoT final-prompt
states, then 24/24 in both modes before the answer.

The supported conclusion is limited to the instrument:

> In these controlled decisions, the pinned top-k J-lens trajectory showed generic
> task semantics and reliable late action preparation, but did not provide a
> human-readable readout of the response-conditioned pathway distinguishing matched
> cases with the same final action.

This does not show that the variables are absent from the residual stream, that they
are causally unused, or that another monitor could not recover them. Patchscope remains
optional exploratory follow-up, but the protocol's continuation criterion is not met by
the cheap J-lens readout alone.
