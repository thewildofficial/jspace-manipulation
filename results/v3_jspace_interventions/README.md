# V3-JI1 results

Status: **Phase B passed its frozen proceed gate; fresh interventions remain
unopened and have not yet been generated.**

Generated artifacts are written under `manifests/`, `raw/`, `summaries/`, and
`figures/`. The final classification must preserve the historical H0R-C failure.

## Burned Phase A/B result

All four category shards passed numerical validation. Maximum FP32 coordinate
target error was `3.5763e-7`; identity was exact; an identity hook changed clean
logits by exactly zero; lens layers 36 and 43 were present.

On the burned, baseline-eligible alpha-1 rows:

- native-raw semantic mean target-vs-source log-odds gain: `+0.5753`;
- cluster-bootstrap 95% CI: `[+0.3564, +0.8106]`;
- positive semantic fraction: `83.3%`;
- delta-matched random mean: `+0.0948`;
- unrelated-semantic mean: `+0.0691`.

The frozen Phase B proceed gate therefore passed. The pre-Phase-D power model,
using the burned source-prompt cluster SD of `0.6901`, estimates `99.85%`
detection probability for a true mean effect of `+0.5` with 48 fresh clusters.

The burned dose response is non-monotonic. Alpha 0.5 was strongly positive,
whereas native-raw alpha 2 was destructive: mean effect `-10.31`, mean
`KL(clean || patched)=21.66`, and mean delta-RMS/RMS `2.03`. Alpha 2 remains in
the frozen fresh curve and must be reported, but the burned result already shows
that it is not a low-distortion regime on this Qwen stack.

Phase B contains 2,550 condition rows and 14 recorded tokenization exclusions,
all in the burned animal category. No exclusion was selected using an
intervention outcome.
