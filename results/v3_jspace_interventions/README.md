# V3-JI1 — J-space intervention replication

Status: **Phase B passed its frozen proceed gate; Phase C failed the final
behavior-only corpus-freeze gate, so the study stopped before Phase D. No fresh
intervention was generated or opened.**

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

## Prospective Phase C stop

Four ordered candidate pools were evaluated using tokenizer checks and clean
next-token inference only. None produced the required three feasible 4x4
category blocks. The final v4 run (`e1544e72c6144b8c815a2b6274388989`)
made all 90 cells tokenizer-eligible, but only 42 were clean top-1 correct:

- alphabet letters: 0/30;
- months: 12/30;
- number words: 30/30.

The selector therefore refused to write `fresh_frozen.json`. Under the final
pre-intervention amendment JI1-D009, this is a mandatory study stop: no further
prompt redesign, relaxed correctness rule, or rescue intervention sweep is
permitted. The prospective causal hypotheses are untested, not rejected. The
positive burned result remains useful implementation and instrument evidence,
but it is not prospective confirmation.
