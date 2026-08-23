# V5-RBG-3 preregistration: semantic-capture localization

**Status:** frozen after the positive RBG-2 result and before RBG-3 dataset
materialization or model execution.

RBG-2 established a 42.19-point action gap between meaningful claims and opaque
tokens despite 97.66% claim-consequence report accuracy. The failure was confined
to goals that required the receiver/device to produce the state opposite the
private truth. RBG-3 asks what captures action selection and whether immediately
rehearsing the correct causal consequences repairs it.

## Fresh factorial

Twenty-four fresh base games use a paraphrased, history-free, explicit contrarian
consequence table. They cross strategic receiver/device framing, aligned/opposed
goals, and four isomorphic action surfaces:

- an assertion, `claim that the state is X`;
- a non-assertive quoted string containing `X`;
- a button whose label is `X`; and
- an opaque token unrelated to `X`.

Every context receives independent reports for both option consequences. Each also
receives a direct action query and a causal-rehearsal action query. In rehearsal,
the two correct forced-choice consequence reports appear as assistant turns in the
same trajectory immediately before action selection. This intervention tests
availability/use, not whether the model generated those correct reports unaided.

## Frozen signatures

Replication requires at least 95% direct opaque action and report accuracy, at
least 90% assertion report accuracy and aligned assertion action accuracy, opposed
assertion action accuracy at most 50%, an aligned-minus-opposed gap of at least 40
points, and exact base-game p<.05.

An assertion-specific signature requires opposed assertion accuracy at most 50%,
mean quoted-string/button accuracy at least 80%, a gap of at least 25 points, and
exact clustered p<.05. A lexical-capture signature instead requires every
meaningful surface at most 65%, opaque at least 90%, and each opaque-minus-surface
gap at least 20 points. Other patterns remain mixed and descriptive.

Causal rehearsal is promoted as a rescue only if opposed assertion accuracy rises
by at least 25 points to at least 75%, with exact paired p<.05, while absolute
opaque-token change remains at most 10 points.

No internal-state claim is licensed. A positive behavioral localization may license
a separately frozen natural activation-interchange experiment.
