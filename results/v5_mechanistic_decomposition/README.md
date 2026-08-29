# V5-RBG-5 — mechanistic decomposition

**Status: behavioral gate failed; all activation and mechanistic output remains unopened.**

RBG-5 reran the focused RBG-4 behavioral conjunction on 48 fresh bases. The
37.50-point inverse-evidence effect replicated, but the absolute consequence-
report gap was 5.208 points against a frozen five-point maximum. Therefore no
activation phase was opened. Had the gate passed, discovery activations would select and hash one natural
table/opaque residual patch inside layers 36–43. The selected patch is then
tested once on the lexically disjoint locked split. Probes and optional J-space
readouts are secondary diagnostics. See [`FINDINGS.md`](FINDINGS.md) for the
complete gate table and claim boundary.

## Frozen execution order

```bash
modal run modal_v5_mechanistic_decomposition.py::freeze_dataset
modal run modal_v5_mechanistic_decomposition.py::preflight
modal run modal_v5_mechanistic_decomposition.py::behavior
modal run modal_v5_mechanistic_decomposition.py::discovery
modal run modal_v5_mechanistic_decomposition.py::locked
modal run modal_v5_mechanistic_decomposition.py::jspace
```

Large residual, probe, and patch payloads are content-addressed in the Modal
volume named `jspace-v5-rbg5-artifacts`; local manifests record their exact paths
and SHA-256 hashes. Download the locked patch and probe payloads to the default
paths shown by the manifests, then run:

```bash
uv run python scripts/analyze_v5_mechanistic_decomposition.py
```

Those later commands are documented for reproducibility but are now closed for
this frozen dataset because behavior did not pass conjunctively.

The preregistration is
[`../../docs/v5/mechanistic-decomposition-preregistration.md`](../../docs/v5/mechanistic-decomposition-preregistration.md).
