# V5-RBG-5 — mechanistic decomposition

**Status: prospectively frozen; no model output has been opened.**

RBG-5 first reruns the focused RBG-4 behavioral conjunction on 48 fresh bases.
If that gate passes, discovery activations select and hash one natural
table/opaque residual patch inside layers 36–43. The selected patch is then
tested once on the lexically disjoint locked split. Probes and optional J-space
readouts are secondary diagnostics.

## Execution order

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

The preregistration is
[`../../docs/v5/mechanistic-decomposition-preregistration.md`](../../docs/v5/mechanistic-decomposition-preregistration.md).

