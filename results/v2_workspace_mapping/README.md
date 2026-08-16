# V2 task-independent Qwen workspace mapping

The frozen neutral-corpus rule selected **layers 36–43** (normalized depth
0.571–0.683) before flexible-generalization results were opened.

For every fitted layer, the run measured J-lens excess kurtosis and centered
J-lens/logit-lens cosine at two positions at or after token 16 on each eligible
neutral prompt. One of eight prompts was shorter than the lens fitting floor and
was retained as an excluded-prompt row; the other seven contributed 14
measurements per layer.

Within normalized depths 0.35–0.85, layers were scored by the preregistered
`z(mean excess kurtosis) − 0.5*z(mean J-lens/logit-lens cosine)` rule. The
lowest-starting contiguous width-eight band with maximal mean score won. No
strategic prompt or flexible-generalization outcome entered this selection.

## Reproducibility

- Command: `modal run modal_v2.py::h0`
- Run: `b1d093ae24ea40fc8c386d3dd50a5ece`
- Code commit: `ed187cba8e2429ce86ba305f467ada30f8be1768`
- Model/lens identities and config hashes: `run_manifest.json`
- Machine-readable selected band: `workspace_band.json`
- Raw measurements: `raw/workspace_diagnostics.jsonl`
- Deterministic summary and figure: `summaries/` and `figures/`

This artifact identifies a band under one explicit operational rule. It does
not establish that every task uses that band, and H0's failed persistent-swap
result prevents treating it as a validated causal workspace interval.
