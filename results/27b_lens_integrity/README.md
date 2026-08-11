# Qwen3.6-27B released-lens integrity check

The official 1,000-prompt Jacobian Lens is structurally compatible with the selected model and produces a sensible reference readout. This clears Gate 1 for observational experiments; it is not a reporting-policy result.

## Structural checks

| Check | Lens | Model | Result |
|---|---:|---:|---|
| Residual width | 5,120 | 5,120 | Pass |
| Source layers | 0–62 (63 layers) | 0–63 (64 layers) | Pass |
| Lens fit prompts | 1,000 | — | Expected official artifact |
| Model revision | — | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Recorded |
| J-lens code | `581d398613e5602a5af361e1c34d3a92ea82ba8e` | — | Pinned |

## Qualitative reference

For the repository's official reference prompt:

```text
Fact: The currency used in the country shaped like a boot is
```

the early layer is token-level noise, while layer 31's top ten includes `Italy` and `意大利`. Across the full stored top-ten table, variants of *Italy* dominate the middle-depth readout. At layer 62 the readout shifts toward local/output words such as `is`, `boot`, `shaped`, and `Italy`, consistent with the expected move toward the output regime.

This validates that the lens can expose the intermediate country concept implied by “shaped like a boot”; it does not establish that the final currency answer itself appears at every layer.

## Toy prompt sanity check

On the selected one-token reporting prompt, the late J-lens readout at the final prompt boundary includes the literal candidates `A` and `B`, and the model-logit readout selects the correct answer. Middle layers do not show a stable policy word in the top ten for this one example. Candidate policy families must therefore be discovered quantitatively across balanced prompts rather than selected from an attractive anecdote.

## File

- `raw/lens_integrity_27b.json`: complete metadata and top-ten token/logit records for every fitted layer and inspected position.
