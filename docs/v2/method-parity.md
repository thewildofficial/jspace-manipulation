# Protocol V2 method-parity ledger

| Method | Label | Basis |
|---|---|---|
| Jacobian loading, transport, model adapter, activation hook convention, and unembedding | **EXACT UPSTREAM** | Direct calls to the pinned `jlens` release |
| Released flexible-generalization prompt set | **EXACT UPSTREAM** | Byte-identified source JSON at the pinned commit |
| J-lens direction `v_t = J_l^T W_U[t]` | **FORMULA PARITY** | Local implementation numerically compared to the formula |
| Two-coordinate pseudoinverse swap | **FORMULA PARITY** | `V=[v_s,v_t]`, `c=V†h`, `h←h+αV(σ(c)-c)` with analytic tests |
| Workspace-band/all-position clamping convention | **FORMULA PARITY** | Released protocol semantics; upstream publishes prompts but not a runner |
| Qwen task-independent workspace selection | **V2 EXTENSION** | Frozen neutral-corpus kurtosis/motor-penalty rule |
| Four-topology comparison and intervention-distance recording | **V2 EXTENSION** | Local matched design |
| Truth/report tasks, reconstruction, probes, recipient games, and monitors | **V2 EXTENSION** | Not run until their upstream gates pass |

The upstream paper describes the J-lens as
`softmax(W_U norm(J_l h_l))` and its coordinate patch as a pseudoinverse swap.
The companion repository supplies the fitted-lens implementation and synthetic
experiment prompts. “Reference-protocol parity” therefore does not mean that an
unreleased Anthropic experiment runner was copied.

Primary sources: [paper](https://transformer-circuits.pub/2026/workspace/) and
[reference repository](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e).
