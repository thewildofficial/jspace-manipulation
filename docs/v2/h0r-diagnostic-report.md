# Protocol V2 H0R-B diagnostic report: Qwen causal-write topology

Status: **exploratory diagnostic complete; candidate frozen; prospective controls unopened**.

The original H0 remains failed. H0R-B used only the already-burned Anthropic
country controls. Neither locked H0R control was evaluated while selecting the
candidate protocol.

## Runs and implementation checks

Three pinned A100-80GB runs produced 4,272 causal diagnostic rows plus 1,392
downstream reconstruction records:

- layer sweep: `cf00eca905f44b50ab5b6ee3f2e3fc07`;
- position and strength sweep: `155fbee330fe459ca4bbfdf85f6c20a5`;
- cancellation, semantics, controls, and reconstruction:
  `24066f0e85ac409296a9555ae212cbf8`.

The largest fp32 coordinate-target error across the layer sweep was
`1.90735e-5`, below the frozen `2e-5` tolerance. The selected configuration's
median basis condition number was 1.75; no eligible selected trial exceeded
the absolute-cosine limit of 0.95.

## Diagnosis

At alpha 1.0, every tested single layer moved target-vs-source log odds in the
predicted direction, but mean output KL was about 1.4--5.2 nats. The previous
H0 topology was therefore not a numerically failed swap; it repeatedly made an
overpowered write.

The strength sweep found a smooth lower-perturbation regime. At layer 34 and
alpha 0.5, argument-through-end writing produced mean gain +1.92, 100% positive
direction, KL 0.048, and mean delta-RMS ratio 0.057. Alpha 0.25 was weaker;
alpha 1.0 and above exceeded the preregistered KL validity limit.

At alpha 0.5, cumulative layers 36--42 produced mean gain +4.34 (scenario-
bootstrap 95% interval +3.62 to +5.21), 100% positive direction, KL 0.227, and
mean delta-RMS ratio 0.017. Even layers 36/38/40/42 and odd layers 37/39/41/43
were similar (+4.30 and +4.26), so the diagnostic does not support strong
odd/even re-swap cancellation in this regime. Alpha 0.5 places the post-hook
source-minus-target coordinate at the midpoint rather than reversing its sign.

Matched identity and random-basis controls were null (-0.010 and -0.011 mean
gain). An unrelated same-category write produced +0.19, and an out-of-window
layer produced +0.29. Direct-answer-coordinate writing produced +1.01, below
the selected semantic protocol but non-null; it remains an essential
prospective control. Donor patching at the exploratory single layer produced
+1.76 versus +1.91 for the symmetric coordinate operation.

A one-shot layer-34 write returned 25% toward the clean source state by layer
39 and 50% by layer 49. The frozen 75% threshold was not reached; the maximum
median clipped return was 0.747. This gradual reconstruction supports repeated
moderate writes rather than one extreme write.

## Frozen candidate

The preregistered validity filters and mean within-metric rank selected:

```text
layers: [36, 37, 38, 39, 40, 41, 42]
positions: argument_through_end
operation: pseudoinverse_coordinate_swap
alpha: 0.5
vector normalization: unit L2 per layer
hook: transformer block output residual
```

The exact machine-readable protocol is
`configs/v2/h0r_candidate_protocol.json`. No layer, position, alpha, operation,
threshold, or validity limit may change during H0R-C or H0R-D.

Target-answer top-1 conversion on burned countries was only 1/33 under this
valid candidate. H0R-C therefore remains a stringent prospective test; the
large burned-set log-odds effect is not treated as validation.
