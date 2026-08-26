# Stage 1C — Gate 1: released-lens integrity

## The question

The Jacobian Lens is the measuring instrument used in the rest of the project.
Before trusting it, we checked that it was built for this model and that it
produced sensible reference readouts.

## What happened

The released 1,000-prompt lens passed the structural checks:

| Check | Expected / observed | Result |
|---|---|---|
| Residual width | Lens 5,120; model 5,120 | Pass |
| Source layers | Lens 0–62; model 0–63 | Pass |
| Fitted prompts | 1,000 official prompts | Recorded |
| Model revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Recorded |
| Lens code | `581d398613e5602a5af361e1c34d3a92ea82ba8e` | Pinned |

For the reference prompt “The currency used in the country shaped like a boot
is”, middle layers surfaced Italy-related tokens as expected. On the toy
reporting prompt, the final readout also contained the literal answer tokens A
and B.

## The useful caution

The lens does not show the same thing at every depth. Early layers look noisy;
middle layers can expose a concept; late layers move toward output words. One
attractive example is therefore not enough to choose an intervention target.
The later pilot used matched prompts and measured families of tokens instead.

## What this clears—and what it does not

This clears the instrument check for observational work. It does not prove
that a reporting policy exists inside the model, and it does not prove that the
lens coordinates cause the model’s behavior.

The complete metadata and top-ten records are in
[`raw/lens_integrity_27b.json`](raw/lens_integrity_27b.json).

The model/lens readout entry point is [`../../modal_app.py`](../../modal_app.py);
the reusable intervention and lens-direction helpers are documented in
[`../../src/jspace_policy/`](../../src/jspace_policy/README.md).

[Previous run: 27B behavior](../27b_behavioral_gate/README.md) · [Next run: observational pilot](../27b_observational_pilot/README.md) · [Results map](../README.md)
