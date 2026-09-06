# Story note: Mid-trajectory ask — asking does nothing; lies rewrite the second press

Canonical protocol ID: **`ask_mid_trajectory`**. Scored `run_id`:
`ask-mid-traj-qwen38-n16-v1` (results folder only). Claim rows C19–C21.

On Qwen3.8 with 16 fresh nonce games, putting a consequence question **between**
two forced presses does not change the second press. Self mid-ask matches the
no-ask control: no flips, perfect choice2. The ask is measurement, not
intervention.

What *does* move the press is a **lied mid-answer**. When the injected mid token
is inverted, about **60%** of second presses flip — and every flip lands on the
wrong action. Words in the mid turn can steer choice2; asking the model to
produce those words itself does not.

Surface split (opaque flips more than prose) is descriptive and runs the
**opposite** way from ask-first swapped poisoning (C15/C16). Do not lock it.

Related next scientific screen (CPU wire only): **`rename_invariant`** — do
second presses follow labels or outcomes when only names change?
