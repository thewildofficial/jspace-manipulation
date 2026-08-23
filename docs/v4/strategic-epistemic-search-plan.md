# V4 strategic epistemic search plan

## Aim

V4 searches for an unexpected computational account of strategic cognition rather
than another generic deception classifier.  The target variables are:

- the sender's private state;
- the receiver-response distribution conditional on each signal;
- the sender's payoff vector;
- the response pathway that distinguishes the best signal from its runner-up;
- the decision margin and selected signal; and
- later reports about those quantities.

The central question is which of these variables are represented, globally
accessible, causally used, and verbally reported.  The Jacobian lens is one
instrument in this stack, not a gate on whether the scientific question may be
asked.

## Candidate discoveries

1. **Audience-indexed control.**  A representation of the receiver's likely
   response causally controls the sender's signal, separately from the sender's
   private state and payoff.
2. **Decision-sufficient workspace compression.**  The full residual stream
   preserves detailed receiver and payoff variables, while J-space preferentially
   exposes a compressed expected-utility margin or action.
3. **Strategic binding.**  The same numeric computation is represented differently
   when framed as influencing another agent versus controlling a non-agentic
   stochastic device.
4. **Pathway-without-action representation.**  When the optimal action is held
   fixed but the decisive receiver response changes, the causal pathway remains
   decodable even though output logits cannot identify it.
5. **Objective/receiver double dissociation.**  Receiver-state patches and payoff
   patches change decisions through separable layer/position routes.
6. **Action-before-explanation dissociation.**  Correct actions can coexist with
   incorrect reports of the decisive response or expected-value margin.
7. **Strategic automaticity.**  If later training is licensed, improving game play
   may reduce J-space dependence while preserving a causal non-J-space mechanism.

## Funnel and budget

The hard study ceiling is **USD 18.50 buffered**.

1. Generate and validate exact games locally: no GPU.
2. Behavior-only screen on all factorial cells: target <= USD 2.00 buffered.
3. Collect selected-layer residuals and J-lens readouts only if behavior passes:
   cumulative target <= USD 7.00.
4. Run bounded receiver/payoff/action counterfactual patches only for prospectively
   selected layers and pairs: cumulative target <= USD 13.00.
5. Spend the remaining <= USD 5.50 on a checkpoint/automaticity pilot only if a
   causal receiver-state mechanism has been established.

Every remote entrypoint refuses to overwrite results and writes a measured cost
row.  Failed behavior gates stop mechanistic execution but do not prohibit a
redesigned future study.

## Dataset structure

Each numeric instance is a three-signal, three-response decision problem.  A
receiver policy matrix gives `P(response | signal)`.  The sender has a payoff for
each response and a cost for each signal.  The exact signal value is

`EV(signal) = sum_response P(response | signal) * payoff(response) - cost(signal)`.

Rows are rendered in strategic sender/receiver language and in an isomorphic
non-agentic controller/device frame.  Three wording splits isolate rendering
generalization.  Opaque action labels are deterministically permuted.

Matched pairs include:

- receiver-pathway pairs: policy changes, payoff fixed, winner and runner-up fixed,
  decisive response changes;
- payoff-pathway pairs: payoff changes, policy fixed, winner and runner-up fixed,
  decisive response changes;
- receiver action-change controls;
- payoff action-change controls; and
- compression pairs with the same winner and approximately matched decision
  margin but different detailed policies.

All probe and bootstrap splits group every framing, wording, and query derived from
one numeric pair.  Output action logits are mandatory baselines.

## Claim discipline

Decodability is evidence that information is present, not used.  A patch that
changes behavior is not receiver modeling unless it beats action-state, shuffled,
and isomorphic controls and preserves unrelated variables.  Self-report mismatch
is not introspection failure unless prompt-only reconstruction controls are beaten.
No row is described as deception unless the model knows the state, predicts a
receiver belief/action inconsistent with it, and selects a message because that
prediction advances the sender payoff.
