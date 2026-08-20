# V2-E1: Strategic Workspace Atlas

Status: **exploratory protocol frozen before model behavior or mechanistic output**.

## Historical and claim boundary

V2-E1 is an observational study. It does not reopen the failed H0/H0R causal
instrument, license the closed causal Stage 2, or reinterpret the negative Stage 1
J-space endpoint. Its purpose is to map which externally defined strategic variables
are recoverable from the residual stream, visible through the pinned Jacobian Lens,
or expressed in output-facing action probabilities.

No result from this study alone establishes deception, intent, consciousness, causal
use, or a generally reliable monitor.

## Scientific object

Each decision is represented as

\[
Z=(X,B,G,V,S,A),
\]

where \(X\) is private state, \(B\) is belief or uncertainty, \(G\) is an
objective, \(V\) is a scalar decision variable, \(S\) is strategy or information
policy, and \(A\) is the selected action. These quantities are calculated by the
game generator before the model is run.

The study compares three views:

1. regularized decoders trained on residual-stream activations;
2. full-vocabulary Jacobian-Lens readouts; and
3. conventional logit-lens and final legal-action logits.

The central exploratory question is which components of \(Z\) enter each view, at
what depth, and whether J-space distinguishes matched cases with the same action but
different strategic causes.

## Game families

The deterministic corpus contains six families:

- **inspection:** a continuous audit-probability control with an exact expected-value
  switching boundary;
- **Kuhn-style one-card poker:** private card, opponent call belief, expected values,
  and bluff/value/check classifications;
- **finite cheap talk:** private state, sender preference conflict, induced receiver
  action, and aligned versus influence messages;
- **signaling:** private type, type-dependent signal cost, induced impression, and
  separating/pooling classifications;
- **selective disclosure:** two true features, receiver threshold, disclosure cost,
  and selective versus full disclosure;
- **controlled Chameleon clue selection:** a language-rich transfer assay with fixed
  clue choices and externally assigned informativeness. Its scoring rule is a
  controlled heuristic, not an equilibrium solution, so it is excluded from the
  exact-game behavioral gate.

Every primary prompt uses the same legal actions `A`, `B`, and `C`, each required to
be a distinct one-token continuation after the frozen answer prefix. The primary
condition uses Qwen's pinned non-thinking chat template and requests no explanation.

## Solver-validation gate

The game systems must pass local tests before tokenization or GPU execution. Tests:

- independently recompute the payoff vector and optimal action for every row;
- verify the stored top-two value margin;
- cover every game, parameter regime, rendering split, and legal action;
- confirm that discovery, validation, and locked splits contain identical parameter
  inventories under different renderings;
- require action diversity and both same-action/different-strategy and
  same-state/different-action dissociations;
- verify paired bootstrap groups never cross a game or split;
- reject duplicate condition identifiers, non-finite values, answer-key leakage, or
  inconsistent content hashes.

A failure stops the pipeline before model execution.

## Splits and information history

Each game has 12 independent parameterized rows in each of three rendering splits:
discovery, validation, and locked replication. Discovery and validation form the open
atlas. Locked replication remains unopened until an exploratory finding and its
analysis rule are frozen. The solver parameter inventory is the same across splits,
but the natural-language template family is different.

Matched variants are clustered by `matched_group_id`. Paraphrases and parameter
counterfactuals are never treated as independent evidence merely because they are
separate rows.

## Behavioral gate

Behavior is run without loading the J-lens or saving residual activations. The open
and locked phases are gated separately. Mechanistic execution requires:

- at least 98% full-vocabulary formatting compliance;
- at least 70% optimal-action accuracy pooled across the five exact games; and
- at least 50% optimal-action accuracy in every exact game.

The Chameleon assay is described behaviorally but excluded from this conjunction.
The gate measures basic task validity, not equilibrium sophistication. Full legal
action probabilities, entropy, action margin, and solver regret are retained even
when top-1 behavior is non-optimal.

## Mechanistic atlas

For every valid row, the primary measurement is the final prompt position before any
answer token. At every fitted Qwen lens layer, store:

- top-50 J-lens token IDs, decoded strings, raw scores, and ranks;
- J-lens and conventional logit-lens scores for all legal actions;
- final output legal-action log probabilities; and
- residual-, J-space-top-k-, and output-based probe metrics for categorical and
  scalar game variables.

A deterministic deep subset—the lexicographically first condition in every game and
open split—also stores the top-10 J-lens readout across the final 32 prompt positions
at every fitted layer.

Top-k J-lens ranking is not called a formal sparse J-space decomposition. Sparse
nonnegative gradient-pursuit decomposition is a separate corroborating extension and
requires its own implementation and validation.

## Exploratory analyses

The open atlas will report:

- behavior and regret by game, state, strategy, and split;
- residual/J-space/output decoder trajectories for \(X,B,G,V,S,A\);
- top-token frequency, prompt-echo rates, and layer persistence;
- same-action/different-strategy and same-state/different-action contrasts;
- action commitment depth under the conventional logit lens;
- J-space concept onset and potential lead time over action commitment; and
- deep layer-by-position traces for the deterministic subset.

Human inspection is permitted in the open phase. Tokens are retained raw. Any
post-hoc semantic lexicon, concept family, layer band, onset rule, effect threshold,
or selected game comparison is exploratory until written into a replication freeze.

## Replication rule

After open-atlas inspection, only a small number of findings may advance. For each,
the freeze must specify the exact games, variable, rows, token lexicon or decoder,
layers/positions, comparator, statistic, direction, bootstrap unit, and success
criterion. The locked rendering family is then opened once.

The strongest result would distinguish same-action strategic causes, generalize to
an unseen rendering and preferably another game, survive prompt-echo and output-logit
controls, and appear before output commitment. An action-only, prompt-echo-only, or
unreplicated token anecdote is not a positive strategic-workspace result.

## Frozen identifiers

The machine-readable protocol is
`configs/v2/workspace_atlas/experiment.json`. It pins the model, tokenizer, lens,
dataset seed, splits, behavioral thresholds, probe variables, bootstrap settings,
readout depth, top-k values, compute type, overwrite refusal, and cost ceiling.
