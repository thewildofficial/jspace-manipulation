# V5-RBG-5 findings: behavior replicated, mechanistic gate closed

## Verdict

The fresh behavioral effect replicated strongly, but the conjunctive frozen gate
failed. Four correct prose demonstrations reduced opposed assertion action
accuracy from 69.79% to 32.29%, a 37.50-point effect with exact base-cluster
`p=.0000178`. Opaque actions stayed at 100%, and table history improved rather
than harmed action by 4.17 points.

The absolute prose consequence-report gap was 5.208 points, just above the frozen
five-point ceiling: no-history reports were 178/192 (92.71%), versus 188/192
(97.92%) with history. Because RBG-5 required every behavioral gate to pass,
discovery activations, probes, activation patches, locked states, and J-space
readouts remain unopened.

## Frozen outcomes

| Outcome | Result | Frozen gate | Verdict |
|---|---:|---:|---|
| All consequence reports | 1321/1344 (98.29%) | at least 95% | Pass |
| Opaque action accuracy | 192/192 (100%) | at least 95% | Pass |
| Prose assertion, no history | 67/96 (69.79%) | contrast arm | — |
| Prose assertion, redundant history | 31/96 (32.29%) | contrast arm | — |
| Prose assertion history harm | 37.50 points | at least 20 points | Pass |
| Exact clustered history p | .0000178 | below .05 | Pass |
| Absolute prose report gap | 5.208 points | at most 5 points | **Fail** |
| Assertion-minus-opaque history harm | 37.50 points | at least 20 points | Pass |
| Exact semantic-specificity p | .0000178 | below .05 | Pass |
| Table history harm | −4.17 points | at most 5 points | Pass |
| Locked eligible failures | table 27; opaque 32 | at least 12 each | Pass |

The report-gap miss was symmetric across splits. Discovery and locked no-history
prose reports were each 89/96, while redundant-history reports were each 94/96.
It is therefore not a single-split anomaly, but the threshold miss is only one
correct report out of 192 away from the boundary.

## Claim boundary

RBG-5 supplies a fresh behavioral replication of inverse evidence and its
semantic/representation specificity. It supplies no new internal evidence. The
predeclared stop rule forbids treating the prepared probe, patch, or J-space code
as an experiment that was run.

## Cost and provenance

- Dataset SHA-256:
  `238ac7d6a36c49851ef2ebabce201aca3ad3c33c2d79251ab67bf2ee16090c8e`
- Git commit: `610da5a51dae543dccbdd7f962cb305da73878ce`
- Modal run: `c100debc2344409eb69e06eaa02a6b81`
- Raw behavior SHA-256:
  `318a62c965f622777e8518353993c27b868095f685035c373ed966abe95ff266`
- Behavior analysis SHA-256:
  `efd25309b3032b174365f96159686d8209f3a9a29e2c2d38e1e09cba37a3db00`
- A100 elapsed time: 305.696 seconds
- Measured cost: USD 0.26591
- Buffered cost: USD 0.31909
