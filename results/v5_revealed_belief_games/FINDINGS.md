# V5-RBG-1 findings: receiver-policy use gate failed

## Verdict

The first revealed-belief screen is negative for its intended claim. Qwen3.6-27B
reached only 52.08% baseline action accuracy, the weakest preregistered cell was
0%, and only 11.46% of matched receiver-type pairs contained two correct,
different actions. The study therefore stops before any internal-state analysis.
It does not establish inferred receiver beliefs, deception, action/report
contradiction, or an epistemic measurement effect.

## Frozen outcomes

| Outcome | Result | Gate |
|---|---:|---:|
| Baseline action accuracy | 100/192 (52.08%) | at least 85% |
| Minimum frame × incentive × type cell | 0/24 (0%) | at least 70% |
| Correct type-counterfactual action pairs | 11/96 (11.46%) | at least 75% |
| Direct report accuracy among correct actions | 82.0% | diagnostic |
| Post-action report accuracy among correct actions | 84.0% | diagnostic |
| Private pre-action policy probe accuracy | 77.60% | diagnostic |
| Action changed after private report | 29.69% | diagnostic |

The preregistered opposed-minus-aligned post-action contradiction effect was
+18.18 percentage points but had cluster sign-flip p=.0625 and was ineligible for
promotion because the receiver-policy gate failed. The private-elicitation action
change interaction was +10.42 points, p=.3018.

## Diagnostic decomposition

The dominant error is a literal-message heuristic. In all 24 strategic aligned
contrarian games, the model chose the truthful claim even though the demonstrations
showed that this made the receiver choose the wrong state. In strategic opposed
contrarian games, 15/24 choices were false claims that likewise induced the
non-target response. In other words, the selected claim's semantic content often
matched the desired receiver conclusion while the demonstrated causal receiver
mapping was ignored.

Private policy elicitation repaired 13/24 strategic aligned contrarian decisions
and all seven initially wrong strategic opposed literal decisions. It harmed eight
initially correct strategic opposed contrarian decisions. These are post-hoc
diagnostics and do not establish a measurement effect. They motivate a fresh test
that separates receiver-policy inference from semantic action-outcome binding.

## Cost and provenance

- Git commit: `0e1598b66f06113d14888f5d4ee4a4d389eca70e`
- GitHub Actions run: `32663979874`
- Modal run: `5b10e27f3f91454e84c06d8acb43e78e`
- A100 elapsed time: 176.323 seconds
- Measured cost: USD 0.15337
- Buffered cost: USD 0.18405
- Raw payload SHA-256:
  `f7e6916ad525f74e5417c1c5ad7f7689b6034c764d0841560fa4a96a2a85efcd`

## Next decision

Do not probe or patch RBG-1. Freeze RBG-2 crossing inferred versus explicit policy
access with meaningful claims versus arbitrary tokens. Promote only if the model
can report the explicit option consequences while meaningful message semantics
still selectively corrupts action choice.

