# V5-RBG-6 findings: endogenous full action trajectory

## Verdict

Issue #11's full-trajectory intervention produced the opposite of a persistent
report/action dissociation in the preregistered follow-up cell. When Qwen3.6-27B
reported both consequences and then saw those exact answers as assistant turns,
it selected the payoff-maximizing action in **46/48** trajectories where both
reports were correct. There were **0/48** cases of “both self-reports correct,
action wrong” (exact 95% binomial interval: 0%–7.40%). The frozen persistence
signature therefore failed.

The trajectory acted as a strong behavioral rescue: self-generated action
accuracy was **95.83%**, versus **37.50%** for the immutable direct baseline, a
**58.33-point gain** with exact clustered sign-flip `p=0.000061`. Correct-report
oracle replay was 100%, showing the rescue is available when the report path is
correct. The matched opaque-token control remained at 100% self-generated action
accuracy.

This is a follow-up on the already observed RBG-4 contexts, not an independent
replication. It supports a narrow claim: in this task and prompt family,
keeping the model's own consequence answers in the action trajectory improves
action selection. It does not establish deception, consciousness, a stable
preference, or an internal mechanism.

## Frozen outcomes

| Endpoint | Result | Frozen criterion |
|---|---:|---:|
| Primary trajectories | 48 | strategic/opposed/assertion/redundant/prose, both report orders |
| Self-report accuracy | 46/48 trajectories had both reports correct (95.83% report-token accuracy) | at least 90% |
| Both reports correct + action wrong | 0/48 (0.00%; exact 95% CI 0.00%–7.40%) | at least 25% for persistence |
| Direct action accuracy | 18/48 (37.50%) | descriptive baseline |
| Self-generated action accuracy | 46/48 (95.83%) | descriptive endpoint |
| Self-generated − direct | +58.33 points | at least +25 points for rescue |
| Exact clustered sign-flip p | 0.000061 | p<.05 for rescue |
| Oracle replay action accuracy | 48/48 (100%) | descriptive transcript control |
| Swapped replay action accuracy | 5/48 (10.42%) | descriptive transcript control |
| Swapped-report-implied action accuracy | 43/48 (89.58%) | descriptive transcript control |
| A→B vs B→A self-action accuracy | 95.83% vs 95.83% | absolute gap ≤10 points |
| Matched opaque self-action accuracy | 48/48 (100%) | at least 90% |

The swapped replay result is an important control: changing only the visible
report tokens reverses the chosen action in 43/48 cases. The model is therefore
not merely ignoring the report turns; it is using their visible content. The
self-generated trajectory agrees with oracle replay whenever its reports are
correct, and its two report errors did not produce a wrong action in the primary
cell.

## Cell diagnostics

Self-generated action accuracy by assertion cell (each `n=48`, including both
report orders):

| Frame | History | Mapping | Direct | Self-generated | Oracle | Swapped |
|---|---|---|---:|---:|---:|---:|
| strategic | none | prose | 75.00% | 97.92% | 100% | 29.17% |
| strategic | none | table | 83.33% | 97.92% | 97.92% | 43.75% |
| strategic | redundant | prose | 37.50% | 95.83% | 100% | 10.42% |
| strategic | redundant | table | 83.33% | 97.92% | 97.92% | 25.00% |
| device | none | prose | 75.00% | 83.33% | 100% | 10.42% |
| device | none | table | 83.33% | 100% | 100% | 27.08% |
| device | redundant | prose | 16.67% | 93.75% | 100% | 14.58% |
| device | redundant | table | 83.33% | 100% | 100% | 31.25% |

## Compute and provenance

- GitHub Actions run: [33872743994](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/33872743994)
- Modal app run: [ap-nlfDrOHpvLH2DDntJzpHt4](https://modal.com/apps/thewildofficial/main/ap-nlfDrOHpvLH2DDntJzpHt4)
- Modal run ID: `625672989c0b4262828e3cd80332dfb9`
- Git commit used for the run: `f2d0e0fedee8eee374f3daeba208236d8a817ab3`
- A100-SXM4-80GB elapsed time: 467.2859 seconds
- Measured Modal cost: USD 0.40646; buffered ledger cost: USD 0.48776
- Raw payload SHA-256: `06b464386cbb9ab52867dbe6758b10b3e14afdee270e46335e47f924c61d6a88`
- Analysis SHA-256: `d9da361fa70adb9e75589175a53f998d19ebe6e18680560da63b3f910296a764`
- Local preflight SHA-256: `e248ed0db29c9097dba5cd580bbc6c220f46ab6b2a4de81592f0c507015a8bfe`
- Prepared branch payload SHA-256: `7bdf10288c043d588e077fccc81a49da5949655602bebe4e3aec9dc5fa8537aa`

Modal emitted one CUDA allocator warning while the function was running, but
the function returned a complete 768-row payload and the workflow finished
successfully. No retry or second paid run was started.

## Files

- [`../configs/v5/full_action_trajectory/experiment.json`](../../configs/v5/full_action_trajectory/experiment.json) — frozen design and thresholds.
- [`../../docs/v5/full-action-trajectory-analysis-plan.md`](../../docs/v5/full-action-trajectory-analysis-plan.md) — protocol and compute placement.
- [`raw/behavior_v1.json`](raw/behavior_v1.json) — immutable trajectory rows.
- [`analysis/behavior_v1_analysis.json`](analysis/behavior_v1_analysis.json) — deterministic offline analysis.
- [`raw/preflight_v1.json`](raw/preflight_v1.json) — local tokenizer/branch preflight.
- [`cost_ledger.jsonl`](cost_ledger.jsonl) — measured cost ledger.
