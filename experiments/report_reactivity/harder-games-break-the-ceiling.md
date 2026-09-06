# Harder games: break the Direct ceiling

**Story.** Current report-reactivity demos use balanced prior trials
(`history_mode=minimal`). Direct / self / control sit at ceiling, so
rewrite-vs-reveal rescue is unidentifiable. Inverse-evidence work in this repo
already showed that **redundant correct history** can worsen wordy action —
enough signal to unlock ask-first / self-report contrasts when Direct fails.

## Outcome (scored) — failed to break Direct

Actions run
[34051102729](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34051102729)
/ story dir `harder-games-qwen38-n16-v1` (`history_mode=redundant`, 16 discovery
bases, Qwen3.8-27B, payload sha256 `a8b43df9…`):

- **Direct / matched_control / self_report / oracle / external_facts: all 1.0**
- Swapped ≈ **0.789** (still leaks; strategic cells worst; worst cell
  prose|opposed|strategic = **0.5**)
- Primary self−control / self−direct = **0.0**
- Parity passed (`batch_single_max_abs=0.25`)

**Verdict.** Redundant demos did **not** break the Direct ceiling. Ask-first
rescue remains unmeasurable this way. Not claimed: that demos make games harder
in a useful way. Methods:
[`harder-games-qwen38-n16-v1-methods.md`](harder-games-qwen38-n16-v1-methods.md);
claims **C17** / **C18**.

Prior minimal-history contrast: Actions 34048123330 / `gha-report16-38-v1`
(Direct also 1.0; swapped ~0.828; worst cell 0.375).

## Knob

| Mode | Meaning |
|---|---|
| `minimal` (default) | Historical demos: four correct repeats of each mapped action (eight “Prior trial” lines). Preserves prior prepare hashes. |
| `redundant` | Harder games: **two extra full correct cycles** (twelve Prior-trial lines). Same nonce lexicon, split hygiene, common final action query, all six arms. |

CLI:

```bash
uv run python experiments/report_reactivity/prepare.py --task report \
  --model Qwen/Qwen3.8-27B --bases 16 \
  --history-mode redundant \
  --output artifacts/prepared/harder-games-report16.json
```

Actions workflow `.github/workflows/report-reactivity.yml` input
`history_mode` (default `minimal`), description:
**Harder games: redundant correct demos**. Passed through to prepare.

## Why this was supposed to unlock ask-first

1. Ceiling on Direct blocks measuring whether self-report *rescues* failures.
2. Redundant correct demos are a controlled difficulty dial (not new lexical
   material, not a new arm set).
3. When Direct drops below ceiling, primary contrasts
   (`self_report − direct`, `self_report − matched_control`) become estimable
   on the same factorial scaffold.
4. Factorial verifier still requires shared scenario across the six arms;
   hashes differ between modes by construction.

## Next bets (after this null)

1. **Mid-trajectory ask-as-intervention** — insert a consequence question
   *between* two forced presses; contrast vs matched non-consequence chatter.
   Protocol: [`ask-mid-trajectory-protocol.md`](ask-mid-trajectory-protocol.md).
   CPU dry-run via Actions `task=ask_mid_trajectory`, `dry_run=true` (not yet
   scored).
2. Stronger difficulty dials only if they stay content-controlled (not new
   lexical material that breaks split hygiene).
3. Do **not** reopen mechanistic “reports fix failures” while Direct stays at
   ceiling on the ask-first scaffold.

## What this note does *not* do

- No Modal / GPU scoring from the documentation PR that records this null.
- Does not rewrite historical `minimal` results or claim-ledger rows C14–C16.
- Incident task ignores `history_mode` (report-only knob; mid-trajectory may
  reuse it).
