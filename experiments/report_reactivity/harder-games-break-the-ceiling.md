# Harder games: break the Direct ceiling

**Story.** Current report-reactivity demos use balanced prior trials
(`history_mode=minimal`). Direct / self / control sit at ceiling, so
rewrite-vs-reveal rescue is unidentifiable. Inverse-evidence work in this repo
already showed that **redundant correct history** can worsen wordy action —
enough signal to unlock ask-first / self-report contrasts when Direct fails.

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

## Why this unlocks the ask-first question

1. Ceiling on Direct blocks measuring whether self-report *rescues* failures.
2. Redundant correct demos are a controlled difficulty dial (not new lexical
   material, not a new arm set).
3. When Direct drops below ceiling, primary contrasts
   (`self_report − direct`, `self_report − matched_control`) become estimable
   on the same factorial scaffold.
4. Factorial verifier still requires shared scenario across the six arms;
   hashes differ between modes by construction.

## What this note does *not* do

- No Modal / GPU scoring here. Parent dry-runs then scores after merge.
- Does not rewrite historical `minimal` results or claim-ledger rows.
- Incident task ignores `history_mode` (report-only knob).
