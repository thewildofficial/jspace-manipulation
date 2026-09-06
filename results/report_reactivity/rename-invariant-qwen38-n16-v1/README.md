# Rename-invariant qwen38 n16 v1

Status: **attempted — instrument parity gate failed; no behavioral scores**.

This directory is a durable pointer for the historical engineering attempt
after PR #23. PR #24 now blocks another paid rename run until the analyzer and
results layout are unlocked.
The result is not a rename-following or consequence-following finding.

| Field | Value |
|---|---|
| Protocol | `rename_invariant` |
| Historical study alias | `RENAME-INVARIANT-1` |
| Model | `Qwen/Qwen3.8-27B` |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Run ID | `rename-invariant-qwen38-n16-v1` |
| Payload SHA256 | `5472740319d43c9b2788d09a85e67bfabee8f1e16f4d9ff94ecfca89e6cbd21b` |
| Records / queries | 128 / 288 |
| Replay parity | 0.0; passed |
| Batch/singleton parity | 0.375; failed (threshold 0.25) |
| Choice agreement | true |
| Scores | empty; withheld by the gate |
| Reserved ceiling | approximately $0.783 |

## Run trail

- [Initial failed prepare, run 34053946547](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34053946547): nonce `DSAVEKA004Q` accidentally contained `SAVE`.
- [Corrected CPU dry-run, run 34054166742](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34054166742): 150 passed, 4 skipped.
- [GPU instrument attempt, run 34054251019](https://github.com/thewildofficial/when-words-override-consequences/actions/runs/34054251019): model loaded, parity failed, no scores.

The first two events validate the prepare fix. The third validates that the
existing parity gate withheld unusable behavioral data. A future scored run
needs a fresh run ID and a passing parity gate; it must not overwrite this
record.
