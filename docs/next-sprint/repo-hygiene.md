# Repository hygiene note

This cleanup keeps the report-reactivity evidence while removing repository
noise from the exploratory Cursor merge sequence.

## Kept, but stored compactly

The three scored CPU payloads are exact provenance artifacts, not disposable
output. They are now gzip-compressed as `prepared.json.gz`; the shared loader
accepts both ordinary JSON and gzip JSON, and future Modal runs write the
compressed form without replacing an existing file. Their payload hashes are
unchanged:

| Run | Payload sha256 |
|---|---|
| `gha-report16-38-v1` | `126ea05173558cd161f017922a936c8248704d2ac35dd34c3213b1de07bf257d` |
| `harder-games-qwen38-n16-v1` | `a8b43df94451398b4edfecddb1a3f1c821c014e1d4212d6cc207be2dd077964c` |
| `ask-mid-traj-qwen38-n16-v1` | `50b854ae18ab584a25234e62687de4d4f6de8c467d5656ec9dda89a0b1d7a5d6` |

Raw scores, manifests, immutable analyses, spend ledgers, methods notes, and
the protocols for scored work remain in place. The archives can be inspected
with `gzip -dc path/to/prepared.json.gz`.

## Removed from the active tree

- `gha_gates.py`, an unused re-export shim; imports point to the tested source
  module `jspace_policy.report_reactivity_gates`.
- The unexecuted ask-after-the-act design stub and the unexecuted branched
  pre-action auditor draft/config. The decision log records that these ideas
  are deferred; their original versions remain available in Git history if
  either line of work is reopened.

The cleanup deliberately does not revert the workflow, scoring fixes, gate
logic, actual discovery results, or the rename-invariant safety lock.
