# Protocol V2 configurations

This folder contains the machine-readable rules for the V2 branch. The JSON
files are meant to be inspected alongside the plain-language documents in
[`../../docs/v2/`](../../docs/v2/README.md).

| File | What it controls |
|---|---|
| [`workspace_mapping.json`](workspace_mapping.json) | The neutral prompts and rule used to select the middle-layer band. |
| [`flexible_generalization_smoke.json`](flexible_generalization_smoke.json) | The released flexible-generalization control task used by H0. |
| [`h0r_diagnostic.json`](h0r_diagnostic.json) | The exploratory burned-data sweep. |
| [`h0r_candidate_protocol.json`](h0r_candidate_protocol.json) | The one protocol frozen before fresh H0R-C controls were opened. |
| [`h0r_locked_controls.json`](h0r_locked_controls.json) | The original computed-intermediate control definition. |
| [`h0r_locked_controls_v2.json`](h0r_locked_controls_v2.json) | The replacement fresh argument corpus used by H0R-C. |
| [`h0r_validation.json`](h0r_validation.json) | H0R validation-run settings and gates. |
| Other JSON files | Candidate tasks and supporting V2 control definitions. |

The original H0 and H0R decisions are documented in [`../../docs/v2/decision-log.md`](../../docs/v2/decision-log.md).

[Back to the configuration guide](../README.md)
