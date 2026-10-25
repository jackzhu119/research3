# Paper3 repository guidance

Use the school Slurm environment and each round's documented effective resource configuration.

## Default result delivery for every subsequent round

Follow [the result upload policy](docs/RESULT_UPLOAD_POLICY_ZH.md) and its [machine-readable settings](docs/RESULT_UPLOAD_POLICY.json) when designing a new round or revising a launch/recovery implementation.

Implement two independent channels:
1. Publish paper analysis data first: complete reports/CSV, compact participant/session metrics and pairing identities, configuration/provenance, and existing statistical evidence. Split large tables without truncating rows. Generate compact metrics alongside reports.
2. Upload the round's complete declared scientific output slowly in the background, including prediction files, validation history and model/optimizer checkpoints, with manifests and hashes.

Both channels are independent of training; preserve upload progress and retry network failures without stopping training or rerunning completed training/tests. Keep paper-data, summary, scientific completion and full-archive receipts distinct. Missing or partial data must be labelled and must not become a new training startup gate.

For analysis and manuscript writing, read the paper-data index and tables first; request a targeted additional export when new statistics need more detail. Do not require full archives for every analysis.

This policy records future implementation requirements. Do not claim currently running rounds were retrofitted merely because these files were added; honor the user's requested change scope.
