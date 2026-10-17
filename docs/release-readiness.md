# Public research repository — release readiness

## Software verification

The repository provides an installable Python package, dataset inspection tools, a camera–LiDAR CNN–LSTM–attention implementation, configurable backbone assignments, optional spatial pooling, experiment reporting, and reproducibility documentation.

Automated GitHub Actions cover Python unit tests, synthetic preflight CLI and training-gate checks, CPU model integration, tracked-file and reachable-history filename/path policies, and local README link validation. Workflow badges and run histories in the README are the authoritative current CI status; this document does not claim that future runs will always pass.

The README's internal links were checked against the main-branch tree: all 24 relative links resolved during the initial inspection. CI now checks local README links on subsequent changes.

## What a reviewer can verify today

1. Read the architecture and research methods linked from README.
2. Install the package and run `python -m unittest discover -s tests -v`.
3. Inspect the GitHub Actions test logs, synthetic fixtures, and model smoke tests.
4. Review the label evidence requirements, preflight integrity gate, and baseline-vs-pooled experiment protocol.
5. Review the dataset license separately before obtaining or sharing DBNet data.

## Explicit scientific boundaries

- The existing automated fixtures are synthetic; they are not a real DBNet benchmark.
- No measured DBNet RMSE, R², or threshold accuracy is asserted.
- Frame-index-paired camera and LiDAR files are not proof of physical timestamp synchronization.
- Label channel/unit evidence and dataset rights require independent review.
- The baseline spatial Flatten model can have a large memory footprint; pooled variants must be reported separately.
- Current-tree and reachable-Git-history **filename** checks do not inspect secrets embedded in file contents, unreachable Git objects, or third-party licensing compliance.

## Remaining work before a scientific results release

- Obtain authorized DBNet files and review their version, structure, target mapping, and provenance.
- Run complete real-data preflight and training, with enough compute and memory.
- Preserve actual configs, checkpoints, validation/test partitions, environment and seed.
- Publish measured RMSE and R² for steering and speed, with ±6° and ±5 km/h threshold accuracies and fair ablations.
- Independently inspect historical Git commits for sensitive or restricted material.

**Release interpretation:** The repository is suitable for demonstrating tested research software and an experimental plan to potential collaborators. It is not yet evidence of a completed, independently reproduced DBNet performance study.
