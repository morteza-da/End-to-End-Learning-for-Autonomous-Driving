# Experimental Reproducibility Checklist

**Status:** Reconstruction planning. No experiments in this checklist should be considered completed unless accompanied by verifiable artifacts.

## 1. Dataset provenance
- [ ] Identify dataset version, source, collection permissions, and redistribution limits.
- [ ] Inventory camera frames, LiDAR files, labels, timestamps, and calibration.
- [ ] Record the exact definition and units of steering and speed targets.
- [ ] Confirm whether each observation has a matching sensor pair.
- [ ] Document missing/corrupted records and exclusion criteria.

## 2. Splits and leakage prevention
- [ ] Define independent train, validation, and test partitions.
- [ ] Check for adjacent-frame, route, driver, and session overlap.
- [ ] Save immutable manifests with sample identifiers and checksums.
- [ ] Apply preprocessing statistics fitted on training data only.

## 3. Model specification
- [ ] Verify input dimensions and LiDAR representation against dissertation materials.
- [ ] Verify how Xception and ResNeXt-50 are assigned and initialized.
- [ ] Specify fusion operator and temporal sequence length.
- [ ] Specify LSTM and attention configurations.
- [ ] Specify steering/speed heads, scaling, and loss weights.
- [ ] Document MC-dropout placement and inference protocol.

## 4. Training protocol
- [ ] Record Python/framework versions and hardware.
- [ ] Pin dependencies and deterministic seeds where feasible.
- [ ] Record batch size, optimizer, learning rate schedule, epochs, and checkpoints.
- [ ] Save experiment configs and training logs.
- [ ] Record data augmentation and early-stopping rules.

## 5. Evaluation
- [ ] Report steering and speed metrics separately: MAE, RMSE, and R².
- [ ] Record units, aggregation rules, and sample counts.
- [ ] Evaluate on a held-out test set untouched during tuning.
- [ ] Compare camera-only, LiDAR-only, and fused baselines.
- [ ] Provide temporal plots, scenario-level analyses, and uncertainty calibration where applicable.
- [ ] Clearly distinguish dissertation-reported results from new reproduction runs.

## 6. Publication and release
- [ ] Verify dissertation metadata, journal citations, and DOI links.
- [ ] Add a model card and dataset card.
- [ ] Add tests and a minimal reproducible example with clearly labeled synthetic fixtures.
- [ ] Check privacy, licenses, and third-party dataset restrictions.
- [ ] Document limitations and out-of-distribution behavior.

## Evidence record template

For each published metric, include:

| Field | Required value |
| --- | --- |
| Source | Dissertation table/figure or new experiment ID |
| Dataset version | Exact version and provenance |
| Split | Saved manifest identifier |
| Target | Steering or speed, including units |
| Metric | Definition and aggregation |
| Result | Measured value, never a placeholder |
| Code revision | Git commit SHA |
| Configuration | Config path and random seed |
| Artifacts | Logs, predictions, and figures |

## Current boundary

The historical codebase was lost. Architecture notes are research documentation, not proof of successful execution. Until datasets and methodology are verified, no benchmark claims or runnable training instructions should be inferred from this repository.
