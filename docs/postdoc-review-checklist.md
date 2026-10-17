# Research repository review checklist

This checklist separates software readiness from scientific validation.

## Public presentation
- [x] Clear camera–LiDAR, CNN–LSTM–attention project scope and author links in README.
- [x] MIT code license and separate dataset attribution.
- [x] Installation, unit tests, preflight, and training commands documented.
- [x] Raw local data and environment secrets excluded from future commits via gitignore.
- [x] CI tracked-file path policy rejects common raw sensor, model-weight, and credential filenames.
- [x] Inspect current main-branch file tree for large binaries and common raw-data formats (no such files found in the inspected tree; placeholder .gitkeep files remain).
- [ ] Review historical commits for previously tracked sensitive or restricted file **contents**; automated history filename scanning is a preliminary check only.

## Reproducible software
- [x] Automated Python tests and CPU model integration workflow, with live badges in README.
- [x] Training CLI supports explicit temporal stride to match the preflight report.
- [x] Synthetic CLI preflight and train-gate acceptance/rejection checks.
- [x] Explicit target mapping, provenance requirements, and label plausibility checks.
- [x] Optional sensor-byte SHA-256 integrity check.
- [x] A/B backbone assignments and separately labeled pooled fusion variants.
- [ ] Execute the complete training runner on real, authorized DBNet files.
- [ ] Confirm reproducibility on a second machine/environment.

## Scientific validation
- [ ] Confirm actual dataset version, label units, evidence and redistribution rights.
- [ ] Confirm physical synchronization independently; frame-index pairing is not proof.
- [ ] Report real train/validation/test counts and drive-disjoint splits.
- [ ] Measure RMSE and R² on held-out data, separately for steering and speed.
- [ ] Report steering ±6° and speed ±5 km/h threshold accuracy separately.
- [ ] Compare flatten baseline with pooled variants using the same held-out windows.
- [ ] Preserve measured configs, software versions, checkpoint identifiers and run artifacts.

**Interpretation:** A checked item indicates repository implementation or test coverage, not that a full DBNet benchmark has been completed. Automated tests may pass without validating the scientific performance of a trained model.

## Public tree inspection

The current main-branch tree was inspected via the GitHub Git Trees API (recursive, non-truncated). It contained 135 tree entries at inspection time. No tracked blobs above 150 KB or files ending in .las, .laz, .jpg, .jpeg, .png, .csv, .pt, .pth, .h5, .env, or .zip were identified by that filename/size scan. This is **not** a comprehensive secrets scan, file-content audit, or review of previous commits. In particular, tracked files with other extensions and Git history still need separate review.
