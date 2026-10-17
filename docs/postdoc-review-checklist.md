# Research repository review checklist

This checklist separates software readiness from scientific validation.

## Public presentation
- [x] Clear camera–LiDAR, CNN–LSTM–attention project scope and author links in README.
- [x] MIT code license and separate dataset attribution.
- [x] Installation, unit tests, preflight, and training commands documented.
- [x] Raw local data and environment secrets excluded from future commits via gitignore.
- [ ] Manually verify no sensitive or third-party restricted files were tracked in earlier commits.

## Reproducible software
- [x] Automated Python tests and CPU model integration workflow.
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
