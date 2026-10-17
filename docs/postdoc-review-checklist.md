# Research repository review checklist

This checklist separates software readiness from scientific validation.

## Final handoff criteria

The public repository can be shared as a **tested research-software demonstration** when its latest Python and CPU integration workflows pass, README links resolve, and the source/data boundary is clear. It must not be presented as a completed DBNet benchmark until real-data experiments and independent scientific review are finished.

Before sending the link to a prospective collaborator, check the current Actions status, confirm the latest README and license, and perform a manual review of Git history **contents** for secrets and restricted material. Automated filename and secret-pattern checks cannot certify this.

## Automated release gate — 2026-10-17

At commit `6a70c1b73d2c3b622eb03dcecb1e1cdcd7606574`, both GitHub Actions checks completed successfully: Python unit tests (run 39060882527) and PyTorch CPU integration (run 39060882495). These runs validate the automated software checks, not real-data benchmark results or a complete manual content audit. Any subsequent change requires a new CI check.

## Public presentation
- [x] Clear camera–LiDAR, CNN–LSTM–attention project scope and author links in README.
- [x] MIT code license and separate dataset attribution.
- [x] Installation, unit tests, preflight, and training commands documented.
- [x] README local links validated automatically by CI.
- [x] Raw local data and environment secrets excluded from future commits via gitignore.
- [x] CI tracked-file path policy rejects common raw sensor, model-weight, and credential filenames.
- [x] CI scans filenames in reachable Git history and applies conservative secret-pattern rules to reachable blobs (skipping blobs over 2 MB and known synthetic scanner-test fixture blobs; not a comprehensive secret audit).
- [x] Inspect current main-branch file tree for large binaries and common raw-data formats (no such files found in the inspected tree; placeholder .gitkeep files remain).
- [ ] Manually review historical commits for previously tracked sensitive or restricted file **contents**, dataset rights, and secrets not covered by heuristic pattern matching.

## Reproducible software
- [x] Automated Python tests and CPU model integration workflow, with live badges in README.
- [x] Training output safety checks prevent writing into raw data and accidental overwrite of existing results.
- [x] Training metadata records Git revision, tracked worktree status, Python and key package versions.
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

## Historical review sampling — 2026-10-17

- Enumerated 342 commits through the GitHub commits API (four pages). This is a commit-list review, **not** a full manual inspection of all historical blob contents.
- Examined changes in two early `data/src` folder-deletion commits: both removed `.gitkeep` placeholders, not raw sensor files.
- Inspected recursive Git trees at the initial commit and two intermediate commits; those three snapshots had no filenames with the checked raw-data/checkpoint extensions and no blobs larger than 150 KB.
- Inspected the change history of an older research document: it was removed as part of a documentation restructuring, and is not present in the current tree. Its historical text should not be treated as independently verified dataset evidence.
- Automated history secret-pattern checks passed in the verified Python CI run. These findings do **not** complete the pending manual review of every historical content blob or dataset redistribution rights.

## Public tree inspection

The main-branch tree was re-inspected via the GitHub Git Trees API (recursive, non-truncated) on 2026-10-17. The latest scan found 137 tracked blobs, none over 150 KB, and no filenames matching the common raw sensor, checkpoint, archive, or credential extensions checked below. No tracked blobs above 150 KB or files ending in .las, .laz, .jpg, .jpeg, .png, .csv, .pt, .pth, .ckpt, .h5, .env, or .zip were identified by that filename/size scan. This is **not** a comprehensive secrets scan, file-content audit, or review of previous commits. In particular, tracked files with other extensions and Git history still need separate review.
