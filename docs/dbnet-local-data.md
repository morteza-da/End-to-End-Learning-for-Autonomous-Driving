# Using the researcher's existing DBNet archives

**[USER-PROVIDED]** Train, validation and test RAR archives have already been uploaded to the user's ChatGPT file Library. **[PREVIOUS PROBE]** One JPEG/LAS pair from each split was decoded; the whole collection has not been audited. Do not ask the researcher to re-upload the archives.

**[SOURCE]** The original DBNet-2018 README describes `train/<drive>/dvr_66x200/*.jpg`, `points_16384/*.las`, and `behavior.csv`; likewise for `val`. Its `test` split lacks behavioral labels in the documented layout. Filenames are zero-based frame numbers, and CSV row i maps to `i-1.jpg` and `i-1.las`. Verify the actual extracted directory spelling before running the script.

## Local setup

Install the project: `python -m pip install -e .`

Extract the already-owned archives into an ignored local folder, e.g.:

```text
data/dbnet-2018/
  train/1/dvr_66x200/0.jpg
  train/1/points_16384/0.las
  train/1/behavior.csv
  val/1/...
  test/1/...
```

Do not extract untrusted archives without checking paths and size limits. Archive-to-split mapping must be confirmed from actual archive contents, not inferred from suffixes `(1)` or `(2)`.

Run: `python scripts/audit_dbnet.py data/dbnet-2018 --output results/local_dbnet_audit.json`.

The audit detects missing pairs and missing labels and counts drive-level records. It does not decode all JPEG/LAS bytes, establish steering/speed label semantics, or prove historical temporal synchronization.

`build_labeled_observations` returns **raw label columns** to avoid inventing physical units. Do not feed them into `Observation(steering_deg, speed_kmh)` until official label definitions are verified.

**Citation:** Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, CVPR. Official repository: https://github.com/driving-behavior/DBNet. See `CITATION_DBNet.bib`.

**Licensing:** The researcher reports email permission from DBNet's authors to use the dataset. The exact redistribution terms remain unverified pending the original email. Keep the dataset out of the public GitHub repository for now; code and download instructions are shareable.
