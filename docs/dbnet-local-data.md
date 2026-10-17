# DBNet local dataset setup

The dataset is managed locally and is not distributed with this repository. A limited JPEG/LAS sample check does not replace a complete dataset integrity audit.

**[SOURCE]** The original DBNet-2018 README describes `train/<drive>/dvr_66x200/*.jpg`, `points_16384/*.las`, and `behavior.csv`; likewise for `val`. Its `test` split lacks behavioral labels in the documented layout. Filenames are zero-based frame numbers, and the dataset documentation describes CSV row i as mapping to frame i−1; however, inspected CSV matrices can be channel-major (two rows spanning many frames). Detect the actual matrix orientation before indexing labels. Verify the actual extracted directory spelling before running the script.

## Local setup

Install the project: `python -m pip install -e .`

Extract the dataset archives into an ignored local folder, e.g.:

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

The audit detects missing pairs and missing labels and counts drive-level records. It does not decode all JPEG/LAS bytes, establish steering/speed label semantics, or prove prior temporal synchronization.

`build_labeled_observations` returns **raw numeric label tuples** to avoid inventing physical units. Do not feed them into `Observation(steering_deg, speed_kmh)` until official label definitions are verified.

**Citation:** Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, CVPR. Official repository: https://github.com/driving-behavior/DBNet. See `CITATION_DBNet.bib`.

**Licensing:** Check the official dataset terms and obtain any required permissions. Redistribution rights have not been verified. Keep the dataset out of the public GitHub repository for now; code and download instructions are shareable.


## Sensor decoding

After `python -m pip install -e '.[data]'`, inspect a genuine frame pair with `python scripts/inspect_dbnet_pair.py data/dbnet-2018/train/1 0`. See `docs/dbnet-media-pipeline.md`. This verifies one pair and preserves raw label values; it does not infer steering/speed units or LiDAR pixel projection.

## Label matrix orientation

Use `autonomous_driving.label_inspection.inspect_behavior_csv` to compare the label matrix shape with the number of indexed frames. Both `frames_as_rows` (N×channels) and `frames_as_columns` (channels×N) are supported. Ambiguous square matrices must be reviewed explicitly. Label values remain unnamed until the dataset's physical units and channel meanings are verified.
