# Research Methods and Experiment Protocol

> Technical reference for dataset processing, multimodal model architecture, training, and evaluation.

## Documentation conventions

Configuration choices, dataset definitions, and evaluation metrics are documented alongside executable code. Results are reported only when supported by saved run artifacts.

## Table of contents

1. Research objective, provenance, and scientific integrity
2. Reproducible software environment and repository setup
3. DBNet dataset specification, licensing, provenance, and split manifests
4. Synchronized camera and LiDAR inputs
5. LiDAR point cloud distance encoding and 2D image construction
6. Image preprocessing and ImageNet transfer learning
7. Backbone experiment A: Camera ResNeXt-50 / LiDAR Xception
8. Backbone experiment B: Camera Xception / LiDAR ResNeXt-50
9. Per-frame Flatten and multimodal Concatenation
10. Sequence construction and LSTM temporal modelling
11. Attention and joint steering/speed prediction
12. Training, optimization, checkpoints, and reproducibility
13. Evaluation: RMSE, R², MAE, threshold-based accuracy
14. Comparison experiments, plots, ablations, and uncertainty
15. Results audit: prior tables vs newly reproduced outcomes
16. Full implementation checklist and long-term archival
17. References, dataset licenses, dependency versions, and change log

---

## Chapter 1 — Research objective and provenance

**[research/RESEARCHER]** Multimodal camera + LiDAR research on predicting steering angle and speed simultaneously.  Each modality was represented as a 299×299 RGB image. Backbone feature maps were Flattened separately, Concatenated without a Dense bottleneck, and processed as multiple consecutive frames by LSTM → Attention → Dense.

 Exact original training framework, sequence length, rasterization, preprocessing normalization, attention type, optimizer hyperparameters, and output-head architecture.

## Chapter 2 — Environment

 Python 3.10+ and standard-library tests are currently available. Notebook files use the Jupyter `ipynb` format. A **new implementation implementation** now uses PyTorch + timm within a Jupyter Notebook.  Dependency versions still require pinning after compatibility and execution tests. A notebook is an interface, not a replacement for a training framework.

```bash
git clone https://github.com/morteza-da/End-to-End-Learning-for-Autonomous-Driving.git
cd End-to-End-Learning-for-Autonomous-Driving
python -m venv .venv
# Activate .venv using your OS-specific command.
python -m pip install -e .
python -m unittest discover -s tests -v
# Install Jupyter separately if needed:
python -m pip install notebook
jupyter notebook
```

Do not claim reproducibility until dependencies, datasets, seeds, checkpoints, and execution environment are recorded.

## Chapter 3 — DBNet

 DBNet is the research dataset, with camera frames, LiDAR information, steering angle, and speed.

 PDF pp. 105–106 describe an 80% training / 10% validation / 10% test division.

 The researcher previously uploaded DBNet archive files to their ChatGPT file Library, including archives representing train, validation, and test. A previous exploratory validation report confirms decoding one JPEG/LAS pair from one drive in each split; this does NOT establish full archive integrity. The supplied DBNet README describes per-drive ordered frames and matching image/LAS integer names, and reports train/val/test organization. **Do not imply the user still needs to upload the dataset.** These Library files are not automatically present in GitHub or in a remote training environment; a local data-access workflow is still required.

 Complete archive integrity, exact label-column units, full manifest counts, verified sensor synchronization beyond matching names, and training environment access.

 `src/autonomous_driving/sequence_manifest.py` validates a generic paired-frame manifest and generates contiguous windows within drive-disjoint splits. This is a defensive new choice, not a verified prior DBNet schema or split method. See `docs/sequence-training.md`.

 DBNet is credited to Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, IEEE/CVF CVPR 2018. Official repository: https://github.com/driving-behavior/DBNet. Preserve this citation in every DBNet-based notebook, result, paper, and figure. The repository states its **code** is Apache 2.0; original dataset rights must be checked separately before redistribution. Full original-author BibTeX: `CITATION_DBNet.bib`; details: `docs/dbnet-attribution.md`.

See `docs/dataset-manifest.md` and `src/autonomous_driving/data_manifest.py`. Do not invent or distribute unlicensed dataset records.

## Chapter 4 — Synchronized multimodal inputs

 RGB camera frames and 3D LiDAR converted to 2D distance-coded RGB images; each resized to 299×299. Both branches refer to the same ordered observation sequence.

 Timestamp alignment tolerance, interpolation, missing sensor records, camera intrinsic/extrinsic calibration.

## Chapter 5 — LiDAR preprocessing

 PDF pp. 73–74: planar distance `sqrt(Px² + Py²)`; blue for distance below 20, green for 20–40, red from 40 upward, with scale divisors 20/40/60.

 `src/autonomous_driving/lidar_encoding.py` implements an explicit piecewise interpretation and round-to-uint8 conversion. It does **not** reconstruct the original 3D-to-2D pixel mapping.

Notebook: `notebooks/02_LiDAR_Preprocessing.ipynb`. Tests: `tests/test_lidar_encoding.py`.

 Projection, occlusion, pixel collision resolution, empty pixel policy, coordinate units, and exact rounding.

### New executable backbone notebook (2026-10-16)

 Run `notebooks/03b_Runnable_Backbone_Features.ipynb` after installing `torch` and `timm`. It calls `src/autonomous_driving/backbone_extraction.py` to instantiate timm feature-map backbones, apply an **illustrative** freeze policy, flatten each branch, and concatenate aligned synthetic frame features. Set `PRETRAINED=True` to request ImageNet weights; the notebook defaults to `False` for an initial no-download shape check. Exact model registry names and pretrained weights **must** be checked against the installed timm version. No execution results are archived yet. See `docs/runnable-backbones.md`.

## Chapter 6 — Transfer learning

**[research/RESEARCHER]** Xception and ResNeXt-50 with ImageNet pretrained weights and partial fine-tuning on DBNet.

 Exact layer freeze boundaries and backbone implementation/version.

## Chapter 7 — Backbone configuration A

 PDF pp. 105–106: camera → ResNeXt-50; LiDAR → Xception.

Config: `src/autonomous_driving/backbone_config.py`.

## Chapter 8 — Backbone configuration B

 PDF pp. 105–106: camera → Xception; LiDAR → ResNeXt-50.

Config: `src/autonomous_driving/backbone_config.py`.

## Chapter 9 — Flatten and feature fusion

 Flatten each branch's feature map, then concatenate feature vectors per synchronized time step. No Dense between fusion and LSTM. See `docs/feature-fusion.md`.

## Chapter 10 — Temporal modelling

 Multiple consecutive aligned frames, fed to an LSTM.

 Exact temporal window length, stride, label alignment, hidden size, number of layers, and recurrent dropout.

 `src/autonomous_driving/temporal_model.py` now provides a batch-first LSTM that directly accepts concatenated features `[B,T,F]`. `stack_temporal_features` constructs these tensors from aligned image sequences using the two spatial-map backbones. Notebook: `notebooks/05_LSTM_Attention_Model.ipynb` (synthetic demonstration only).

## Chapter 11 — Attention and joint outputs

 LSTM → Attention → Dense, predicting steering angle and speed simultaneously.

 Original attention scoring formula, layer widths, output head arrangement, and target scaling.

 A transparent additive temporal attention layer computes weights over LSTM outputs, followed by one two-unit Dense regression layer. This is a new implementation choice, not a prior fact. See `docs/temporal-lstm-attention.md`.

## Chapter 12 — Training

 PDF p. 107 describes RMSE as the training loss; PDF pp. 105–106 mention Adam.

 Exact prior joint loss aggregation, learning rate, batch size, epoch count, early stopping, seed and checkpoint selection.

 `src/autonomous_driving/training.py` implements a configurable weighted mean of separate steering and speed RMSE and an epoch helper for **precomputed fused features only**. `notebooks/06_Sequence_Training.ipynb` demonstrates it on synthetic data; this is **not** end-to-end backbone fine-tuning or a DBNet result.

## Chapter 13 — Evaluation

 PDF pp. 108–110: RMSE, R² and threshold-based accuracy. Steering accepted when absolute error ≤6 degrees; speed accepted when absolute error ≤5 km/h. See `src/autonomous_driving/metrics.py`, `src/autonomous_driving/threshold_metrics.py` and `docs/threshold-accuracy.md`.

## Chapter 14 — Comparative experiments

 PDF pp. 100–104 describe candidate experiments; document every experiment's configuration, split, random seed, training budget, and independent test metrics before comparison.

## Chapter 15 — prior vs new results

 PDF p. 118 Table 4-3 reports prior threshold-based accuracies of 94.91% steering and 94.09% speed for its best listed model. These are **prior research values**, **not results produced by this repository**.

 Table 4-4 includes prior RMSE values for different forecast horizons. Do not mistake the project's visualization interval for the LSTM input sequence length.

 None recorded yet.

## Chapter 16 — implementation and archive checklist

- [x] Establish evidence and source provenance conventions
- [x] Add evaluation metric implementations and synthetic tests
- [x] Add LiDAR color encoding and synthetic tests
- [x] Define validated A/B backbone modality assignments
- [x] Establish notebook structure and master archive
- [ ] Verify CI test runs
- [x] Locate previously uploaded DBNet train/validation/test archives in the user's file Library
- [ ] Complete integrity audit of all supplied archives and stage data for training (do not upload raw data publicly)
- [ ] Confirm 2D LiDAR projection algorithm or label a new one
- [x] Add runnable PyTorch/timm backbone construction and synthetic shape-check notebook (execution not yet verified)
- [ ] Pin verified pretrained backbones and dependency versions
- [x] Add Flatten → Concatenate → LSTM → Attention → Dense code and synthetic notebook (end-to-end execution not yet verified)
- [x] Add generic drive-disjoint split checks and contiguous sequence-window code (synthetic only)
- [x] Add a synthetic temporal-head training and validation notebook (not full end-to-end training)
- [ ] Verify train/validation/test temporal separation on real DBNet
- [ ] Execute controlled training, evaluation, and ablations
- [ ] Archive seeds, manifests, hyperparameters, logs, plots, checkpoints, and results
- [ ] Export this living manual as a final navigable PDF and DOCX with all code listings, figures, appendices, and an environment lockfile

## Chapter 17 — Change log

- 2026-10-16: Master archival manual established; notebook-first presentation adopted. Existing implementation remains a implementation, not a recovered original.

### Related research documents
- `docs/research-architecture.md`
- `docs/research-methodology-notes.md`
- `docs/multimodal-input-pipeline.md`
- `docs/feature-fusion.md`
- `docs/lidar-color-encoding.md`
- `docs/backbone-implementation-plan.md`
- `docs/threshold-accuracy.md`
- `docs/runnable-backbones.md`
- `docs/dbnet-attribution.md`
- `docs/temporal-lstm-attention.md`
- `docs/sequence-training.md`


## DBNet original-data access and executable layout audit (October 2026)

**[USER-PROVIDED]** The researcher reports receiving dataset-use permission from the DBNet authors by email. The precise redistribution wording remains to be reviewed, but this does not prevent writing or privately executing dataset-aware code. The original train/val/test archives were previously uploaded to the researcher's ChatGPT file Library. They do not need to be uploaded again.

 `src/autonomous_driving/dbnet_layout.py` and `scripts/audit_dbnet.py` now recognize the documented extracted DBNet directory structure, pair JPEG/LAS by numeric index, preserve positional `behavior.csv` values without assuming units, and generate per-drive/split audit summaries. The test split may have no `behavior.csv`, per the source README. `docs/dbnet-local-data.md` describes safe local extraction and audit.  Archive-wide decoding, correct label semantics, actual environment staging, and verified full-data training.


## DBNet sensor decoding — implemented implementation

 `src/autonomous_driving/dbnet_media.py` and `scripts/inspect_dbnet_pair.py` now decode original DBNet JPEG frames and LAS point clouds, using Pillow and laspy optional dependencies. Camera images are resized to the researcher's confirmed 299×299 backbone input; the precise resizing interpolation is a new implementation choice. LAS XYZ is decoded with header scales/offsets. Raw behavior CSV fields remain positional and uninterpreted pending unit verification.  3D-to-2D LiDAR pixel projection, full sensor pair validation, and actual real-data training. See `docs/dbnet-media-pipeline.md`.
