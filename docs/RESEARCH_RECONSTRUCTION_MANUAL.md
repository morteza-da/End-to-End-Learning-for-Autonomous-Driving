# Research Reconstruction Manual — Master Archive

> **Living master document** for reconstructing the doctoral research on multimodal end-to-end autonomous driving. This is **not** the original dissertation or a claim that lost code has been recovered. It is an evidence-labelled, reproducible reconstruction.

## Evidence legend
- **[THESIS]** Directly documented in the supplied 156-page Persian dissertation; cite PDF page/table.
- **[RESEARCHER]** Explicitly confirmed by the researcher.
- **[RECONSTRUCTION]** New implementation choice, not verified as the historical choice.
- **[OPEN]** Unknown or not yet validated.
- **[VERIFIED-RUN]** Actual code execution or measured experiment, with environment and artifact recorded.

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
15. Results audit: historical tables vs newly reproduced outcomes
16. Full reimplementation checklist and long-term archival
17. References, dataset licenses, dependency versions, and change log

---

## Chapter 1 — Research objective and provenance

**[THESIS/RESEARCHER]** Multimodal camera + LiDAR research on predicting steering angle and speed simultaneously. The researcher reports that the original source code was lost; this repository therefore reconstructs rather than recovers the implementation.

**[RESEARCHER]** Both backbones were initialized with ImageNet pretrained weights; early layers frozen and upper layers fine-tuned using DBNet. Each modality was represented as a 299×299 RGB image. Backbone feature maps were Flattened separately, Concatenated without a Dense bottleneck, and processed as multiple consecutive frames by LSTM → Attention → Dense.

**[OPEN]** Exact original training framework, sequence length, rasterization, preprocessing normalization, attention type, optimizer hyperparameters, and output-head architecture.

## Chapter 2 — Environment

**[RECONSTRUCTION]** Python 3.10+ and standard-library tests are currently available. Notebook files use the Jupyter `ipynb` format. A **new reconstruction implementation** now uses PyTorch + timm within a Jupyter Notebook. This is **not** confirmation of the historical research framework. Dependency versions still require pinning after compatibility and execution tests. A notebook is an interface, not a replacement for a training framework.

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

**[RESEARCHER]** DBNet is the research dataset, with camera frames, LiDAR information, steering angle, and speed.

**[THESIS]** PDF pp. 105–106 describe an 80% training / 10% validation / 10% test division.

**[OPEN]** Dataset download/permission, precise schema, sample identifiers, synchronization, temporal split leakage prevention, sequence sampling, and normalization.

**[SOURCE ATTRIBUTION]** DBNet is credited to Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, IEEE/CVF CVPR 2018. Official repository: https://github.com/driving-behavior/DBNet. Preserve this citation in every DBNet-based notebook, result, paper, and figure. The repository states its **code** is Apache 2.0; original dataset rights must be checked separately before redistribution. Full original-author BibTeX: `CITATION_DBNet.bib`; details: `docs/dbnet-attribution.md`.\n\nSee `docs/dataset-manifest.md` and `src/autonomous_driving/data_manifest.py`. Do not invent or distribute unlicensed dataset records.

## Chapter 4 — Synchronized multimodal inputs

**[RESEARCHER]** RGB camera frames and 3D LiDAR converted to 2D distance-coded RGB images; each resized to 299×299. Both branches refer to the same ordered observation sequence.

**[OPEN]** Timestamp alignment tolerance, interpolation, missing sensor records, camera intrinsic/extrinsic calibration.

## Chapter 5 — LiDAR preprocessing

**[THESIS]** PDF pp. 73–74: planar distance `sqrt(Px² + Py²)`; blue for distance below 20, green for 20–40, red from 40 upward, with scale divisors 20/40/60.

**[RECONSTRUCTION]** `src/autonomous_driving/lidar_encoding.py` implements an explicit piecewise interpretation and round-to-uint8 conversion. It does **not** reconstruct the original 3D-to-2D pixel mapping.

Notebook: `notebooks/02_LiDAR_Preprocessing.ipynb`. Tests: `tests/test_lidar_encoding.py`.

**[OPEN]** Projection, occlusion, pixel collision resolution, empty pixel policy, coordinate units, and exact rounding.

### New executable backbone notebook (2026-10-16)

**[RECONSTRUCTION]** Run `notebooks/03b_Runnable_Backbone_Features.ipynb` after installing `torch` and `timm`. It calls `src/autonomous_driving/backbone_extraction.py` to instantiate timm feature-map backbones, apply an **illustrative** freeze policy, flatten each branch, and concatenate aligned synthetic frame features. Set `PRETRAINED=True` to request ImageNet weights; the notebook defaults to `False` for an initial no-download shape check. Exact model registry names and pretrained weights **must** be checked against the installed timm version. No execution results are archived yet. See `docs/runnable-backbones.md`.

## Chapter 6 — Transfer learning

**[THESIS/RESEARCHER]** Xception and ResNeXt-50 with ImageNet pretrained weights and partial fine-tuning on DBNet.

**[OPEN]** Exact layer freeze boundaries and backbone implementation/version.

## Chapter 7 — Backbone configuration A

**[THESIS]** PDF pp. 105–106: camera → ResNeXt-50; LiDAR → Xception.

Config: `src/autonomous_driving/backbone_config.py`.

## Chapter 8 — Backbone configuration B

**[THESIS]** PDF pp. 105–106: camera → Xception; LiDAR → ResNeXt-50.

Config: `src/autonomous_driving/backbone_config.py`.

## Chapter 9 — Flatten and feature fusion

**[RESEARCHER]** Flatten each branch's feature map, then concatenate feature vectors per synchronized time step. No Dense between fusion and LSTM. See `docs/feature-fusion.md`.

## Chapter 10 — Temporal modelling

**[RESEARCHER]** Multiple consecutive aligned frames, fed to an LSTM.

**[OPEN]** Exact temporal window length, stride, label alignment, hidden size, number of layers, and recurrent dropout.\n\n**[RECONSTRUCTION]** `src/autonomous_driving/temporal_model.py` now provides a batch-first LSTM that directly accepts concatenated features `[B,T,F]`. `stack_temporal_features` constructs these tensors from aligned image sequences using the two spatial-map backbones. Notebook: `notebooks/05_LSTM_Attention_Model.ipynb` (synthetic demonstration only).

## Chapter 11 — Attention and joint outputs

**[RESEARCHER]** LSTM → Attention → Dense, predicting steering angle and speed simultaneously.

**[OPEN]** Original attention scoring formula, layer widths, output head arrangement, and target scaling.\n\n**[RECONSTRUCTION]** A transparent additive temporal attention layer computes weights over LSTM outputs, followed by one two-unit Dense regression layer. This is a new implementation choice, not a historical fact. See `docs/temporal-lstm-attention.md`.

## Chapter 12 — Training

**[THESIS]** PDF p. 107 describes RMSE as the training loss; PDF pp. 105–106 mention Adam.

**[OPEN]** Exact joint loss aggregation, learning rate, batch size, epoch count, early stopping, seed and checkpoint selection.

## Chapter 13 — Evaluation

**[THESIS]** PDF pp. 108–110: RMSE, R² and threshold-based accuracy. Steering accepted when absolute error ≤6 degrees; speed accepted when absolute error ≤5 km/h. See `src/autonomous_driving/metrics.py`, `src/autonomous_driving/threshold_metrics.py` and `docs/threshold-accuracy.md`.

## Chapter 14 — Comparative experiments

**[THESIS]** PDF pp. 100–104 describe candidate experiments; document every experiment's configuration, split, random seed, training budget, and independent test metrics before comparison.

## Chapter 15 — Historical vs new results

**[THESIS]** PDF p. 118 Table 4-3 reports historical threshold-based accuracies of 94.91% steering and 94.09% speed for its best listed model. These are **historical dissertation values**, **not results produced by this repository**.

**[THESIS]** Table 4-4 includes historical RMSE values for different forecast horizons. Do not mistake the dissertation's visualization interval for the LSTM input sequence length.

**[VERIFIED-RUN]** None recorded yet.

## Chapter 16 — Reimplementation and archive checklist

- [x] Establish evidence and source provenance conventions
- [x] Add evaluation metric implementations and synthetic tests
- [x] Add LiDAR color encoding and synthetic tests
- [x] Define validated A/B backbone modality assignments
- [x] Establish notebook structure and master archive
- [ ] Verify CI test runs
- [ ] Obtain/validate legal access to DBNet
- [ ] Confirm 2D LiDAR projection algorithm or label a new one
- [x] Add runnable PyTorch/timm backbone construction and synthetic shape-check notebook (execution not yet verified)
- [ ] Pin verified pretrained backbones and dependency versions
- [x] Add Flatten → Concatenate → LSTM → Attention → Dense code and synthetic notebook (end-to-end execution not yet verified)
- [ ] Verify train/validation/test temporal separation
- [ ] Execute controlled training, evaluation, and ablations
- [ ] Archive seeds, manifests, hyperparameters, logs, plots, checkpoints, and results
- [ ] Export this living manual as a final navigable PDF and DOCX with all code listings, figures, appendices, and an environment lockfile

## Chapter 17 — Change log

- 2026-10-16: Master archival manual established; notebook-first presentation adopted. Existing implementation remains a reconstruction, not a recovered original.

### Related research documents
- `docs/research-architecture.md`
- `docs/dissertation-methodology-notes.md`
- `docs/multimodal-input-pipeline.md`
- `docs/feature-fusion.md`
- `docs/lidar-color-encoding.md`
- `docs/backbone-implementation-plan.md`
- `docs/threshold-accuracy.md`
- `docs/runnable-backbones.md`\n- `docs/dbnet-attribution.md`\n- `docs/temporal-lstm-attention.md`
