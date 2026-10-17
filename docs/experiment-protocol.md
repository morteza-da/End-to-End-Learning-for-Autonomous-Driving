# Reproducible experiment configuration and reporting

This project supports two modality assignments:

| Configuration | Camera branch | LiDAR branch |
| --- | --- | --- |
| A | ResNeXt-50 | Xception |
| B | Xception | ResNeXt-50 |

Each branch uses spatial feature maps → Flatten. Features are concatenated directly per frame before LSTM → temporal attention → joint steering and speed outputs.

## Record configurations

```python
from autonomous_driving.experiment_config import (
    ExperimentConfig, save_experiment_config
)
from autonomous_driving.experiment_reporting import seed_everything

config = ExperimentConfig(
    assignment="A",
    sequence_length=4,
    batch_size=1,
    learning_rate=1e-4,
    epochs=10,
    seed=42,
)
seed_everything(config.seed)
save_experiment_config(config, "results/run_a/config.json")
```

These are **configurable starting values**, not experimentally selected hyperparameters or evidence of training completion. Record all choices before model fitting. Use drive-disjoint train/validation/test splits; do not tune on test data. Confirm label channels and units before running supervised learning.

## Record measured evaluation results

```python
from autonomous_driving.experiment_reporting import (
    evaluate_driving_predictions, write_experiment_report
)

# Supply arrays collected from a real evaluation pass.
metrics = evaluate_driving_predictions(
    steering_true, steering_pred, speed_true, speed_pred
)
write_experiment_report(
    "results/run_a/validation_metrics.json",
    config=config,
    metrics=metrics,
    dataset_id="DBNet-2018",
    split="val",
    checkpoint="models/run_a.pt",
    git_commit="<actual Git commit SHA>",
)
```

The evaluation utility computes RMSE and R², and threshold accuracy at steering ±6° and speed ±5 km/h. The report stores the supplied metrics and experiment metadata; it does not run inference or invent results. R² is undefined for constant ground truth and may be NaN; the report writer rejects nonfinite metrics, so a constant-target evaluation requires an explicit reporting policy.

## Reproducibility checklist

- Record dataset source/version, train/validation/test drive IDs, and audited sample counts.
- Confirm channel order, steering/speed units, and any label filtering.
- Record LiDAR angular projection, collision policy, image normalization, and sensor alignment assumptions.
- Record timm/PyTorch versions, pretrained checkpoint identities, frozen/trainable module names, random seed, optimizer, and hardware.
- Save real checkpoint paths, experiment config, and validation/test metrics for each backbone assignment.
- Do not claim identical GPU determinism solely from setting a random seed.
- Never commit restricted DBNet data, private correspondence, or large model checkpoints without confirming redistribution rights.

Unit tests exercise config serialization, input validation, synthetic metric calculations, and report formatting. They are not substitutes for DBNet benchmark evaluation.

## Training runner and checkpoint selection

After confirming the dataset's label channels and physical units, use `scripts/train_multimodal.py`. Both label-channel flags and unit flags are mandatory; the script cannot infer them. The upstream DBNet provider establishes steering channel 0 in degrees; speed channel 1 is corroborated in km/h by secondary literature. Consult [the evidence record](dbnet-label-evidence.md) and verify the actual dataset before training. Both a reviewed evidence JSON and a passing matching preflight report are mandatory:

```bash
python scripts/audit_dbnet_preflight.py --data-root data/dbnet-2018 --label-provenance path/to/reviewed-evidence.json --steering-channel 0 --speed-channel 1 --steering-unit deg --speed-unit km/h
python scripts/train_multimodal.py --data-root data/dbnet-2018 --label-provenance path/to/reviewed-evidence.json --preflight-report results/dbnet-preflight.json --assignment A --steering-channel 0 --speed-channel 1 --steering-unit deg --speed-unit km/h --pool-size 2
python scripts/train_multimodal.py --data-root data/dbnet-2018 --label-provenance path/to/reviewed-evidence.json --preflight-report results/dbnet-preflight.json --assignment B --steering-channel 0 --speed-channel 1 --steering-unit deg --speed-unit km/h --pool-size 2
```

The optional pooling is a separate experimental variant; omit `--pool-size` for direct Flatten. A/B output folders are organized by assignment and fusion variant; for example `results/multimodal/A/pool-2x2/validation_metrics.json`. For matched validation comparisons use `scripts/compare_fusion_variants.py` with compatible reports.

The runner saves `config.json`, `run_metadata.json`, `latest_metrics.json`, and the best validation checkpoint `best.pt` in separate A/B folders. It reports batch-averaged joint training/validation objectives, then reloads the best checkpoint to produce `validation_metrics.json` (aggregate RMSE, R², and threshold accuracy) and `validation_predictions.json` (per-window targets and predictions). It does **not** evaluate the unlabeled DBNet test split. These commands have not been validated on the complete dataset. Full-resolution flattened CNN maps can require substantial GPU memory; inspect `feature_width` and memory requirements before training.

## Memory preflight for flattened CNN features

Before allocating the LSTM, the runner infers the concatenated flattened CNN feature width and calls `check_temporal_memory`. The estimator calculates exact LSTM/attention/head parameter counts for the configured widths and a conservative FP32 parameter+gradient+Adam-moments estimate. The default `--max-temporal-adam-mib 1024` rejects temporal parameter/optimizer footprints larger than 1024 MiB. Adjust this limit only after checking available memory.

**Important:** This is **not** a whole-model peak GPU memory estimator. It excludes CNN weights, forward/backward activations, batch/sequence size effects, temporary tensors, and CUDA allocator overhead. Passing the preflight does not guarantee that training fits on a GPU. Direct Flatten → Concatenate → LSTM is retained; no unreported feature bottleneck is introduced.

## Synthetic end-to-end smoke test

Run `python scripts/smoke_test_multimodal.py --sequence-length 2 --batch-size 2` to check the two-branch spatial interface, flattened fusion, temporal attention, output heads and aggregate evaluation on **synthetic** tensors. The smoke test uses tiny randomly initialized CNNs, not Xception/ResNeXt50 or DBNet. It is a software interface check, **not** an accuracy benchmark. GitHub Actions `PyTorch CPU integration` installs CPU PyTorch to execute these tests rather than silently skipping optional model tests.

## Undefined R² handling

When validation targets have zero variance, R² is mathematically undefined. The evaluation output records `null` for R² in strict JSON rather than inventing a score or emitting non-standard `NaN`. RMSE and threshold accuracy remain defined for finite nonempty arrays. This is an explicit undefined metric, not a measured performance value.

## Temporal parameter-memory validation

The CPU integration workflow instantiates representative PyTorch LSTM + attention + regression-head models and checks that their actual parameter counts equal the analytic `estimate_temporal_memory` calculation. The reported Adam-state memory assumes four copies of each parameter's element storage (weights, gradients and two moments); it **excludes** convolutional backbones, intermediate activations, input batches, temporary buffers, optimizer metadata and allocator overhead. Passing the preflight cap does **not** guarantee a training run will fit in GPU memory. Backbone output widths must be measured for the installed `timm` models and chosen input size before making any hardware sizing claim.

## Backbone feature-map inspection

With `torch` and `timm` installed, inspect the actual spatial outputs before training:

```bash
python scripts/probe_backbones.py --assignment A --output results/backbone-A.json
python scripts/probe_backbones.py --assignment B --output results/backbone-B.json
```

The command uses 299×299 synthetic input and randomly initialized backbone weights by default, avoiding a pretrained-weight download. It reports each branch's measured `[1,C,H,W]` output, the concatenated flattened width, and the corresponding temporal-model parameter and Adam-state estimates. This is not a measurement of total GPU memory and not a dataset accuracy evaluation. The CI probe tests use small mocked backbones; **the full timm Xception and ResNeXt-50 forward passes have not been verified by those tests**. Run the commands in a suitable environment and inspect the generated JSON before selecting hardware.

### Optional temporal-memory cap

After measuring the installed CNN output shapes, enforce an explicit cap on the *temporal parameter/Adam-state estimate*:

```bash
python scripts/probe_backbones.py --assignment A --max-temporal-adam-mib 1024
python scripts/probe_backbones.py --assignment B --max-temporal-adam-mib 1024
```

A failed cap is a deliberate preflight rejection; a passed cap does not certify total RAM/VRAM sufficiency. The full-backbone shape measurements for A and B are available as downloadable GitHub Actions artifacts from the dedicated workflow. Record the exact PyTorch/timm versions alongside measurements when comparing runs.

## Measured backbone widths and optional pooling study

The CPU shape probe on 2026-10-17 (Python 3.11.17, torch 2.14.1+cpu, timm 1.0.30, random backbone weights) measured a `[1,2048,10,10]` spatial output for **each** camera and LiDAR branch at 299×299 input, in both A and B assignments. Direct Flatten/Concat therefore yields 409,600 features per frame. At LSTM hidden size 64 and one layer, the temporal model has 104,874,691 parameters and an estimated 1,600.26 MiB for weights, gradients, and Adam moments alone. See [the measured GitHub Actions run](https://github.com/morteza-da/End-to-End-Learning-for-Autonomous-Driving/actions/runs/39048816595).

An **optional, distinct experimental variant** uses adaptive spatial average pooling on each CNN output before flattening and concatenation (`feature_compression.py`). For the measured 2048-channel outputs:

| Pool grid | Fused feature width | Relative input width |
| --- | ---: | ---: |
| None (baseline, 10×10) | 409,600 | 100% |
| 5×5 | 102,400 | 25% |
| 2×2 | 16,384 | 4% |
| 1×1 | 4,096 | 1% |

These are feature-width calculations, **not training results**. Spatial pooling changes the model architecture and can discard spatial detail; benchmark each variant on identical drive-disjoint splits and the same steering/speed evaluation metrics before drawing performance conclusions. The existing baseline training path remains unchanged.

### Explicit pooled end-to-end variant

`build_end_to_end_model(..., pool_size=None)` remains the original full-spatial Flatten/Concat model. For a separate spatial-pooling experiment, pass `pool_size=1`, `2`, or `5` to both `infer_feature_width(camera, lidar, pool_size=N)` and `build_end_to_end_model(camera, lidar, input_size=width, pool_size=N)`. The pooling operation is differentiable and its CNN gradients are checked by the PyTorch integration test. It is not yet exposed as a flag in the training CLI; no dataset-based comparison or accuracy claim has been made.

### Training CLI: isolated fusion variants

The training command accepts `--pool-size 1`, `--pool-size 2`, or `--pool-size 5`. Omit the flag for the original direct Flatten/Concat baseline. The output layout isolates checkpoints and metrics in `results/multimodal/A/flatten/`, `results/multimodal/A/pool-2x2/`, etc. `run_metadata.json` and the best checkpoint record the selected variant. All variants still require independently verified steering/speed label channels and units. The CLI does **not** imply that training has been executed or that any variant is more accurate.

The canonical `ExperimentConfig` now records `pool_size` (`null` for the full-spatial baseline, or `1`, `2`, `5` for explicit pooled variants). Configuration round-trip and invalid-value tests run in CPU CI. Compare only runs with matching data split, target schema, random seed, pretrained-weight policy, optimization schedule and backbone trainability. Report steering/speed RMSE, R² where defined, and the documented threshold accuracies; do not infer an accuracy benefit from parameter savings alone.

### Validation comparison CSV (measured results only)

After running real, separately configured experiments, generate a comparison table:

```bash
python scripts/compare_fusion_variants.py \
  results/multimodal/A/flatten/validation_metrics.json \
  results/multimodal/A/pool-2x2/validation_metrics.json \
  --output results/fusion-comparison.csv
```

The tool reads existing validation JSON reports and extracts steering/speed RMSE, R² (blank if undefined), threshold accuracies (±6° steering; ±5 km/h speed), and sample counts. It rejects mismatched configurations, validation counts, non-validation reports, and duplicate fusion variants. It never fabricates results or compares validation with test scores. Equal sample counts alone do **not** prove identical underlying frame/drive membership; dataset-manifest identity must be established independently before a publishable comparison.

### Exact temporal-window membership verification

Each dataset split now exposes a SHA-256 fingerprint of its sorted temporal windows, based on split, drive ID, frame indices, sample IDs and relative sensor paths. The training CLI records train/validation hashes in `run_metadata.json` and includes the validation fingerprint in `validation_metrics.json` notes. The comparison tool refuses reports without a matching validation fingerprint, even if their sample counts agree. This establishes **manifest window membership identity only**; it does not establish sensor timestamp synchronization, calibration, equality of underlying file bytes, or equality of label contents.

### Mandatory target-label provenance

Training requires `--label-provenance path/to/reviewed-evidence.json` alongside the explicit steering/speed channel and unit flags. The JSON must supply a source title, HTTP(S) source URL, evidence excerpt, reviewer, review date, and the exact matching channel indices and physical units. A deliberately invalid placeholder is provided at `docs/label-provenance-template.json`; **do not use it as evidence or replace it with invented claims**. The guard verifies presence and consistency only; it does not independently authenticate the source or establish DBNet label meanings. An actual researcher must inspect authoritative documentation before enabling a DBNet experiment. The provenance record is copied into local run metadata; review it for confidential details before sharing.

### DBNet channel order verified against upstream implementation

See `docs/dbnet-label-evidence.md` for exact primary-source code expressions and the distinction between source-code channel mapping and secondary-publication speed-unit corroboration. The documented working mapping is steering channel 0 in degrees and speed channel 1 in km/h; source-review provenance and actual dataset inspection are still required before any real training. Do not apply the original upstream normalized-speed transform to labels that are already interpreted in physical units.

### Physical target-range preflight

Before loading CNN weights, the training CLI audits all mapped train/validation labels. Its default **configurable research plausibility bounds** are steering between -40° and +40° and speed between 0 and 120 km/h (inclusive). Out-of-bound observations stop training rather than being clipped or silently discarded. Per-split observed minima, maxima and outlier counts are recorded in `run_metadata.json` for completed preflights. These limits are assumptions for this implementation, **not independently verified hard bounds for every DBNet sample**; inspect and revise the policy if legitimate data exceed them. Passing this audit does not establish label correctness or data quality.

### Read-only DBNet preflight

Before any real training, run the independent audit on a locally extracted, authorized dataset:

```bash
python scripts/audit_dbnet_preflight.py \
  --data-root data/dbnet-2018 \
  --label-provenance path/to/reviewed-evidence.json \
  --steering-channel 0 --speed-channel 1 \
  --steering-unit deg --speed-unit km/h \
  --sequence-length 4 --stride 1 \
  --output results/dbnet-preflight.json
```

The command does not train, load PyTorch, upload raw data, or copy sensor files. It reports sensor pairing counts, missing files, per-split label minima/maxima and outliers, valid temporal-window counts, and membership SHA-256 fingerprints. It exits nonzero when readiness checks fail; the JSON is still saved for diagnosis. The steering ±40° and speed 0–120 km/h limits are provisional research plausibility bounds, not guaranteed DBNet data bounds. A passing audit does not prove physical sensor synchronization, label provenance authenticity, or file integrity.

### Mandatory matching preflight for training

The training CLI now requires `--preflight-report results/dbnet-preflight.json` as well as `--label-provenance`. It rejects failed audits, mismatched label channel/unit mappings, temporal sequence length or stride, plausibility limits, window counts, and exact train/validation window membership hashes **before constructing neural networks**. Generate the report using `scripts/audit_dbnet_preflight.py` on the same dataset root and configuration. This gate checks manifest identity, not file-content hashes, timestamps, source authenticity, or the underlying physical meaning of labels.

The preflight gate additionally requires the same resolved local dataset root and source URL in the reviewed evidence record. It cannot establish whether source evidence is authentic or whether unchanged file paths contain unchanged bytes.

The audit now also records `canonical_targets_sha256` over sample/split/drive/frame identities and canonical steering/speed floating-point values. The training gate recomputes this digest and rejects modified labels even when window paths and counts are unchanged. This is a label-integrity check, **not** a hash of camera/LiDAR file contents.

### Optional full sensor-content integrity gate

To detect changed camera JPEG or LiDAR LAS bytes between audit and training, append `--hash-sensor-files` to `scripts/audit_dbnet_preflight.py` and append `--require-sensor-hashes` to `scripts/train_multimodal.py`. Both commands then stream SHA-256 hashes of every paired sensor file referenced by labeled observations; the training gate rejects any mismatch. The JSON report contains only a combined manifest digest and file count, not raw sensor bytes. This can be slow for large datasets and does not validate physical synchronization, file decodability, or dataset license. Without these flags, file-byte integrity is **not** checked.
