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

After confirming the dataset's label channels and physical units, use `scripts/train_multimodal.py`. Both label-channel flags and unit flags are mandatory; the script cannot infer them. The example values below are **placeholders** and must be replaced with verified dataset metadata:

```bash
python scripts/train_multimodal.py --data-root data/dbnet-2018 --assignment A --steering-channel 0 --speed-channel 1 --steering-unit deg --speed-unit km/h --output results/multimodal
python scripts/train_multimodal.py --data-root data/dbnet-2018 --assignment B --steering-channel 0 --speed-channel 1 --steering-unit deg --speed-unit km/h --output results/multimodal
python scripts/compare_backbones.py results/multimodal/A/latest_metrics.json results/multimodal/B/latest_metrics.json
```

The runner saves `config.json`, `run_metadata.json`, `latest_metrics.json`, and the best validation checkpoint `best.pt` in separate A/B folders. It reports batch-averaged joint training/validation objectives, then reloads the best checkpoint to produce `validation_metrics.json` (aggregate RMSE, R², and threshold accuracy) and `validation_predictions.json` (per-window targets and predictions). It does **not** evaluate the unlabeled DBNet test split. These commands have not been validated on the complete dataset. Full-resolution flattened CNN maps can require substantial GPU memory; inspect `feature_width` and memory requirements before training.

## Memory preflight for flattened CNN features

Before allocating the LSTM, the runner infers the concatenated flattened CNN feature width and calls `check_temporal_memory`. The estimator calculates exact LSTM/attention/head parameter counts for the configured widths and a conservative FP32 parameter+gradient+Adam-moments estimate. The default `--max-temporal-adam-mib 1024` rejects temporal parameter/optimizer footprints larger than 1024 MiB. Adjust this limit only after checking available memory.

**Important:** This is **not** a whole-model peak GPU memory estimator. It excludes CNN weights, forward/backward activations, batch/sequence size effects, temporary tensors, and CUDA allocator overhead. Passing the preflight does not guarantee that training fits on a GPU. Direct Flatten → Concatenate → LSTM is retained; no unreported feature bottleneck is introduced.
