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

The evaluation utility computes MAE, RMSE, R², and threshold accuracy at steering ±6° and speed ±5 km/h. The report stores the supplied metrics and experiment metadata; it does not run inference or invent results. R² is undefined for constant ground truth and may be NaN; the report writer rejects nonfinite metrics, so a constant-target evaluation requires an explicit reporting policy.

## Reproducibility checklist

- Record dataset source/version, train/validation/test drive IDs, and audited sample counts.
- Confirm channel order, steering/speed units, and any label filtering.
- Record LiDAR angular projection, collision policy, image normalization, and sensor alignment assumptions.
- Record timm/PyTorch versions, pretrained checkpoint identities, frozen/trainable module names, random seed, optimizer, and hardware.
- Save real checkpoint paths, experiment config, and validation/test metrics for each backbone assignment.
- Do not claim identical GPU determinism solely from setting a random seed.
- Never commit restricted DBNet data, private correspondence, or large model checkpoints without confirming redistribution rights.

Unit tests exercise config serialization, input validation, synthetic metric calculations, and report formatting. They are not substitutes for DBNet benchmark evaluation.
