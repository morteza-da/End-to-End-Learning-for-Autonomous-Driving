# Implementation Status

This repository is being rebuilt from dissertation documentation after loss of the original local source code.

| Component | Status |
| --- | --- |
| Architecture documentation | Drafted; requires cross-checking with thesis |
| Dataset manifests and validation | Not implemented |
| Camera preprocessing | Not implemented |
| LiDAR preprocessing | Not implemented |
| Xception / ResNeXt-50 branches | Not implemented |
| Sensor fusion and temporal LSTM | Not implemented |
| Attention and prediction heads | Not implemented |
| MC dropout uncertainty | Not implemented |
| Training and checkpointing | Not implemented |
| Evaluation utilities | Initial framework-independent MAE, RMSE, R² functions added; tests written but not yet executed in CI |
| Historical experimental reproduction | Not attempted |

## Local validation

Install the package and run the unit tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

These tests use synthetic numbers solely to verify metric calculations. They do not validate the doctoral model or historical dissertation metrics.

## Before implementing the neural network

Confirm dataset layout, sensor calibration, target units, split manifests, sequence length, preprocessing, and model hyperparameters from primary research materials. Preserve provenance for all newly reconstructed code.
