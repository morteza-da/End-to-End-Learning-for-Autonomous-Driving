# Temporal dataset and training scaffold

**Official DBNet citation:** Chen et al. (2018), *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively*, CVPR; https://github.com/driving-behavior/DBNet. See `CITATION_DBNet.bib`.

## Implemented
- `src/autonomous_driving/sequence_manifest.py`: generic observation schema, per-drive split validation, contiguous sliding windows, no window crossing drive boundaries, and configurable stride.
- `src/autonomous_driving/training.py`: PyTorch two-target weighted RMSE and training/validation epoch for **precomputed fused features**.
- `notebooks/06_Sequence_Training.ipynb`: synthetic demonstration only.
- `tests/test_sequence_manifest.py` and `tests/test_training.py`: tests, not yet confirmed executed by CI.

## Explicit new choices
- Drive-disjoint splits to reduce leakage. The dissertation specifies 80/10/10 but does not establish this split policy.
- Last-frame steering/speed targets per window.
- Weighted average of separate steering and speed RMSE, with weights defaulting to 1.0. The original joint target loss aggregation is unknown.
- Illustrative sequence length 3 in notebook; this is **not** an original dissertation parameter.

## Limitations
No real DBNet parser or sensor synchronization implementation, no verified original LiDAR pixel projection, no checkpointing or full image-to-regression training. `run_epoch` currently consumes synthetic/precomputed fused features, so it does **not** fine-tune the camera/LiDAR backbones. Batch-averaged RMSE is a monitoring statistic and not identical to dataset-wide RMSE; compute final evaluation on all held-out predictions. Do not report notebook random-data losses as research results.

**Reference:** official PyTorch `Dataset` and `DataLoader` documentation: https://docs.pytorch.org/docs/stable/data.html.
