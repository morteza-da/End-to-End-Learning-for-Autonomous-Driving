# End-to-end camera–LiDAR training

## Data flow

1. Use a validated `Observation` manifest with `drive_id`, frame index, split, camera path, LiDAR path, and **verified** steering (degrees) and speed (km/h).
2. `make_sequence_dataset` constructs contiguous windows within individual drives. The last frame provides each window's target.
3. JPEG frames and LAS clouds are decoded lazily. LAS points are converted into 299×299 spherical RGB range images with an occupancy mask.
4. The camera and LiDAR CNN branches extract spatial feature maps. Each map is **Flattened**; vectors are **Concatenated** per frame, without a Dense bottleneck.
5. The fused sequence enters LSTM → temporal attention → a two-output regression head.

## Python API

```python
from torch.utils.data import DataLoader
from autonomous_driving.backbone_extraction import make_multimodal_backbones
from autonomous_driving.end_to_end import (
    infer_feature_width, build_end_to_end_model, run_multimodal_epoch,
)
from autonomous_driving.sequence_dataset import make_sequence_dataset

# observations: validated list of Observation objects with verified label units
dataset = make_sequence_dataset(
    observations, "data/dbnet-2018", sequence_length=4, split="train"
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
camera_cnn, lidar_cnn = make_multimodal_backbones("A", pretrained=True)
feature_width = infer_feature_width(camera_cnn, lidar_cnn)
model = build_end_to_end_model(
    camera_cnn, lidar_cnn, input_size=feature_width, hidden_size=64
)
# optimizer = torch.optim.Adam(
#     (p for p in model.parameters() if p.requires_grad), lr=1e-4
# )
# train_loss = run_multimodal_epoch(model, loader, optimizer=optimizer)
```

This API example is not a benchmark run. Install optional dependencies with `pip install -e '.[data,model]'`. Import `torch` before constructing an optimizer. Use `freeze_all_except_last_n_children` to specify and record which backbone stages are trainable.

## Important operational constraints

- **Label mapping:** Raw DBNet `behavior.csv` column order and physical units are not inferred. Populate the observation manifest only after checking dataset documentation.
- **Memory:** Flattening spatial maps can produce a large LSTM input matrix. Inspect `feature_width`, available accelerator memory, batch size, and sequence length before allocating the temporal model. Do not silently replace Flatten with pooling or insert a Dense projection.
- **Normalization:** ImageNet mean/std are available as a configurable preprocessing option for both RGB branches. Verify the intended normalization for LiDAR pseudo-RGB.
- **Training:** `run_multimodal_epoch` supports joint gradient updates, but it does not yet implement checkpoint management, learning-rate schedules, mixed precision, or complete experiment tracking.
- **Evaluation:** The epoch loss is an average of batch objectives, not dataset-global RMSE. Report independently aggregated per-target metrics on held-out data.
- **Data governance:** Raw DBNet archives are not distributed in this repository. Cite Chen et al. (CVPR 2018) and follow their data terms.

## Validation

```bash
python -m pip install -e '.[data,model]'
python -m unittest discover -s tests -v
```

`tests/test_end_to_end.py` provides a synthetic gradient-flow integration test using lightweight CNNs. Full pretrained backbone execution and real DBNet training remain to be measured.
