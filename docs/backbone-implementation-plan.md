# Two-Backbone Implementation Plan


## Verified assignment

| research configuration | Camera RGB 299x299 | 2D LiDAR RGB 299x299 |
| --- | --- | --- |
| A | ResNeXt-50 | Xception |
| B | Xception | ResNeXt-50 |

Both architectures were initialized with ImageNet pretrained weights. Early backbone layers were frozen and upper layers fine-tuned on DBNet (researcher-confirmed). The exact block boundaries are **not documented here as prior fact**.

## implementation interface

`src/autonomous_driving/backbone_config.py` provides immutable, validated A/B assignment definitions and an explicit fine-tuning plan object. It does not load networks or download pretrained weights.

The intended forward path is:

1. Input aligned camera and 2D LiDAR frames (299x299 each).
2. Apply the assigned ImageNet-pretrained backbone to each modality.
3. Flatten each branch's feature map independently.
4. Concatenate feature vectors along the feature axis (no Dense before LSTM).
5. Feed temporally ordered fused vectors into LSTM.
6. Apply attention, then Dense prediction layers for steering angle and vehicle speed.

## Before implementing runnable backbones

- Select and pin an implementation providing compatible **Xception** and **ResNeXt-50** with ImageNet weights; do not assume arbitrary library variants match the prior versions.
- Inspect layer names and explicitly choose freeze boundaries as **new implementation hyperparameters**, logging their values.
- Check that both backbones output spatial feature maps rather than classification logits, so Flatten can be applied as described.
- Verify output tensor dimensions and memory requirements at 299x299; flattened maps may be large.
- Confirm image normalization, sequence length, attention variant, and loss aggregation from primary evidence when possible.
- Test with synthetic inputs before attempting any training on DBNet.

**Integrity:** prior research results and newly implemented experiment results must be clearly separated.
