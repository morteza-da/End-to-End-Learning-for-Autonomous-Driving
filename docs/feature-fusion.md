# Confirmed Feature Fusion: Concatenation Before LSTM

## Researcher-confirmed design

The outputs of the two pretrained, partially fine-tuned backbones were **concatenated at the feature level** and the fused features were passed to the **LSTM**.

This was **feature concatenation**, not model stacking, feature averaging, or feature summation. The intent was to retain both branches' extracted feature components at the fusion stage.

For each synchronized time step `t`:

```text
Camera RGB (299×299) ── backbone A ── camera features ─┐
                                                       ├─ concatenate ── fused features ── LSTM
LiDAR distance image (299×299) ─ backbone B ─ features ┘
```

Both assignments of Xception and ResNeXt-50 to camera and LiDAR were tested (see `backbone-modality-assignments.md`).

In mathematical notation, `fused_t = concat(camera_features_t, lidar_features_t, feature_axis)`. Across a time window, `[fused_1, ..., fused_T]` forms the input sequence to the LSTM.

## Confirmed feature-map vectorization

The researcher confirms that **Flatten** was used to convert each backbone's feature maps into a feature vector **before concatenation**. Thus the intended path is **backbone feature maps → Flatten (per branch) → feature concatenation → LSTM**. Global Average Pooling (GAP) must not be substituted for Flatten in the faithful reconstruction.

The exact feature-map shapes, concatenated feature dimension, and any optional dimensionality-reduction layer remain to be verified against the dissertation.

## Implementation constraints

- Concatenate feature representations along the **feature dimension**, not the temporal or batch dimensions.
- Match each camera feature with the corresponding synchronized LiDAR observation.
- Keep the temporal ordering intact for the LSTM.
- Preserve the two feature branches at fusion; do not introduce averaging, addition, or a learned gating step unless documented.
- Flatten the backbone feature maps separately before concatenating their feature vectors.
- Confirm whether attention followed the LSTM and the exact prediction-head design from the dissertation.

**Scientific precision:** Concatenation retains the extracted feature components in the fused representation; it does **not** prove that all information from the original sensor measurements survives backbone extraction or later network processing.

## Status

Fusion operator and placement before LSTM: **confirmed by researcher**. Flatten before concatenation: **confirmed by researcher**. Exact tensor dimensions, sequence length, attention mechanism, and downstream output heads: **awaiting verification**.
