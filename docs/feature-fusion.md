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

The researcher confirms **no Dense layer between concatenation and LSTM**. The concatenated feature vector was fed directly to the LSTM. The processing order after fusion was **LSTM → Attention → Dense**. Exact feature-map shapes, concatenated feature dimensions, temporal sequence length, and Dense/output-head settings still require verification.

## Confirmed temporal sequence input

The researcher confirms that the LSTM processed **sequences of multiple consecutive driving frames**, not isolated camera–LiDAR pairs. At each time step, features from the synchronized camera and LiDAR representations were separately flattened and concatenated; the ordered fused vectors then formed the LSTM input of conceptual shape `[batch, sequence_length, fused_feature_dimension]` (batch-first convention shown for clarity, not a verified framework setting).

The **exact sequence length, stride, temporal sampling interval, sequence boundaries, and target alignment** are not yet confirmed. No particular frame count should be presented as historical fact.

## Confirmed prediction targets

The model **simultaneously predicted steering angle and vehicle speed** from the shared temporal representation. This is a joint two-target regression task. Whether the final implementation used one two-unit Dense layer or two separate output heads remains unconfirmed; do not assume either arrangement without dissertation evidence.

## Implementation constraints

- Concatenate feature representations along the **feature dimension**, not the temporal or batch dimensions.
- Match each camera feature with the corresponding synchronized LiDAR observation.
- Keep the temporal ordering intact for the LSTM.
- Preserve the two feature branches at fusion; do not introduce averaging, addition, or a learned gating step unless documented.
- Flatten the backbone feature maps separately before concatenating their feature vectors.
- Place Attention after the LSTM and Dense after Attention; verify attention variant and the exact Dense/output-head design from the dissertation.

**Scientific precision:** Concatenation retains the extracted feature components in the fused representation; it does **not** prove that all information from the original sensor measurements survives backbone extraction or later network processing.

## Status

Fusion operator and placement before LSTM: **confirmed by researcher**. Flatten before concatenation: **confirmed by researcher**. LSTM → Attention → Dense ordering and absence of pre-LSTM Dense: **confirmed by researcher**. Simultaneous steering-and-speed prediction: **confirmed by researcher**. Consecutive-frame sequence input: **confirmed by researcher**. Exact tensor dimensions, sequence length, attention variant, and output-head implementation: **awaiting verification**.
