# Dissertation Methodology Notes

These notes summarize the supplied 156-page doctoral dissertation. They describe historical research and do not imply that reconstructed code has reproduced the results.

## Training (PDF pp. 105–107)

- Dataset split: 80% training, 10% validation, 10% test.
- Both Xception and ResNeXt backbones use ImageNet pretrained weights.
- Configuration A: camera to ResNeXt, 2D LiDAR RGB to Xception.
- Configuration B: camera to Xception, 2D LiDAR RGB to ResNeXt.
- Adam optimizer and RMSE regression loss are described.
- Researcher separately confirms frozen early backbone layers, DBNet fine-tuning of upper layers, Flatten before feature concatenation, and consecutive-frame LSTM input.

## Evaluation (PDF pp. 108–110)

- RMSE and R-squared are reported separately.
- Threshold accuracy uses absolute steering error <= 6 degrees and speed error <= 5 km/h.
- Threshold accuracy is not R-squared or classification accuracy.

## LiDAR transformation (PDF pp. 73–74)

- The dissertation describes mapping 3D LiDAR points into a color-coded 2D RGB image.
- Its distance formula is sqrt(Px^2 + Py^2), with blue, green, and red channels using scales 20, 40, and 60 respectively.
- The printed conditional layout is ambiguous, so exact branch logic and point-to-pixel mapping must be verified before implementation.

## Remaining unknowns

- Exact LSTM sequence length and frame stride.
- Exact per-output RMSE loss aggregation and normalization.
- Exact LiDAR RGB conditional implementation and projection calibration.
- Exact frozen blocks, feature dimensions, and attention settings.

Do not substitute evaluation of 1,000 frames over 25 seconds (PDF p. 118) for the unknown LSTM input sequence length.
