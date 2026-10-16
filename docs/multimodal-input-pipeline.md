# Multimodal Input Pipeline — Researcher-Confirmed Details

## Modalities and spatial dimensions

The doctoral research used two **different sensor modalities**, each represented as a **299 × 299 image** before backbone feature extraction.

| Modality | Source | Transformation | Backbone input |
| --- | --- | --- | --- |
| Camera | RGB camera image | Resize to 299 × 299 | 2D image, 299 × 299 |
| LiDAR | 3D LiDAR data | Convert to 2D, color-coded distance image, then 299 × 299 | 2D distance-encoded image, 299 × 299 |

The LiDAR representation is a **2D image derived from 3D range data**, where color represents distance. It is not raw 3D point-cloud input to the CNN.

## Backbone training

Both backbones, **Xception** and **ResNeXt-50**, were initialized with **ImageNet-pretrained weights**. Earlier layers were frozen and upper layers fine-tuned on the research **DBNet** dataset.

The two branches consume **different sensor representations**. The exact mapping of Xception versus ResNeXt-50 to camera versus LiDAR is **not yet confirmed** and must not be assumed.

## Critical open implementation details

- Confirm backbone-to-modality assignment.
- Confirm the 3D LiDAR-to-2D projection or rasterization algorithm (e.g., coordinate frame, field of view, pixel mapping, occlusion and missing returns).
- Confirm the distance-to-color encoding (colormap, distance range, normalization, handling of invalid measurements).
- Confirm image channel order and whether ImageNet normalization was applied to the color-coded LiDAR images.
- Confirm temporal synchronization of camera and LiDAR observations.
- Confirm interpolation used in resizing.
- Confirm feature fusion, sequence length, and downstream LSTM/attention specifications.

## Reproducibility boundary

The input shape and high-level LiDAR transformation are confirmed by the researcher. No specific projection algorithm, calibration matrix, distance colormap, or backbone mapping is asserted without further evidence. The original code was lost; reconstructed code must be clearly labeled and validated.
