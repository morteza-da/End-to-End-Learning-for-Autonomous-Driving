# Backbone-to-Modality Assignment Experiments

## Confirmed experimental configurations

The researcher confirms that **both backbone-to-sensor assignments were evaluated** in the dissertation.

| Configuration | RGB camera (299 × 299) | 2D color-coded LiDAR distance image (299 × 299) |
| --- | --- | --- |
| A | ResNeXt-50 | Xception |
| B | Xception | ResNeXt-50 |

For both configurations, each backbone was initialized with **ImageNet-pretrained weights**. Earlier layers were frozen, and upper layers were fine-tuned on **DBNet**.

This document supersedes the earlier unresolved backbone-to-modality assignment in `docs/multimodal-input-pipeline.md`: the research investigated **both assignments**, not a single unknown assignment.

## Evaluation design to reconstruct

- Keep dataset splits, input representation, training budget, and evaluation procedure comparable between configurations.
- Log the exact trainable blocks and optimizer hyperparameters for each experiment.
- Compare steering and speed predictions using separately reported MAE, RMSE, and R², with target units and sample counts.
- Include model seeds, checkpoints, predictions, and split manifests in new experiments.
- Report results only from verified dissertation tables or newly executed runs.

## Not yet confirmed

- The dissertation's Table 4-3 (PDF p. 118) reports the highest threshold-based accuracies for its ResNeXt–Xception–Attention–LSTM row: 94.91% steering and 94.09% speed. Match model-name order against diagrams before claiming a modality mapping from the row label.
- Whether other hyperparameters and training schedules were held constant.
- Exact LiDAR projection and distance colormap.
- Backbone layer freeze boundaries, fusion operator, LSTM sequence length, and attention implementation.

**Source clarification:** The A/B mapping follows the dissertation's explicit descriptions on PDF pp. 105–106, superseding the earlier swapped A/B labels in this document. Both assignments were evaluated.\n\n**Integrity note:** This is a researcher-confirmed record of the experimental design. No numerical results or winning configuration are asserted here.
