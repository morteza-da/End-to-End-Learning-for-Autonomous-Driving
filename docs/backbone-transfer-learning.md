# Backbone Transfer Learning Protocol

## Confirmed by the researcher

Both **Xception** and **ResNeXt-50** were initialized from **ImageNet-pretrained weights**. In **both backbones**, the earlier layers were **frozen**, while upper layers were **fine-tuned using the research DBNet dataset** to adapt higher-level representations to the driving task.

| Component | Initialization | Early layers | Upper layers |
| --- | --- | --- | --- |
| Xception | ImageNet-pretrained | Frozen | Fine-tuned on DBNet |
| ResNeXt-50 | ImageNet-pretrained | Frozen | Fine-tuned on DBNet |

This is **partial fine-tuning / transfer learning**, not training both backbones from scratch and not freezing the entire feature extractors.

## Implementation constraints

1. Load compatible ImageNet-pretrained weights for each architecture.
2. Explicitly freeze early-stage parameters and enable gradients for the selected upper stages.
3. Record the exact frozen/trainable parameter names and counts in experiment logs.
4. Train on DBNet training split only, using validation data for model selection.
5. Do not infer layer boundaries, feature dimensions, image preprocessing, optimizer settings, or the training schedule without primary source evidence.
6. Distinguish newly reconstructed experiments from historical dissertation findings.

## Still requiring confirmation

- Framework and exact Xception / ResNeXt-50 implementations
- Frozen/unfrozen layer or block boundaries for each backbone
- Whether fine-tuning was joint with downstream fusion/LSTM/attention or staged
- ImageNet normalization and input resolution
- Fusion interface and whether both backbones process the same image input
- Learning rates, epochs, and checkpoint selection
- DBNet train/validation/test split construction

**Evidence status:** The transfer-learning setup above is directly confirmed by the researcher. Other implementation choices remain open pending dissertation/code documentation.
