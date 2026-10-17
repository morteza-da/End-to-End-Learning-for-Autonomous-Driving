# Research Architecture and Reproducibility Roadmap

## Scope and provenance

This document specifies the model architecture and evaluation workflow.

## Research objective

Learn an end-to-end mapping from temporally aligned camera and LiDAR observations to continuous vehicle-control targets:

- Steering angle (degrees)
- Vehicle speed (km/h)

The reported research setting considers steering values approximately within ±40° and speeds within 0–120 km/h. These are research target ranges, not safety or deployment guarantees.

## Reported conceptual pipeline

```text
RGB camera frames ──> Xception feature extractor ──┐
                                                     ├─> multimodal feature integration
LiDAR observations ──> ResNeXt-50-based extractor ───┘
                                                               │
                                                               v
                                                     temporal sequence model
                                                             (LSTM)
                                                               │
                                                               v
                                                        attention module
                                                               │
                                                               v
                                                   steering + speed heads
```

**Important:** The precise LiDAR representation, backbone-to-modality assignment, feature-fusion operator, temporal window, output scaling, and training hyperparameters must be checked against the research before implementation is described as a faithful reproduction. The diagram is conceptual rather than an exact executable specification.

## implementation milestones

### 1. Dataset audit
- Establish the authoritative dataset manifest and file provenance.
- Verify timestamps, camera–LiDAR correspondence, calibration, labels, and units.
- Document subject/route/sequence boundaries to prevent train–test leakage.
- Define and freeze train/validation/test splits.

### 2. Baseline implementation
- Build deterministic loaders with input validation.
- Implement camera-only and LiDAR-only baselines.
- Add shape, range, and missing-data tests.

### 3. Proposed model
- Reconstruct Xception and ResNeXt-50 feature extraction from documented specifications.
- Implement multimodal fusion, LSTM, attention, and regression heads.
- Track model versions, random seeds, preprocessing, and configuration.

### 4. Evaluation
- Report MAE, RMSE, and R² separately for steering and speed.
- Include temporal prediction traces, error distributions, and per-scenario breakdowns.
- Estimate predictive uncertainty using MC dropout if supported by the verified methodology.
- Separate prior research-reported results from independently reproduced results.

### 5. Release readiness
- Provide pinned dependencies, a reproducible run command, and automated tests.
- Publish only data and model weights that can legally and ethically be shared.
- Add dataset/model cards, limitations, and citation information.

## Reproducibility policy

Do not treat preliminary results, remembered scores, illustrative plots, or placeholder data as validated experiments. Each reported number should identify its source, evaluation split, metric definition, and run configuration.

## Collaboration direction

Potential extensions include language-grounded scene interpretation, semantic–control consistency, uncertainty-aware evaluation, and cross-dataset transfer. These are **future research directions**, not completed findings.

## Researcher

**Morteza Aghamohammadi**  
PhD in Mechanical Engineering  
Toronto, Canada  
[LinkedIn](https://www.linkedin.com/in/morteza-a/) · [ORCID](https://orcid.org/0009-0001-4837-2298)
