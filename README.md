# End-to-End Learning for Autonomous Driving

**PhD research project | Deep Learning · Camera–LiDAR Fusion · Intelligent Vehicle Control**

**Researcher:** Morteza Aghamohammadi, PhD in Mechanical Engineering  
**Location:** Toronto, Canada  
**LinkedIn:** https://www.linkedin.com/in/morteza-a/  
**ORCID:** https://orcid.org/0009-0001-4837-2298

## Research overview

This repository documents doctoral research on end-to-end learning for autonomous driving. The research investigates multimodal perception and temporal modeling for predicting **steering angle and vehicle speed** from camera and LiDAR observations.

The proposed research architecture combines **Xception** and **ResNeXt-50** feature extraction with **LSTM** temporal modeling and an **attention mechanism**. Monte Carlo dropout is investigated for uncertainty estimation.

## Research workflow

1. Prepare and synchronize camera and LiDAR observations.
2. Extract complementary visual and spatial features.
3. Fuse multimodal features and model temporal dependencies.
4. Predict steering and speed.
5. Evaluate prediction accuracy, generalization, and uncertainty.

## Dataset

The doctoral research used a driving dataset of more than 11,000 frames with camera and LiDAR information. The researcher has already supplied the original DBNet train, validation, and test archives for reconstruction. The original dataset is credited to **Chen et al., “LiDAR-Video Driving Dataset: Learning Driving Policies Effectively,” CVPR 2018** ([official DBNet repository](https://github.com/driving-behavior/DBNet); [BibTeX](CITATION_DBNet.bib)). The archives are available in the researcher's private file Library, **not** publicly hosted here. A real-layout local audit tool is available at [`scripts/audit_dbnet.py`](scripts/audit_dbnet.py), with instructions in [`docs/dbnet-local-data.md`](docs/dbnet-local-data.md). Full integrity checks and label-unit verification remain pending.

## Repository status

> **Reconstruction in progress.** The original local source-code files were lost before they could be uploaded to GitHub. This repository is being rebuilt from dissertation materials and available research records. Newly written implementations must not be interpreted as the original historical training code.

The architecture described above reflects the dissertation's research design. Reconstructed scripts, dependency versions, datasets, pretrained weights, and reproducibility instructions will be added only after they are implemented and checked.

## Planned organization

```text
docs/          Research methodology and architecture documentation
src/           Reconstructed data processing, modeling, and evaluation code
configs/       Experiment configurations
tests/         Unit and integration tests
notebooks/     Exploratory analyses
results/       Verified figures, metrics, and experiment logs
```

The repository now includes reconstructed modules, notebooks, documentation, tests, and a local DBNet audit script. **No full real-data training run has yet been verified.**

## Results and reproducibility

Quantitative results, baselines, split definitions, evaluation scripts, and reproducible run commands will be published here after verification against the dissertation and available experimental records. No placeholder scores are presented as independently reproduced results.

## Research interests

- Autonomous driving and intelligent transportation systems
- Deep learning and computer vision
- Multimodal sensor fusion (camera and LiDAR)
- Temporal sequence modeling and attention
- Vehicle dynamics and intelligent control
- Reliable and interpretable machine learning

## Collaboration

Interested in academic research collaborations and postdoctoral opportunities in autonomous driving, intelligent vehicles, and multimodal machine learning, particularly in Canada.

**Connect:** [LinkedIn](https://www.linkedin.com/in/morteza-a/) · [ORCID](https://orcid.org/0009-0001-4837-2298)

## Acknowledgment

This project is based on doctoral research. Full academic acknowledgments, references, publication links, and citation details will be added after verification.
