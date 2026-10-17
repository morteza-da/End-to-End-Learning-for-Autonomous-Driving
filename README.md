# End-to-End Learning for Autonomous Driving

**Camera–LiDAR Fusion · Deep Learning · Intelligent Vehicle Control**

**Morteza Aghamohammadi, PhD** · Toronto, Canada  
[LinkedIn](https://www.linkedin.com/in/morteza-a/) · [ORCID](https://orcid.org/0009-0001-4837-2298)

## Overview

Multimodal learning for joint **steering-angle and vehicle-speed prediction** from temporally aligned camera and LiDAR observations. The model combines ImageNet-pretrained **Xception** and **ResNeXt-50** feature extractors, feature-level concatenation, an **LSTM**, temporal **attention**, and regression outputs. Both camera/backbone assignments are supported.

## Model pipeline

```text
Camera RGB (299×299) ── CNN ── Flatten ─┐
                                       ├─ Concatenate ─ LSTM ─ Attention ─ Dense ─ Steering + Speed
LiDAR RGB (299×299) ── CNN ── Flatten ──┘
```

The LiDAR RGB image is derived from a 3D point cloud using distance-based color encoding. Exact rasterization settings must be specified for each experiment.

## Dataset

Experiments use **DBNet**, introduced by Chen et al., *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively* (CVPR 2018). [Official dataset](https://github.com/driving-behavior/DBNet) · [BibTeX](CITATION_DBNet.bib) · [Data setup](docs/dbnet-local-data.md).

The repository includes a dataset layout audit, JPEG/LAS decoders, paired-frame validation, and temporal sequence utilities. Dataset files are not included in this public repository.

## Getting started

```bash
python -m pip install -e '.[data,model]'
python -m unittest discover -s tests -v
python scripts/audit_dbnet.py data/dbnet-2018
python scripts/inspect_dbnet_pair.py data/dbnet-2018/train/1 0
```

The last two commands require locally extracted DBNet data. See [dataset instructions](docs/dbnet-local-data.md) and [sensor processing](docs/dbnet-media-pipeline.md).

## Repository contents

- `src/autonomous_driving/` — dataset utilities, CNN backbones, LiDAR encoding, fusion, LSTM-attention model, metrics, and training utilities
- `notebooks/` — LiDAR processing, backbone features, temporal modeling, and training examples
- `scripts/` — DBNet dataset audit and sensor-pair inspection
- `tests/` — unit tests for data, model interfaces, and metrics
- `docs/` — [research methods](docs/RESEARCH_METHODS.md), architecture, and experiment protocols

## Evaluation

Report MAE, RMSE, R², and threshold accuracy separately for steering and speed. Include dataset version, split, random seed, preprocessing configuration, checkpoint, and environment with each measured result. Training and benchmark numbers are published only when accompanied by actual run artifacts.

## Research and collaboration

Research interests include autonomous driving, multimodal perception, temporal learning, intelligent vehicles, uncertainty estimation, and vehicle dynamics. Open to academic collaborations and postdoctoral research opportunities.

## License and citation

The project source code is distributed under the [MIT License](LICENSE). For academic use, cite this software using [CITATION.cff](CITATION.cff). When using the DBNet dataset, also cite the original dataset authors via [CITATION_DBNet.bib](CITATION_DBNet.bib). The software license does not grant rights to redistribute third-party datasets.

## Acknowledgment

DBNet dataset credit: Chen et al. (CVPR 2018). Dataset access and redistribution are governed by the original authors' terms.
