# End-to-End Learning for Autonomous Driving

**Camera–LiDAR Fusion · Deep Learning · Intelligent Vehicle Control**

**Morteza Aghamohammadi, PhD** · Toronto, Canada  
[LinkedIn](https://www.linkedin.com/in/morteza-a/) · [ORCID](https://orcid.org/0009-0001-4837-2298)

## Overview

Multimodal learning for joint **steering-angle and vehicle-speed prediction** from frame-index-paired camera and LiDAR observations. The model combines ImageNet-pretrained **Xception** and **ResNeXt-50** feature extractors, feature-level concatenation, an **LSTM**, temporal **attention**, and regression outputs. Both camera/backbone assignments are supported.

## Model pipeline

```text
Camera RGB (299×299) ── CNN ── Flatten ─┐
                                       ├─ Concatenate ─ LSTM ─ Attention ─ Dense ─ Steering + Speed
LiDAR RGB (299×299) ── CNN ── Flatten ──┘
```

The LiDAR RGB input is produced from a 3D point cloud using [configurable spherical range-image projection](docs/lidar-range-image.md) and distance-based color encoding. Record angular field of view and rasterization settings for each experiment.

## Dataset

Experiments use **DBNet**, introduced by Chen et al., *LiDAR-Video Driving Dataset: Learning Driving Policies Effectively* (CVPR 2018). [Official dataset](https://github.com/driving-behavior/DBNet) · [BibTeX](CITATION_DBNet.bib) · [Data setup](docs/dbnet-local-data.md).

The repository includes a dataset layout audit, JPEG/LAS decoders, paired-frame validation, temporal sequence utilities, and an integrated CNN–LSTM–attention training interface. Dataset files are not included in this public repository.

## Getting started

```bash
python -m pip install -e '.[data,model]'
python -m unittest discover -s tests -v
python scripts/audit_dbnet.py data/dbnet-2018
python scripts/audit_sensor_pairing.py data/dbnet-2018 --output results/sensor_pairing_audit.json
python scripts/inspect_dbnet_pair.py data/dbnet-2018/train/1 0
```

The last two commands require locally extracted DBNet data. See [dataset instructions](docs/dbnet-local-data.md) and [sensor processing](docs/dbnet-media-pipeline.md).

## Repository contents

- `src/autonomous_driving/` — dataset utilities, CNN backbones, LiDAR encoding, fusion, LSTM-attention model, metrics, and training utilities
- `notebooks/` — LiDAR processing, backbone features, temporal modeling, and training examples
- `scripts/` — DBNet dataset audit and sensor-pair inspection
- `tests/` — unit tests for data, model interfaces, and metrics
- `docs/` — [research methods](docs/RESEARCH_METHODS.md), architecture, and experiment protocols

## End-to-end training interface

The [sequence dataset](src/autonomous_driving/sequence_dataset.py) loads frame-index-paired camera/LiDAR windows and verified steering/speed targets. The [end-to-end model](src/autonomous_driving/end_to_end.py) connects two spatial feature extractors to the LSTM-attention regression head. See [training integration](docs/end-to-end-training.md). Dataset label units and channel mappings must be established before real-data training; see [DBNet label validation](docs/dbnet-label-validation.md) and the explicit [target mapping module](src/autonomous_driving/target_mapping.py).

## Training readiness and reproducibility

Real-data training requires a **reviewed label-provenance record** and a **passing, matching data preflight report**. See [DBNet label evidence](docs/dbnet-label-evidence.md), [data preflight runner](scripts/audit_dbnet_preflight.py), and [experiment protocol](docs/experiment-protocol.md). These checks validate channel configuration, frame-index membership and provisional physical ranges; they do **not** prove timestamp synchronization, authenticity of supporting documentation, or dataset file integrity. No real DBNet performance results are claimed.

```bash
python scripts/audit_dbnet_preflight.py \\
  --data-root data/dbnet-2018 \\
  --label-provenance path/to/reviewed-evidence.json \\
  --steering-channel 0 --speed-channel 1 \\
  --steering-unit deg --speed-unit km/h \\
  --output results/dbnet-preflight.json

python scripts/train_multimodal.py \\
  --data-root data/dbnet-2018 \\
  --label-provenance path/to/reviewed-evidence.json \\
  --preflight-report results/dbnet-preflight.json \\
  --steering-channel 0 --speed-channel 1 \\
  --steering-unit deg --speed-unit km/h \\
  --assignment A --pool-size 2
```

The optional `--pool-size 2` selects an experimental 2×2 pooled variant. Omit it for the direct spatial Flatten baseline, which has a substantially larger temporal parameter footprint. The commands require an authorized local dataset and genuine supporting evidence; they have not been validated against the full DBNet data.

## Training and checkpointing

Use [the training runner](scripts/train_multimodal.py) to train configuration A or B after independently verifying DBNet target-channel order and physical units. Each run saves its configuration, trainable-stage metadata, validation objectives, and best checkpoint. [Compare the two configurations](scripts/compare_backbones.py) using measured validation objectives; see the [experiment protocol](docs/experiment-protocol.md). Real-data benchmark results are not yet reported.

## Evaluation

Use [validated experiment configurations](src/autonomous_driving/experiment_config.py) and [measured-result reporting](src/autonomous_driving/experiment_reporting.py) for reproducible A/B comparisons; see the [experiment protocol](docs/experiment-protocol.md). Report RMSE and R², and threshold accuracy separately for steering and speed. Include dataset version, split, random seed, preprocessing configuration, checkpoint, and environment with each measured result. Training and benchmark numbers are published only when accompanied by actual run artifacts.

## Research and collaboration

Research interests include autonomous driving, multimodal perception, temporal learning, intelligent vehicles, uncertainty estimation, and vehicle dynamics. Open to academic collaborations and postdoctoral research opportunities.

## License and citation

The project source code is distributed under the [MIT License](LICENSE). For academic use, cite this software using [CITATION.cff](CITATION.cff). When using the DBNet dataset, also cite the original dataset authors via [CITATION_DBNet.bib](CITATION_DBNet.bib). The software license does not grant rights to redistribute third-party datasets.

## Acknowledgment

DBNet dataset credit: Chen et al. (CVPR 2018). Dataset access and redistribution are governed by the original authors' terms.
