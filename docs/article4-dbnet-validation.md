# Article 4: DBNet validation workflow

This folder documents the new **physics-aware evaluation** workflow. It is separate from model training. **No model retraining or 586-window inference is claimed here.**

## Data preparation

The original DBNet archive stays outside Git. To run in Colab, mount the Google Drive account holding `MyDrive/DBNet/dbnet-2018.zip`, then:

```bash
python scripts/article4_audit_zip.py /content/drive/MyDrive/DBNet/dbnet-2018.zip --output /content/drive/MyDrive/DBNet/Article4_Full_Dataset_Audit
python scripts/article4_build_manifest.py /content/drive/MyDrive/DBNet/dbnet-2018.zip --split val --sequence-length 4 --stride 4 --output /content/drive/MyDrive/DBNet/Article4_Full_Dataset_Audit
```

The first script audits file counts and frame-index pairing **without extracting** media. The second generates consecutive, non-overlapping four-frame windows and attaches the **last frame's** steering/speed label using its original frame index. It excludes windows missing either sensor. It does not prove timestamp synchronization, sensor calibration, steering-wheel-versus-road-wheel convention, or training/validation leakage.

## Observations from the 2026-11-02 Colab audit

| Split | Drives | Camera frames | LiDAR frames | Paired frame IDs |
|---|---:|---:|---:|---:|
| Train | 56 | 6,625 | 6,569 | 6,569 |
| Validation | 20 | 2,400 | 2,349 | 2,349 |
| Test | 20 | 2,400 | 2,376 | 2,376 |

The validation manifest contained **586 windows from 20 drives**, with sequence length 4 and stride 4. These numbers were observed in a user-run Colab session and are not outputs of GitHub CI. The validation labels included speeds from -4 to 62 km/h and steering values from -140 to 180 degrees. Negative speed and steering conventions require source-level review; do not silently clip them.

The original checkpoint was recovered at `MyDrive/DBNet/gpu_pilot_backup/A/pool-2x2/best.pt` (about 342.79 MB); it was **loaded and inspected**, not retrained. The previous 15-window pilot must not be presented as the new 586-window inference result.

## Scientific checks before publication

- Verify source label units and the target mapping, including negative speeds.
- Audit train/validation drive identity and checkpoint provenance.
- Run the checkpoint on all manifest windows with the **original training preprocessing**.
- Compare zero-steering, mean, and persistence baselines.
- Report per-drive results and drive-level bootstrap confidence intervals.
- Evaluate stop-state speed errors, friction-margin sensitivity and controlled positive violations.
- Distinguish a numerical feasibility estimate from a certified lower bound; do not call an uncertified bound a safety guarantee.
- Keep raw DBNet media, checkpoints, and potentially sensitive user Drive paths outside the repository.

## Reuse

Scripts require Python 3.9+ standard library only. Manifest paths refer to ZIP members, not extracted files. Subsequent inference code must open those members from the original ZIP and use the repository's existing image/LiDAR preprocessing.
