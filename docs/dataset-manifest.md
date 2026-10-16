# Dataset Manifest Contract (Draft)

The reconstructed project uses an explicit CSV manifest rather than assuming a particular directory structure.

Required columns:

| Column | Meaning |
| --- | --- |
| sample_id | Unique observation identifier |
| split | train, val, or test |
| camera_path | Relative path to camera data |
| lidar_path | Relative path to LiDAR data |
| steering | Numeric steering target; **unit to be confirmed** |
| speed | Numeric speed target; **unit to be confirmed** |

Example **synthetic** record (not from the dissertation):

```csv
sample_id,split,camera_path,lidar_path,steering,speed
synthetic-001,train,camera/001.png,lidar/001.bin,0.0,10.0
```

The validation utility checks column presence, unique IDs, split values, finite labels, path safety, and optionally that sensor files exist. It **does not** prove camera–LiDAR synchronization, calibration, temporal independence, leakage-free splits, label units, or data quality.

Before training, confirm the actual dissertation dataset format, timestamps, calibration, steering convention, speed units, sequence construction, and licensing. Do not publish private or restricted data. Never represent synthetic examples as historical research samples.

Local use after `python -m pip install -e .`:

```python -c "from autonomous_driving.data_manifest import validate_manifest; print(validate_manifest('manifest.csv', 'dataset_root'))"```

Run synthetic unit tests with `python -m unittest discover -s tests -v`.
