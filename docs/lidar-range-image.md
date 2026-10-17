# LiDAR spherical range-image processing

This implementation transforms LAS point clouds into distance-coded RGB images suitable for the LiDAR CNN branch.

**Coordinate mapping.** For each finite LiDAR-frame point `(x,y,z)`, calculate azimuth `atan2(y,x)` and elevation `atan2(z,sqrt(x²+y²))`. Map angles linearly into the configured image field of view, with larger elevations at the top. Default output resolution is **299 × 299**; the example field of view is azimuth **−180° to 180°** and elevation **−25° to 15°**. These angular limits are configurable processing parameters and should be checked against the sensor's actual coverage.

**Color mapping.** Planar distance `d=sqrt(x²+y²)` is encoded as blue for `d<20`, green for `20≤d<40`, and red for `d≥40`, with channel scaling divisors 20, 40, and 60 respectively. The values use the units supplied by LAS coordinates.

**Visibility.** Invalid or out-of-field points are excluded. For colliding returns, the nearest 3D point is selected. Empty pixels are black. The returned occupancy mask identifies measured pixels separately from black image values. This is a **spherical LiDAR range image**, not a camera-view projection or a calibrated image-to-image alignment.

**Code:** `src/autonomous_driving/lidar_projection.py`. **Tests:** `tests/test_lidar_projection.py`. Install with `python -m pip install -e '.[data]'`.

```python
from autonomous_driving.lidar_projection import load_lidar_rgb, RangeImageConfig
rgb_chw, occupied = load_lidar_rgb(
    "data/dbnet-2018/train/1/points_16384/0.las",
    RangeImageConfig(width=299, height=299),
)
print(rgb_chw.shape, occupied.sum())
```

The exact camera–LiDAR calibration, angle limits, and model preprocessing statistics should be recorded with each experiment. Raw dataset content is not committed to the repository. Cite Chen et al. (CVPR 2018) for DBNet.

## Paired camera and LiDAR inputs

Use `src/autonomous_driving/sensor_pair.py` to load synchronized camera JPEG and LiDAR LAS frames into two CHW float32 RGB arrays. Both inputs have matching spatial dimensions and are ready for backbone-specific normalization and temporal batching.

```python
from autonomous_driving.sensor_pair import load_indexed_pair
sample = load_indexed_pair("data/dbnet-2018/train/1", 0)
print(sample["camera"].shape, sample["lidar"].shape)
```

The returned LiDAR occupancy mask is available for diagnostics. Behavior labels must be mapped to steering/speed only after verifying the CSV schema and units.
