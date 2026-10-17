# DBNet label validation and target mapping

The DBNet dataset has numeric `behavior.csv` values, but raw matrix orientation, channel meanings, and physical units must be checked independently before supervised training.

## Inspect the matrix

```python
from autonomous_driving.label_inspection import inspect_behavior_csv
report = inspect_behavior_csv(
    "data/dbnet-2018/train/1/behavior.csv", expected_frames=120
)
print(report["shape"], report["orientation"], report["row_ranges"])
```

Supported layouts: `frames_as_rows` (N × channels) and `frames_as_columns` (channels × N). An N × N matrix is ambiguous; choose its orientation explicitly with `extract_raw_frame_labels` after verifying it. A mismatch or nonnumeric/nonfinite value raises an error.

## Explicit target schema

After confirming channel definitions and units from authoritative DBNet metadata:

```python
from autonomous_driving.dbnet_layout import build_labeled_observations
from autonomous_driving.target_mapping import TargetSchema, map_raw_observations

raw = build_labeled_observations("data/dbnet-2018")
# Replace these EXAMPLE settings only with independently verified definitions.
schema = TargetSchema(
    steering_channel=0, speed_channel=1,
    steering_unit="deg", speed_unit="km/h"
)
observations = map_raw_observations(raw, schema)
```

**The example schema is not a verified description of DBNet.** Do not execute training with it unless the source documentation confirms both channel positions and units. The mapping supports degrees/radians and km/h, m/s, or mph, converting model targets to degrees and km/h.

The DBNet README describes CSV lines indexed by frame. Separate dataset inspection has also found matrices shaped 2 × 120; therefore file structure must be checked rather than inferred from the README alone. No performance metrics are reported until data alignment, target semantics, and model execution are validated.
