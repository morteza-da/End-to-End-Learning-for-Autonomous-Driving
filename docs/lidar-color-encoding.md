# LiDAR Distance-to-RGB Reconstruction

## Dissertation evidence

**Source:** supplied 156-page PhD dissertation, PDF pages 73–74, Equation (3-1) and Figure (3-5).

The dissertation describes conversion of 3D LiDAR point clouds to a 2D RGB representation. Its printed distance equation is:

`distance = sqrt(Px^2 + Py^2)`

The piecewise color equation specifies distance regions below 20, below 40, and otherwise, with blue/green/red channel scales of 20/40/60.

## Implemented interpretation

| Distance | Output RGB tuple | Active channel |
| --- | --- | --- |
| 0 <= d < 20 | (0, 0, round(255 * d / 20)) | Blue |
| 20 <= d < 40 | (0, round(255 * d / 40), 0) | Green |
| d >= 40 | (round(min(255, 255 * d / 60)), 0, 0) | Red |

This implements the **printed piecewise expression as interpreted**, with inactive channels zero. The integer rounding convention is chosen for the reconstruction and is not independently verified against the lost source code.

The values 20, 40, and 60 are treated as using the distance units of the input coordinates; verify the original LiDAR coordinate units before use.

## Scope and reproducibility limitations

`src/autonomous_driving/lidar_encoding.py` transforms point coordinates to RGB **colors only**. It does **not** implement point-cloud projection, pixel coordinate assignment, occlusion handling, empty-pixel filling, camera calibration, or the final 299x299 resize. The dissertation passage does not specify enough detail to reproduce those steps faithfully.

Do **not** present the color-only output as the dissertation's completed 2D LiDAR image preprocessing pipeline.

The implementation is standard-library-only and has synthetic unit tests. No original LiDAR dataset or historical research results are included.

Run:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```
