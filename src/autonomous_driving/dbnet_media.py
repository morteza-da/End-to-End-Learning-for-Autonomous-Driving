"""Load verified DBNet JPEG/LAS pairs without inventing target semantics.

Optional runtime dependencies: Pillow, numpy, laspy. No data are redistributed.
"""
from __future__ import annotations
from pathlib import Path


def load_camera(path: str | Path, size: tuple[int, int] = (299, 299)):
    """Return RGB float32 CHW image in [0,1].

    Resizing to 299x299 is documented by the researcher; interpolation is a
    disclosed reconstruction choice (Pillow bilinear).
    """
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB").resize(size, resample=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)


def load_lidar_xyz(path: str | Path):
    """Read real-world XYZ, including LAS header scale and offset via laspy."""
    import numpy as np
    import laspy
    cloud = laspy.read(str(path))
    xyz = np.column_stack((cloud.x, cloud.y, cloud.z)).astype(np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid LiDAR XYZ values in {path}")
    return xyz


def verify_frame_pair(camera_path: str | Path, lidar_path: str | Path):
    """Decode both sensor files; return metadata, not a training sample."""
    from PIL import Image
    with Image.open(camera_path) as image:
        image.verify()
    with Image.open(camera_path) as image:
        size = image.size
        mode = image.mode
    xyz = load_lidar_xyz(lidar_path)
    return {"camera_size": size, "camera_mode": mode,
            "lidar_point_count": int(xyz.shape[0]),
            "xyz_finite": True}
