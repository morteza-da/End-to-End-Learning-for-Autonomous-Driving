"""Lazy multimodal sequence batches from a validated observation manifest."""
from __future__ import annotations

from pathlib import Path

from .sequence_manifest import validate_observations, build_sequence_manifest
from .lidar_projection import RangeImageConfig


def _resolve_under_root(root: Path, relative: str) -> Path:
    """Resolve a manifest path and reject traversal outside the dataset root."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return candidate


def make_sequence_dataset(observations, dataset_root, *, sequence_length,
                          stride=1, split="train", lidar_config=None,
                          normalize_imagenet=True):
    """Return a PyTorch Dataset of (camera[T,3,299,299],
    lidar[T,3,299,299], target[2]).

    Input manifest must specify steering_deg and speed_kmh with verified units.
    This loader never guesses raw behavior.csv column positions or target units.
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from .sensor_pair import load_sensor_pair

    if split not in {"train", "val", "test"}:
        raise ValueError("Invalid split")
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    config = lidar_config or RangeImageConfig()
    if (config.width, config.height) != (299, 299):
        raise ValueError("Current CNN backbones require 299x299 input")
    windows = [
        window for window in build_sequence_manifest(
            validate_observations(observations),
            sequence_length=sequence_length, stride=stride
        ) if window[0].split == split
    ]
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    from .dataset_fingerprint import fingerprint_windows
    membership_sha256 = fingerprint_windows(windows) if windows else None

    class DBNetSequenceDataset(Dataset):
        window_membership_sha256 = membership_sha256
        def __len__(self):
            return len(windows)

        def __getitem__(self, index):
            window = windows[index]
            cameras, lidars = [], []
            for observation in window:
                camera_path = _resolve_under_root(root, observation.camera_path)
                lidar_path = _resolve_under_root(root, observation.lidar_path)
                pair = load_sensor_pair(camera_path, lidar_path, lidar_config=config)
                camera, lidar = pair["camera"], pair["lidar"]
                if normalize_imagenet:
                    camera = (camera - mean) / std
                    lidar = (lidar - mean) / std
                cameras.append(torch.from_numpy(np.ascontiguousarray(camera)))
                lidars.append(torch.from_numpy(np.ascontiguousarray(lidar)))
            target = torch.tensor(
                [window[-1].steering_deg, window[-1].speed_kmh],
                dtype=torch.float32
            )
            return torch.stack(cameras), torch.stack(lidars), target

    return DBNetSequenceDataset()
