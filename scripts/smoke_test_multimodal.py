"""Small synthetic end-to-end smoke test; no DBNet data or pretrained weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_smoke_test(*, sequence_length=3, batch_size=2, seed=42):
    """Exercise camera/LiDAR feature fusion, temporal forward and evaluation."""
    if sequence_length < 2 or batch_size < 1:
        raise ValueError("sequence_length >= 2 and batch_size >= 1 required")
    import torch
    from torch import nn
    from autonomous_driving.end_to_end import build_end_to_end_model, infer_feature_width
    from autonomous_driving.model_evaluation import evaluate_driving_loader

    torch.manual_seed(seed)

    class TinySpatialBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 2, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

        def forward(self, x):
            return self.layers(x)

    camera_backbone = TinySpatialBackbone()
    lidar_backbone = TinySpatialBackbone()
    width = infer_feature_width(camera_backbone, lidar_backbone)
    model = build_end_to_end_model(
        camera_backbone, lidar_backbone, input_size=width,
        hidden_size=8, num_layers=1
    )
    camera = torch.rand(batch_size, sequence_length, 3, 299, 299)
    lidar = torch.rand(batch_size, sequence_length, 3, 299, 299)
    targets = torch.stack(
        (torch.linspace(-5, 5, batch_size),
         torch.linspace(20, 40, batch_size)), dim=1
    )
    with torch.no_grad():
        outputs = model(camera, lidar)
    metrics, predictions = evaluate_driving_loader(
        model, [(camera, lidar, targets)]
    )
    return {
        "data_type": "synthetic",
        "dataset": "none",
        "pretrained_weights": False,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "feature_width": width,
        "attention_shape": list(outputs["attention_weights"].shape),
        "prediction_count": len(predictions["steering_pred_deg"]),
        "metric_names": list(metrics.keys()),
        "performance_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_smoke_test(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size, seed=args.seed
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
