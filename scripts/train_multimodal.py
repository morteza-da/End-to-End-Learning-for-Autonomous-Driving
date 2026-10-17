"""Run a verified-label DBNet multimodal training experiment.

Usage:
  python scripts/train_multimodal.py --data-root data/dbnet-2018 \
    --steering-channel 0 --speed-channel 1 \
    --steering-unit deg --speed-unit km/h --assignment A

Channel and unit flags are REQUIRED and must come from verified dataset
documentation. No raw DBNet files are written to the repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/multimodal"))
    parser.add_argument("--assignment", choices=("A", "B"), required=True)
    parser.add_argument("--steering-channel", type=int, required=True)
    parser.add_argument("--speed-channel", type=int, required=True)
    parser.add_argument("--steering-unit", choices=("deg", "rad"), required=True)
    parser.add_argument("--speed-unit", choices=("km/h", "m/s", "mph"), required=True)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-trainable-last-n", type=int, default=1)
    parser.add_argument("--lidar-trainable-last-n", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from autonomous_driving.backbone_extraction import (
        freeze_all_except_last_n_children, make_multimodal_backbones
    )
    from autonomous_driving.dbnet_layout import build_labeled_observations
    from autonomous_driving.end_to_end import (
        build_end_to_end_model, infer_feature_width, run_multimodal_epoch
    )
    from autonomous_driving.experiment_config import (
        ExperimentConfig, save_experiment_config
    )
    from autonomous_driving.experiment_reporting import (
        seed_everything, write_experiment_report
    )
    from autonomous_driving.sequence_dataset import make_sequence_dataset
    from autonomous_driving.target_mapping import TargetSchema, map_raw_observations

    config = ExperimentConfig(
        assignment=args.assignment, sequence_length=args.sequence_length,
        batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate, seed=args.seed,
        pretrained=not args.no_pretrained,
        camera_trainable_last_n=args.camera_trainable_last_n,
        lidar_trainable_last_n=args.lidar_trainable_last_n,
    )
    schema = TargetSchema(
        steering_channel=args.steering_channel,
        speed_channel=args.speed_channel,
        steering_unit=args.steering_unit, speed_unit=args.speed_unit
    )
    seed_everything(config.seed)
    raw = build_labeled_observations(args.data_root)
    observations = map_raw_observations(raw, schema)
    datasets = {
        split: make_sequence_dataset(
            observations, args.data_root, sequence_length=config.sequence_length,
            stride=config.stride, split=split
        )
        for split in ("train", "val")
    }
    if not all(len(dataset) for dataset in datasets.values()):
        raise ValueError("Train and validation both require nonempty sequence windows")
    loaders = {
        split: DataLoader(dataset, batch_size=config.batch_size,
                          shuffle=(split == "train"), num_workers=0)
        for split, dataset in datasets.items()
    }

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    camera, lidar = make_multimodal_backbones(
        config.assignment, pretrained=config.pretrained
    )
    freeze_info = {
        "camera": freeze_all_except_last_n_children(
            camera, config.camera_trainable_last_n
        ),
        "lidar": freeze_all_except_last_n_children(
            lidar, config.lidar_trainable_last_n
        ),
    }
    camera, lidar = camera.to(device), lidar.to(device)
    width = infer_feature_width(camera, lidar, device=device)
    model = build_end_to_end_model(
        camera, lidar, input_size=width,
        hidden_size=config.hidden_size, num_layers=config.lstm_layers
    ).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate
    )

    output = args.output / config.assignment
    output.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output / "config.json")
    (output / "run_metadata.json").write_text(json.dumps({
        "target_schema": {
            "steering_channel": schema.steering_channel,
            "speed_channel": schema.speed_channel,
            "steering_unit": schema.steering_unit,
            "speed_unit": schema.speed_unit,
        },
        "feature_width": width,
        "trainable_stages": freeze_info,
        "window_counts": {split: len(ds) for split, ds in datasets.items()},
        "device": device,
    }, indent=2) + "\n", encoding="utf-8")

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = run_multimodal_epoch(
            model, loaders["train"], optimizer=optimizer, device=device,
            steering_weight=config.steering_weight,
            speed_weight=config.speed_weight
        )
        val_loss = run_multimodal_epoch(
            model, loaders["val"], device=device,
            steering_weight=config.steering_weight,
            speed_weight=config.speed_weight
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
              flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.as_dict(),
                "feature_width": width,
            }, output / "best.pt")
        write_experiment_report(
            output / "latest_metrics.json", config=config,
            metrics={"train_objective": train_loss,
                     "validation_objective": val_loss,
                     "best_validation_objective": best_val,
                     "epoch": epoch},
            dataset_id="DBNet-2018", split="train+val",
            checkpoint=output / "best.pt",
            notes="Batch-averaged weighted RMSE objectives; not per-target test metrics"
        )


if __name__ == "__main__":
    main()
