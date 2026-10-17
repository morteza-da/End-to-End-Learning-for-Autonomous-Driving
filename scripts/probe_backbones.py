"""Probe installed timm backbone shapes and LSTM parameter-state estimates."""
import argparse
import json
from pathlib import Path

from autonomous_driving.backbone_probe import probe_backbones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", choices=["A", "B"], default="A")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true",
                        help="Download pretrained weights if available (default: disabled)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-temporal-adam-mib", type=float,
                        help="Fail if temporal parameter/Adam estimate exceeds this cap")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe_backbones(
        args.assignment, pretrained=args.pretrained,
        hidden_size=args.hidden_size, num_layers=args.lstm_layers,
        device=args.device
    ).as_dict()
    if args.max_temporal_adam_mib is not None:
        if args.max_temporal_adam_mib <= 0:
            parser.error("--max-temporal-adam-mib must be positive")
        if result["temporal_adam_state_mib"] > args.max_temporal_adam_mib:
            parser.error(
                f"Temporal parameter/Adam estimate {result['temporal_adam_state_mib']:.1f} MiB "
                f"exceeds cap {args.max_temporal_adam_mib:.1f} MiB; "
                "backbone and activation memory are additional."
            )
    result["pretrained_weights_requested"] = args.pretrained
    result["input_shape"] = [1, 3, 299, 299]
    result["total_training_memory_measured"] = False
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
