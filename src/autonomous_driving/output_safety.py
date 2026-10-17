"""Protect local datasets from accidental experiment-output writes."""
from pathlib import Path


def require_output_outside_dataset(data_root, output_root):
    dataset = Path(data_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError(
            "Experiment output must be outside the raw dataset directory"
        )
    return output
