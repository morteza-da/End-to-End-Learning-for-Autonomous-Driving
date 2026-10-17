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


def require_fresh_experiment_directory(output_dir, *, allow_existing=False):
    """Avoid silently replacing checkpoints and metrics from a previous run."""
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not allow_existing:
        raise ValueError(
            "Experiment output directory is not empty; select a fresh --output "
            "or explicitly pass --overwrite-output"
        )
    return output
