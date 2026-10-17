"""Read-only inspection of DBNet behavior.csv storage orientation.

Reports shape and finite numeric ranges without assigning physical units or
assuming a particular label column/row order.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path


def inspect_behavior_csv(path, *, expected_frames=None):
    """Return CSV shape, candidate orientation, and numeric row summaries."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(stream)
            if row and any(cell.strip() for cell in row)
        ]
    if not rows:
        raise ValueError(f"Empty label file: {path}")
    widths = [len(row) for row in rows]
    if len(set(widths)) != 1:
        raise ValueError(f"Nonrectangular label matrix: {path}")
    try:
        values = [[float(cell) for cell in row] for row in rows]
    except ValueError as exc:
        raise ValueError(f"Non-numeric label matrix: {path}") from exc
    if any(not math.isfinite(v) for row in values for v in row):
        raise ValueError(f"Nonfinite label values: {path}")
    nrows, ncols = len(values), widths[0]
    if expected_frames is None:
        orientation = "undetermined"
    elif nrows == expected_frames and ncols == expected_frames:
        orientation = "ambiguous"
    elif nrows == expected_frames:
        orientation = "frames_as_rows"
    elif ncols == expected_frames:
        orientation = "frames_as_columns"
    else:
        orientation = "frame_count_mismatch"
    return {
        "path": str(path),
        "shape": [nrows, ncols],
        "expected_frames": expected_frames,
        "orientation": orientation,
        "row_ranges": [[min(row), max(row)] for row in values],
    }


def extract_raw_frame_labels(path, *, frame_count, orientation):
    """Return numeric frame tuples without attaching target names or units."""
    report = inspect_behavior_csv(path, expected_frames=frame_count)
    if orientation not in {"frames_as_rows", "frames_as_columns"}:
        raise ValueError("Explicit orientation is required")
    if report["orientation"] not in {orientation, "ambiguous"}:
        raise ValueError(
            f"Expected {orientation}, observed {report['orientation']}"
        )
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        matrix = [
            [float(cell.strip()) for cell in row]
            for row in csv.reader(stream)
            if row and any(cell.strip() for cell in row)
        ]
    if orientation == "frames_as_rows":
        return [tuple(row) for row in matrix]
    return [tuple(matrix[channel][frame] for channel in range(len(matrix)))
            for frame in range(frame_count)]
