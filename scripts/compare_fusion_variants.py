"""Build a CSV table from existing, compatible validation_metrics.json files."""
import argparse
from autonomous_driving.experiment_comparison import (
    load_comparison_rows, write_comparison_csv
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Validation metrics JSON files")
    parser.add_argument("--output", default="results/fusion-comparison.csv")
    args = parser.parse_args()
    rows = load_comparison_rows(args.reports)
    write_comparison_csv(rows, args.output)
    print(f"Wrote {len(rows)} measured variant(s) to {args.output}")


if __name__ == "__main__":
    main()
