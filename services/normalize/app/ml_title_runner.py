import argparse
import csv
import logging
from pathlib import Path

from app.logging import configure_logging
from app.worker import NormalizeWorker


logger = logging.getLogger("jobl.normalize.ml_title")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML-only title normalization (keeps API/frontend normalization untouched)"
    )
    parser.add_argument("--text", type=str, default=None, help="Normalize one title and print result")
    parser.add_argument("--input-csv", type=str, default=None, help="Input CSV file path")
    parser.add_argument("--output-csv", type=str, default=None, help="Output CSV file path")
    parser.add_argument(
        "--title-column",
        type=str,
        default="title_raw",
        help="CSV column name containing source title",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")

    try:
        if args.text:
            print(NormalizeWorker.normalize_title_for_ml(args.text))
            return 0

        if not args.input_csv or not args.output_csv:
            raise SystemExit("Either --text or (--input-csv and --output-csv) must be provided")

        input_path = Path(args.input_csv)
        output_path = Path(args.output_csv)
        _run_csv(input_path=input_path, output_path=output_path, title_column=args.title_column)
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    return 0


def _run_csv(input_path: Path, output_path: Path, title_column: str) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        if title_column not in (reader.fieldnames or []):
            raise SystemExit(f"Column '{title_column}' not found in {input_path}")

        fieldnames = list(reader.fieldnames or [])
        if "title_ml" not in fieldnames:
            fieldnames.append("title_ml")

        with output_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            row_count = 0
            for row in reader:
                row["title_ml"] = NormalizeWorker.normalize_title_for_ml(row.get(title_column) or "")
                writer.writerow(row)
                row_count += 1

    logger.info("ml title normalization completed rows=%s output=%s", row_count, output_path)


if __name__ == "__main__":
    run()
