import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from app.io_utils import read_jsonl, write_jsonl
from app.logging import configure_logging


logger = logging.getLogger("jobl.training.split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split labeled JSONL into train/val/test (stratified by language_code)")
    parser.add_argument("--in", dest="input_path", type=str, default="data/raw/labeled.jsonl", help="Input JSONL path")
    parser.add_argument("--out-dir", type=str, default="data/splits", help="Output directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging("INFO")

    if round(args.train + args.val + args.test, 6) != 1.0:
        raise SystemExit("--train + --val + --test must equal 1.0")

    try:
        rows = read_jsonl(Path(args.input_path))
        if not rows:
            raise SystemExit("No rows found in input dataset")

        train_rows, val_rows, test_rows = _stratified_split(
            rows=rows,
            train_ratio=args.train,
            val_ratio=args.val,
            seed=args.seed,
        )

        out_dir = Path(args.out_dir)
        write_jsonl(out_dir / "train.jsonl", train_rows)
        write_jsonl(out_dir / "val.jsonl", val_rows)
        write_jsonl(out_dir / "test.jsonl", test_rows)

        manifest = {
            "input_rows": len(rows),
            "seed": args.seed,
            "ratios": {"train": args.train, "val": args.val, "test": args.test},
            "counts": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
            "language_counts": {
                "train": _language_counts(train_rows),
                "val": _language_counts(val_rows),
                "test": _language_counts(test_rows),
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130

    logger.info(
        "split completed input=%s train=%s val=%s test=%s out_dir=%s",
        len(rows),
        len(train_rows),
        len(val_rows),
        len(test_rows),
        args.out_dir,
    )
    return 0


def _stratified_split(
    *,
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        lang = str(row.get("language_code") or "").strip()
        by_lang.setdefault(lang, []).append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for lang_rows in by_lang.values():
        rng.shuffle(lang_rows)
        total = len(lang_rows)

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        if total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= total:
                n_val = max(1, total - n_train - 1)

        n_test = total - n_train - n_val
        if n_test < 0:
            n_test = 0

        train_rows.extend(lang_rows[:n_train])
        val_rows.extend(lang_rows[n_train : n_train + n_val])
        test_rows.extend(lang_rows[n_train + n_val :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def _language_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        lang = str(row.get("language_code") or "")
        counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(run())
