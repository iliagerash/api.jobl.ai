import argparse
import logging

from app.config import settings
from app.logging import configure_logging
from app.worker import NormalizeWorker


logger = logging.getLogger("jobl.normalize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jobl normalization worker")
    parser.add_argument("--batch-size", type=int, default=None, help="Override NORMALIZE_BATCH_SIZE")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches")
    parser.add_argument("--from-id", type=int, default=0, help="Start processing from id > value")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    batch_size = args.batch_size or settings.normalize_batch_size

    configure_logging(settings.log_level)
    worker = NormalizeWorker(target_database_url=settings.target_database_url)

    logger.info(
        "normalize worker started batch_size=%s max_batches=%s from_id=%s",
        batch_size,
        args.max_batches,
        args.from_id,
    )
    result = worker.run(
        batch_size=batch_size,
        max_batches=args.max_batches,
        from_id=args.from_id,
    )
    logger.info(
        "normalize run completed batches=%s scanned=%s updated=%s",
        result.batches,
        result.scanned,
        result.updated,
    )


if __name__ == "__main__":
    run()
