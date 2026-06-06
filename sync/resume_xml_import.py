import argparse
import logging

from sync.logging import configure_logging
from sync.resume_config import settings
from sync.resume_xml_worker import ResumeXmlWorker

logger = logging.getLogger("jobl.sync.resume_xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import resume XML feeds into PostgreSQL")
    parser.add_argument(
        "--all-files",
        action="store_true",
        default=False,
        help="Process every XML file in RESUME_XML_FEEDS_DIR, not only files changed in the lookback window.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Process only this XML filename (repeatable). Example: --file=au.workus.org.xml",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=None,
        help="Override RESUME_XML_LOOKBACK_HOURS (default: 24).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override RESUME_XML_BATCH_SIZE (default: 500).",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    configure_logging(settings.log_level)

    worker = ResumeXmlWorker(
        target_database_url=settings.database_url,
        feeds_dir=settings.resume_xml_feeds_dir,
        lookback_hours=args.lookback_hours if args.lookback_hours is not None else settings.resume_xml_lookback_hours,
    )

    only_files = set(args.files or [])
    batch_size = args.batch_size if args.batch_size is not None else settings.resume_xml_batch_size

    try:
        result = worker.run_once(
            batch_size=batch_size,
            all_files=args.all_files,
            only_files=only_files or None,
        )
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    except FileNotFoundError:
        logger.exception("resume xml feeds directory is missing or invalid")
        return 1

    logger.info(
        "resume xml import completed files_scanned=%s files_processed=%s parsed=%s inserted=%s skipped_existing=%s skipped_invalid=%s",
        result.files_scanned,
        result.files_processed,
        result.parsed,
        result.inserted,
        result.skipped_existing,
        result.skipped_invalid,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
