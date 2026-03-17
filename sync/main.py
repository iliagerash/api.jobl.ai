import argparse
import logging

from sync.config import settings
from sync.logging import configure_logging
from sync.worker import SyncWorker


logger = logging.getLogger("jobl.sync")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jobl sync worker")
    parser.add_argument(
        "--db",
        action="append",
        dest="dbs",
        help="Source DB name to sync (repeatable). Example: --db=americas --db=australia",
    )
    parser.add_argument(
        "--country",
        dest="countries",
        default=None,
        help="Comma-separated country codes to sync. Example: --country=us,ca",
    )
    parser.add_argument(
        "--resync",
        action="store_true",
        default=False,
        help="Re-sync from the beginning, ignoring export_ai filter and resetting last_job_id.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    only_dbs = set(args.dbs or [])
    only_countries = {c.strip().upper() for c in args.countries.split(",")} if args.countries else None
    configure_logging(settings.log_level)
    worker = SyncWorker(
        source_db_driver=settings.source_db_driver,
        source_db_host=settings.source_db_host,
        source_db_port=settings.source_db_port,
        source_db_user=settings.source_db_user,
        source_db_password=settings.source_db_password,
        source_db_ssl_disabled=settings.source_db_ssl_disabled,
        target_database_url=settings.database_url,
        export_destination=settings.export_destination,
    )

    logger.info(
        "sync worker started batch_size=%s",
        settings.sync_batch_size,
    )
    if only_dbs:
        logger.info("db filter enabled dbs=%s", ",".join(sorted(only_dbs)))
    if only_countries:
        logger.info("country filter enabled countries=%s", ",".join(sorted(only_countries)))
    if args.resync:
        logger.info("resync mode enabled — export_ai filter skipped, last_job_id reset to 0")
    try:
        result = worker.run_once(
            batch_size=settings.sync_batch_size,
            only_dbs=only_dbs,
            only_countries=only_countries,
            resync=args.resync,
        )
    except KeyboardInterrupt:
        logger.warning("interrupted by user (Ctrl+C), exiting gracefully")
        return 130
    logger.info(
        "sync run completed fetched=%s upserted=%s marked_exported=%s",
        result.fetched,
        result.upserted,
        result.marked_exported,
    )
    return 0


if __name__ == "__main__":
    run()
