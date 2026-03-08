import logging
import time

from app.config import settings
from app.logging import configure_logging
from app.worker import SyncWorker


logger = logging.getLogger("jobl.sync")


def run() -> None:
    configure_logging(settings.log_level)
    worker = SyncWorker(
        source_db_driver=settings.source_db_driver,
        source_db_host=settings.source_db_host,
        source_db_port=settings.source_db_port,
        source_db_user=settings.source_db_user,
        source_db_password=settings.source_db_password,
        target_database_url=settings.target_database_url,
        export_destination=settings.export_destination,
    )

    logger.info(
        "sync worker started interval=%ss batch_size=%s",
        settings.sync_interval_seconds,
        settings.sync_batch_size,
    )
    while True:
        result = worker.run_once(batch_size=settings.sync_batch_size)
        logger.info(
            "sync iteration completed fetched=%s upserted=%s marked_exported=%s",
            result.fetched,
            result.upserted,
            result.marked_exported,
        )
        time.sleep(settings.sync_interval_seconds)


if __name__ == "__main__":
    run()
