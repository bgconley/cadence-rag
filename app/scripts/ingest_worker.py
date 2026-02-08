from __future__ import annotations

from redis import Redis
from rq import Connection, Worker

from app.config import settings
from app.logging_utils import configure_logging, get_logger


def main() -> None:
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    connection = Redis.from_url(settings.redis_url)
    logger.info(
        "ingest_worker.start queue=%s redis=%s",
        settings.ingest_queue_name,
        settings.redis_url,
    )
    with Connection(connection):
        worker = Worker([settings.ingest_queue_name])
        worker.work()


if __name__ == "__main__":
    main()
