from __future__ import annotations

import argparse
import time

from app.config import settings
from app.ingest_fs import scan_inbox_once
from app.logging_utils import configure_logging, get_logger


def main() -> None:
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    parser = argparse.ArgumentParser(
        description="Scan ingest inbox for ready bundles and enqueue them."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan pass and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=settings.ingest_poll_seconds,
        help="Polling interval in seconds when not using --once.",
    )
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")

    while True:
        try:
            summary = scan_inbox_once()
            logger.info(
                "ingest_scanner.pass discovered=%s queued=%s duplicates=%s invalid=%s",
                summary["discovered"],
                summary["queued"],
                summary["duplicates"],
                summary["invalid"],
            )
        except Exception as exc:  # pragma: no cover - runtime hardening for service loop
            logger.exception("ingest_scanner.pass_failed error=%s", str(exc))
            if args.once:
                raise
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
