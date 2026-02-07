from __future__ import annotations

import argparse
import time

from app.config import settings
from app.ingest_fs import scan_inbox_once


def main() -> None:
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
        summary = scan_inbox_once()
        print(
            "[ingest_scanner] "
            f"discovered={summary['discovered']} "
            f"queued={summary['queued']} "
            f"duplicates={summary['duplicates']} "
            f"invalid={summary['invalid']}"
        )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
