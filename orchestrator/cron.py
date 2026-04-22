"""Scheduler for the nightly ingestion and daily digest jobs.

Runs as a long-lived process on the same VPS as the API. APScheduler handles
triggering; each job just calls into the agent module.
"""
from __future__ import annotations

import logging
import signal

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from agents.agent1_discovery import run_ingestion
from agents.agent3_notifications import send_daily_digest

log = logging.getLogger(__name__)


def _run_ingestion() -> None:
    log.info("cron: starting ingestion")
    summary = run_ingestion()
    log.info("cron: ingestion summary %s", summary)


def _run_digest() -> None:
    log.info("cron: starting daily digest")
    result = send_daily_digest()
    log.info("cron: digest result %s", result)


def build_scheduler() -> BlockingScheduler:
    sched = BlockingScheduler(timezone="UTC")
    # Nightly ingestion 02:00 UTC
    sched.add_job(_run_ingestion, CronTrigger(hour=2, minute=0), id="ingestion")
    # Daily digest 07:00 UTC
    sched.add_job(_run_digest, CronTrigger(hour=7, minute=0), id="digest")
    return sched


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sched = build_scheduler()

    def _shutdown(_sig, _frame):
        log.info("cron: shutting down")
        sched.shutdown(wait=False)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    log.info("cron: started; jobs=%s", [j.id for j in sched.get_jobs()])
    sched.start()


if __name__ == "__main__":
    main()
