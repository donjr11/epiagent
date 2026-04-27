"""Scheduler for the nightly ingestion and daily digest jobs.

Runs as a long-lived process on the same VPS as the API. APScheduler handles
triggering; the digest job calls the HTTP endpoint (design §9.3) and forwards
the rendered HTML to SMTP. Ingestion still runs in-process.
"""
from __future__ import annotations

import logging
import signal

import httpx
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from agents.agent1_discovery import run_ingestion
from agents.agent3_notifications import sender
from core import config

log = logging.getLogger(__name__)


def _digest_url() -> str:
    return f"http://{config.SCORING_HOST}:{config.SCORING_PORT}/digest/daily"


def _run_ingestion() -> None:
    log.info("cron: starting ingestion")
    summary = run_ingestion()
    log.info("cron: ingestion summary %s", summary)


def _run_digest() -> None:
    """Hit GET /digest/daily, then mail the HTML to the admin list."""
    url = _digest_url()
    log.info("cron: requesting daily digest from %s", url)
    try:
        resp = httpx.get(url, timeout=120.0)
        resp.raise_for_status()
    except Exception as exc:
        log.exception("cron: digest HTTP request failed: %s", exc)
        return
    err = sender.send_html(
        subject="EpiAgent digest",
        html=resp.text,
        to=config.ADMIN_EMAILS,
    )
    log.info(
        "cron: digest send result sent=%s error=%s html_bytes=%d",
        err is None, err, len(resp.text),
    )


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
