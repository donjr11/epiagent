"""SMTP sender. Errors are logged and returned so the orchestrator can
surface them in the dashboard (per design doc §9.3)."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional

from core import config

log = logging.getLogger(__name__)


def send_html(subject: str, html: str, to: list[str]) -> Optional[str]:
    """Send an HTML email. Returns None on success or an error string."""
    if not to:
        return "no recipients configured (ADMIN_EMAILS)"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.SMTP_FROM
    msg["To"] = ", ".join(to)
    msg.set_content("HTML email — see HTML part.")
    msg.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as s:
            if config.SMTP_PORT in (587, 25):
                try:
                    s.starttls()
                except Exception:
                    pass  # not all local relays support STARTTLS
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASS)
            s.send_message(msg)
        log.info("Sent digest to %d recipient(s)", len(to))
        return None
    except Exception as exc:
        log.exception("SMTP send failed")
        return f"{type(exc).__name__}: {exc}"
