"""Email Agent — sends CV as attachment via Gmail OAuth2, notifies George via Telegram."""
import base64
import logging
import mimetypes
import os
import re
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import httpx

from agents.telegram_agent import send_message

logger = logging.getLogger(__name__)

_GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL_SEND_URL  = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
_CV_PATH = Path(__file__).parent.parent / "images" / "RusuGeorgeCV.pdf"

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _gmail_credentials_ok() -> bool:
    ok = all([
        os.getenv("GMAIL_CLIENT_ID"),
        os.getenv("GMAIL_CLIENT_SECRET"),
        os.getenv("GMAIL_REFRESH_TOKEN"),
    ])
    if not ok:
        logger.warning("Gmail credentials not set — email cannot be sent.")
    return ok


async def _get_access_token() -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            _GMAIL_TOKEN_URL,
            data={
                "client_id":     os.getenv("GMAIL_CLIENT_ID"),
                "client_secret": os.getenv("GMAIL_CLIENT_SECRET"),
                "refresh_token": os.getenv("GMAIL_REFRESH_TOKEN"),
                "grant_type":    "refresh_token",
            },
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


def _build_cv_email(to_addr: str) -> str:
    """Build a base64url-encoded raw RFC 2822 message with cv.pdf attached."""
    msg = MIMEMultipart()
    msg["To"]      = to_addr
    msg["From"]    = "George Rusu <georgel1988@gmail.com>"
    msg["Subject"] = "George Rusu — CV"

    body = MIMEText(
        "Hi,\n\n"
        "Please find George Rusu's CV attached.\n\n"
        "Feel free to reach out if you have any questions.\n\n"
        "Best regards,\nGeorge Rusu\n"
        "georgel1988@gmail.com | linkedin.com/in/rusugeorge",
        "plain",
    )
    msg.attach(body)

    if _CV_PATH.exists():
        with open(_CV_PATH, "rb") as f:
            attachment = MIMEApplication(f.read(), _subtype="pdf")
        attachment.add_header(
            "Content-Disposition", "attachment", filename="RusuGeorgeCV.pdf"
        )
        msg.attach(attachment)
    else:
        logger.warning("cv.pdf not found at %s — sending email without attachment", _CV_PATH)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return raw


async def send_cv(to_email: str) -> bool:
    """Send the CV to *to_email* via the Gmail API. Returns True on success."""
    if not _gmail_credentials_ok():
        return False
    access_token = await _get_access_token()
    raw = _build_cv_email(to_email)
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            _GMAIL_SEND_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            json={"raw": raw},
        )
        resp.raise_for_status()
    return True


async def run_email(
    query: str,
    on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    on_telegram: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> str:
    if on_progress:
        await on_progress("Looking for your email address…")

    visitor_email = _extract_email(query)
    if not visitor_email:
        return (
            "I'd be happy to send George's CV to you! "
            "Please share your email address so I can deliver it.\n"
            "Example: \"Send the CV to name@example.com\""
        )

    if on_progress:
        await on_progress(f"Sending CV to {visitor_email}…")

    sent = await send_cv(visitor_email)

    if sent:
        try:
            if on_telegram:
                await on_telegram(f"Notifying George via Telegram…")
            await send_message(
                f"📎 <b>CV sent</b> to <code>{visitor_email}</code>\n"
                "Via: Lush AI"
            )
            if on_telegram:
                await on_telegram("George notified.")
        except Exception:
            logger.warning("Telegram notification failed — email was still sent.")
        return (
            f"George's CV has been sent to {visitor_email}. "
            "Please check your inbox (and spam folder just in case). "
            "Feel free to reach out if you have any questions!"
        )

    # Fallback — Gmail creds missing or send failed
    if on_progress:
        await on_progress("Email service unavailable — providing alternatives…")
    return (
        "I wasn't able to send the email right now. Here are a few ways to get George's CV:\n\n"
        "• Download directly: https://georgelush.github.io/Portfolio/images/RusuGeorgeCV.pdf\n"
        "• Email George: georgel1988@gmail.com\n"
        "• LinkedIn DM: linkedin.com/in/rusugeorge"
    )
