"""Calendar Agent — Human-in-the-Loop via Telegram Inline Keyboard.

Flow:
  1. Collect 5 required fields from visitor via chat (multi-turn).
  2. Store pending meeting in Upstash Redis (TTL 24h).
  3. Send George a Telegram message with ✅ Approve / ❌ Reject buttons.
  4. Tell visitor: "Request sent — confirmation coming to {email}."
  5. On Approve (webhook): create Google Calendar event + Meet link → email visitor.
  6. On Reject (webhook): email visitor with polite message → delete from Redis.
"""
import json
import logging
import os
import uuid
from typing import Any, Callable, Coroutine, Optional

import httpx

from agents.email_agent import _get_access_token, _gmail_credentials_ok
from agents.telegram_agent import edit_message, send_message, send_message_with_buttons
from portfolio_config import ASSISTANT_NAME, CONTACT_EMAIL

logger = logging.getLogger(__name__)

_UPSTASH_URL   = lambda: os.getenv("UPSTASH_REDIS_REST_URL", "")
_UPSTASH_TOKEN = lambda: os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
_MEETING_TTL   = 86400  # 24 hours


# ── Upstash Redis REST helpers ────────────────────────────────────────────────

def _upstash_ok() -> bool:
    ok = bool(_UPSTASH_URL() and _UPSTASH_TOKEN())
    if not ok:
        logger.warning("Upstash Redis credentials not set.")
    return ok


def _upstash_headers() -> dict:
    return {"Authorization": f"Bearer {_UPSTASH_TOKEN()}"}


async def _upstash_command(*args) -> any:
    """Execute an Upstash Redis REST command using the pipeline-style POST body."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            _UPSTASH_URL(),
            headers=_upstash_headers(),
            json=list(args),
        )
        resp.raise_for_status()
        return resp.json().get("result")


async def redis_set(key: str, value: dict, ttl: int = _MEETING_TTL) -> bool:
    """SET key (JSON) with EX ttl via Upstash REST API."""
    if not _upstash_ok():
        return False
    await _upstash_command("SET", key, json.dumps(value), "EX", ttl)
    return True


async def redis_get(key: str) -> Optional[dict]:
    """GET key and decode JSON from Upstash REST API."""
    if not _upstash_ok():
        return None
    result = await _upstash_command("GET", key)
    if result is None:
        return None
    return json.loads(result)


async def redis_delete(key: str) -> bool:
    """DEL key via Upstash REST API."""
    if not _upstash_ok():
        return False
    await _upstash_command("DEL", key)
    return True


# ── Google Calendar (OAuth2 — same credentials as Gmail) ─────────────────────

async def _create_calendar_event(meeting: dict) -> Optional[str]:
    """Create a Google Calendar event + Meet link using OAuth2 refresh token."""
    if not _gmail_credentials_ok():
        logger.warning("Gmail/Calendar OAuth2 credentials not set.")
        return None

    try:
        from datetime import datetime, timedelta

        access_token = await _get_access_token()

        # Parse time and add 1 hour for end time
        start_dt = datetime.strptime(
            f"{meeting['preferred_date']}T{meeting['preferred_time']}:00",
            "%Y-%m-%dT%H:%M:%S"
        )
        end_dt = start_dt + timedelta(hours=1)

        event_body = {
            "summary":     f"Meeting with {meeting['visitor_name']}: {meeting['meeting_topic']}",
            "description": f"Scheduled via {ASSISTANT_NAME}\nVisitor: {meeting['visitor_email']}",
            "start": {"dateTime": start_dt.isoformat(), "timeZone": "Europe/Bucharest"},
            "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "Europe/Bucharest"},
            "attendees": [{"email": meeting["visitor_email"]}],
            "conferenceData": {"createRequest": {"requestId": str(uuid.uuid4())}},
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events?conferenceDataVersion=1",
                headers={"Authorization": f"Bearer {access_token}"},
                json=event_body,
            )
            resp.raise_for_status()
            event = resp.json()

        # Extract Meet link
        meet_link = None
        for ep in event.get("conferenceData", {}).get("entryPoints", []):
            if ep.get("entryPointType") == "video":
                meet_link = ep.get("uri")
                break
        return meet_link or event.get("htmlLink")

    except Exception:
        logger.exception("Failed to create Google Calendar event")
        return None


# ── Email helpers (using Gmail via email_agent internals) ─────────────────────

async def _send_confirmation_email(meeting: dict, meet_link: str) -> None:
    """Send meeting confirmation to visitor."""
    import base64
    from email.mime.text import MIMEText

    if not _gmail_credentials_ok():
        logger.warning("Gmail creds not set — cannot send confirmation email.")
        return

    body = (
        f"Hi {meeting['visitor_name']},\n\n"
        f"Your meeting with George has been confirmed!\n\n"
        f"Date:  {meeting['preferred_date']}\n"
        f"Time:  {meeting['preferred_time']} (Romania EEST)\n"
        f"Topic: {meeting['meeting_topic']}\n"
        f"Meet:  {meet_link}\n\n"
        "Looking forward to speaking with you!\n\n"
        "Best regards,\nGeorge Rusu"
    )

    from email.mime.multipart import MIMEMultipart
    msg = MIMEMultipart()
    msg["To"]      = meeting["visitor_email"]
    msg["From"]    = f"George Rusu <{CONTACT_EMAIL}>"
    msg["Subject"] = f"Meeting Confirmed — {meeting['preferred_date']} at {meeting['preferred_time']}"
    msg.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    access_token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"raw": raw},
        )
        resp.raise_for_status()


async def _send_rejection_email(meeting: dict) -> None:
    import base64
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    if not _gmail_credentials_ok():
        logger.warning("Gmail creds not set — cannot send rejection email.")
        return

    body = (
        f"Hi {meeting['visitor_name']},\n\n"
        "Thank you for your interest in meeting with George!\n\n"
        "The proposed time is unfortunately not available.\n"
        "George will reach out to you with an alternative time shortly.\n\n"
        f"Best regards,\nGeorge Rusu\n{CONTACT_EMAIL}"
    )

    msg = MIMEMultipart()
    msg["To"]      = meeting["visitor_email"]
    msg["From"]    = f"George Rusu <{CONTACT_EMAIL}>"
    msg["Subject"] = "Re: Meeting Request"
    msg.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    access_token = await _get_access_token()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"raw": raw},
        )
        resp.raise_for_status()


# ── Webhook handlers (called from app.py /telegram/webhook) ──────────────────

async def _edit_telegram_message(chat_id: int, message_id: int, text: str) -> None:
    """Edit original Telegram message via QStash relay — removes inline buttons and updates text."""
    if not chat_id or not message_id:
        return
    try:
        await edit_message(chat_id, message_id, text)
    except Exception:
        logger.warning("Could not edit Telegram message %s", message_id)


async def handle_approve(meeting_uuid: str, chat_id: int = None, message_id: int = None) -> str:
    key = f"meeting:pending:{meeting_uuid}"
    meeting = await redis_get(key)
    if not meeting:
        return f"Meeting {meeting_uuid} not found (expired or already handled)."

    meet_link = await _create_calendar_event(meeting)
    if meet_link:
        await _send_confirmation_email(meeting, meet_link)
        status = f"✅ Approved and confirmed. Meet link: {meet_link}"
    else:
        status = "✅ Approved (no Meet link — Google Calendar not configured)."

    await redis_delete(key)

    # Edit original message: show result, remove buttons
    await _edit_telegram_message(
        chat_id, message_id,
        f"📅 <b>Meeting Request</b> — ✅ Approved\n\n"
        f"Name:  {meeting.get('visitor_name')}\n"
        f"Email: {meeting.get('visitor_email')}\n"
        f"Date:  {meeting.get('preferred_date')} at {meeting.get('preferred_time')}\n"
        f"Topic: {meeting.get('meeting_topic')}\n\n"
        f"{status}",
    )
    return status


async def handle_reject(meeting_uuid: str, chat_id: int = None, message_id: int = None) -> str:
    key = f"meeting:pending:{meeting_uuid}"
    meeting = await redis_get(key)
    if not meeting:
        return f"Meeting {meeting_uuid} not found (expired or already handled)."

    await _send_rejection_email(meeting)
    await redis_delete(key)

    # Edit original message: show result, remove buttons
    await _edit_telegram_message(
        chat_id, message_id,
        f"📅 <b>Meeting Request</b> — ❌ Rejected\n\n"
        f"Name:  {meeting.get('visitor_name')}\n"
        f"Email: {meeting.get('visitor_email')}\n"
        f"Date:  {meeting.get('preferred_date')} at {meeting.get('preferred_time')}\n"
        f"Topic: {meeting.get('meeting_topic')}\n\n"
        "Rejection email sent to visitor.",
    )
    return "❌ Meeting rejected and visitor notified."


# ── Field collection state machine ────────────────────────────────────────────

_FIELDS = ["visitor_name", "visitor_email", "preferred_date", "preferred_time", "meeting_topic"]

# Phrases that look like requests, not names — prevent misidentification
import re as _re
_NAME_BLACKLIST = _re.compile(
    r"^(vreau|want|i want|i'?d like|schedule|book|set up|please|hi|hello|eu|am|i am|"
    r"can you|could you|would you|i need|need to|looking to|interested)\b",
    _re.IGNORECASE,
)

_RO_DETECT = _re.compile(
    r'[ăâîșțĂÂÎȘȚ]|'
    r'\b(vreau|buna|salut|multumesc|mersi|da|nu|cum|unde|care|sunt|este|sau|daca|'
    r'pentru|despre|poti|vrei|aveti|puteti|merge|face|pune|lua|si|cu|un|o|al|ai|'
    r'ale|meu|tau|lui|lor|noi|voi|ei|ea|cand|cat|ce|cine|acum|azi|maine|bine)\b',
    _re.IGNORECASE
)

def _detect_lang(text: str) -> str:
    return 'ro' if _RO_DETECT.search(text) else 'en'

_FIELD_PROMPTS = {
    'en': {
        "visitor_name":   "What's your name?",
        "visitor_email":  "What's your email address? (Confirmation will be sent here.)",
        "preferred_date": "What date works for you? (e.g. 2026-06-15)",
        "preferred_time": "What time works for you? (e.g. 14:00, Romania EEST)",
        "meeting_topic":  "What would you like to discuss?",
    },
    'ro': {
        "visitor_name":   "Cum te numești?",
        "visitor_email":  "Care este adresa ta de email? (Confirmarea va fi trimisă aici.)",
        "preferred_date": "Ce dată îți convine? (ex: 2026-06-15)",
        "preferred_time": "La ce oră? (ex: 14:00, ora României)",
        "meeting_topic":  "Ce ai dori să discuți cu George?",
    },
}

# In-memory partial state (session_id → partial dict).
# In production this would also live in Redis, but for the chat session
# (which is single-process) in-memory is sufficient and avoids extra Redis round-trips.
_pending_partial:    dict[str, dict] = {}
_validation_errors:  dict[str, int]  = {}   # session_id → error count
_VALIDATION_LIMIT = 3

_CONTACT_MSG = (
    "Too many incorrect attempts. Please reach out to George directly:\n\n"
    f"• Email: {CONTACT_EMAIL}\n"
    "• LinkedIn: linkedin.com/in/rusugeorge\n"
    "• Phone: +40752383920"
)

_CONTACT_MSG_BILINGUAL = {
    'en': (
        "Too many incorrect attempts. Please reach out to George directly:\n\n"
        f"• Email: {CONTACT_EMAIL}\n"
        "• LinkedIn: linkedin.com/in/rusugeorge\n"
        "• Phone: +40752383920"
    ),
    'ro': (
        "Prea multe încercări greșite. Te rog contactează-l pe George direct:\n\n"
        f"• Email: {CONTACT_EMAIL}\n"
        "• LinkedIn: linkedin.com/in/rusugeorge\n"
        "• Telefon: +40752383920"
    ),
}

_ATTEMPTS_LEFT = {
    'en': lambda n: f"\n({n} {'attempt' if n == 1 else 'attempts'} left)",
    'ro': lambda n: f"\n({n} {'încercare rămasă' if n == 1 else 'încercări rămase'})",
}

def _record_validation_error(session_id: str, error_msg: str) -> str:
    lang = _pending_partial.get(session_id, {}).get("_lang", "en")
    count = _validation_errors.get(session_id, 0) + 1
    _validation_errors[session_id] = count
    if count >= _VALIDATION_LIMIT:
        _pending_partial.pop(session_id, None)
        _validation_errors.pop(session_id, None)
        return "__CALENDAR_LIMIT__"
    left = _VALIDATION_LIMIT - count
    return error_msg + _ATTEMPTS_LEFT[lang](left)


def _validate_date_only(date_str: str) -> Optional[str]:
    """Check date format and that it's not in the past. Returns error or None."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    BUCHAREST = ZoneInfo("Europe/Bucharest")
    today = datetime.now(BUCHAREST).date()

    try:
        meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {
            'en': "That date format doesn't look right. Please use YYYY-MM-DD (e.g. 2026-06-15).",
            'ro': "Formatul datei nu pare corect. Te rog folosește YYYY-MM-DD (ex: 2026-06-15).",
        }

    if meeting_date <= today:
        from datetime import timedelta
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        return {
            'en': f"Meetings must be booked at least one day in advance. Please choose a future date (e.g. {tomorrow}).",
            'ro': f"Meetingurile se programează cu cel puțin o zi înainte. Te rog alege o dată viitoare (ex: {tomorrow}).",
        }

    return None


def _validate_datetime_quick(date_str: str, time_str: str) -> Optional[str]:
    """Validate date/time without API calls. Returns error message or None if valid."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    BUCHAREST = ZoneInfo("Europe/Bucharest")
    now = datetime.now(BUCHAREST)

    try:
        meeting_dt = datetime.strptime(
            f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=BUCHAREST)
    except ValueError:
        # clear_date=False — bad time format, keep the date
        return {
            'en': "That time format doesn't look right. Please use HH:MM (e.g. 14:00).",
            'ro': "Formatul orei nu pare corect. Te rog folosește HH:MM (ex: 14:00).",
        }, False

    hour = int(time_str.split(":")[0])
    if hour < 8 or hour >= 18:
        # clear_date=False — only the time is wrong, keep the date
        return {
            'en': "George is available between 08:00 and 18:00 (Romania EEST). What time works for you?",
            'ro': "George este disponibil între 08:00 și 18:00 (ora României). La ce oră îți convine?",
        }, False

    if meeting_dt <= now + timedelta(hours=2):
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        # clear_date=True — slot too soon, need a new date
        return {
            'en': f"That slot is too soon (minimum 2 hours from now). What date works? (e.g. {tomorrow})",
            'ro': f"Acel slot este prea apropiat (minimum 2 ore de acum). Ce dată îți convine? (ex: {tomorrow})",
        }, True

    return None, False


async def _check_calendar_availability(date_str: str, time_str: str) -> Optional[str]:
    """Check FreeBusy for the whole day. If requested slot is busy, show free slots."""
    if not _gmail_credentials_ok():
        return None

    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        BUCHAREST   = ZoneInfo("Europe/Bucharest")
        day_start   = datetime.strptime(f"{date_str} 09:00", "%Y-%m-%d %H:%M").replace(tzinfo=BUCHAREST)
        day_end     = datetime.strptime(f"{date_str} 18:00", "%Y-%m-%d %H:%M").replace(tzinfo=BUCHAREST)
        slot_start  = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=BUCHAREST)
        slot_end    = slot_start + timedelta(hours=1)

        access_token = await _get_access_token()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://www.googleapis.com/calendar/v3/freeBusy",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "timeMin": day_start.isoformat(),
                    "timeMax": day_end.isoformat(),
                    "items":   [{"id": "primary"}],
                },
            )
            resp.raise_for_status()
            busy_periods = resp.json().get("calendars", {}).get("primary", {}).get("busy", [])

        # Parse busy ranges
        busy_ranges = []
        for bp in busy_periods:
            bs = datetime.fromisoformat(bp["start"]).astimezone(BUCHAREST)
            be = datetime.fromisoformat(bp["end"]).astimezone(BUCHAREST)
            busy_ranges.append((bs, be))

        def _overlaps(s, e):
            return any(bs < e and be > s for bs, be in busy_ranges)

        # Requested slot is free
        if not _overlaps(slot_start, slot_end):
            return None

        # Find all free 1-hour slots in 09:00–17:00
        free_slots = []
        for hour in range(9, 18):
            s = datetime.strptime(f"{date_str} {hour:02d}:00", "%Y-%m-%d %H:%M").replace(tzinfo=BUCHAREST)
            if not _overlaps(s, s + timedelta(hours=1)):
                free_slots.append(f"{hour:02d}:00")

        if free_slots:
            return {
                'en': f"George is busy at {time_str} on {date_str}.\nAvailable slots: {' · '.join(free_slots)}\nWhich time works for you?",
                'ro': f"George este ocupat la {time_str} pe {date_str}.\nSloturi disponibile: {' · '.join(free_slots)}\nCe oră îți convine?",
            }
        return {
            'en': f"George is fully booked on {date_str} (09:00–18:00). Could you suggest a different date?",
            'ro': f"George este complet ocupat pe {date_str} (09:00–18:00). Poți sugera o altă dată?",
        }

    except Exception:
        logger.exception("Calendar availability check failed — assuming free")
        return None


async def run_calendar(
    query: str,
    session_id: str = "default",
    on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    on_telegram: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> str:
    if on_progress:
        await on_progress("Checking what information I need…")

    partial = _pending_partial.get(session_id, {})

    # Detect language on first message, persist for the session
    if "_lang" not in partial:
        partial["_lang"] = _detect_lang(query)
    lang = partial["_lang"]

    # Track which fields were already set before this message
    fields_before = set(k for k in partial.keys() if not k.startswith("_"))

    # Fill in whatever the user just provided by scanning the message for known patterns
    import re
    email_m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", query)
    date_m  = re.search(r"\d{4}-\d{2}-\d{2}", query)
    time_m  = re.search(r"\b\d{1,2}:\d{2}\b", query)

    fields_extracted_this_turn = set()

    if email_m and "visitor_email" not in partial:
        partial["visitor_email"] = email_m.group(0)
        fields_extracted_this_turn.add("visitor_email")
    if date_m and "preferred_date" not in partial:
        partial["preferred_date"] = date_m.group(0)
        fields_extracted_this_turn.add("preferred_date")
    if time_m and "preferred_time" not in partial:
        partial["preferred_time"] = time_m.group(0)
        fields_extracted_this_turn.add("preferred_time")

    # Determine which field we were asking for last turn (i.e. next missing before this turn)
    asked_for = None
    for field in _FIELDS:
        if field not in fields_before:
            asked_for = field
            break

    # visitor_name: only collect when asked, short answer, no structured data, not a request phrase
    if asked_for == "visitor_name" and "visitor_name" not in partial \
            and not fields_extracted_this_turn and len(query.split()) <= 4 \
            and not _NAME_BLACKLIST.match(query.strip()):
        partial["visitor_name"] = query.strip()
        fields_extracted_this_turn.add("visitor_name")

    # meeting_topic: only when all other 4 fields were already set AND we asked for it
    four_fields = {"visitor_name", "visitor_email", "preferred_date", "preferred_time"}
    if asked_for == "meeting_topic" and four_fields.issubset(partial) \
            and "meeting_topic" not in partial:
        partial["meeting_topic"] = query.strip()

    _pending_partial[session_id] = partial

    def _resolve(err):
        """Resolve bilingual error dict to string using session lang."""
        return err[lang] if isinstance(err, dict) else err

    # Validate date alone as soon as it's extracted
    if "preferred_date" in fields_extracted_this_turn:
        date_err = _validate_date_only(partial["preferred_date"])
        if date_err:
            partial.pop("preferred_date", None)
            _pending_partial[session_id] = partial
            return _record_validation_error(session_id, _resolve(date_err))

    # Validate date+time combined as soon as both are present (no API call)
    if "preferred_date" in partial and "preferred_time" in partial:
        quick_err, clear_date = _validate_datetime_quick(partial["preferred_date"], partial["preferred_time"])
        if quick_err:
            if clear_date:
                partial.pop("preferred_date", None)
            partial.pop("preferred_time", None)
            _pending_partial[session_id] = partial
            return _record_validation_error(session_id, _resolve(quick_err))

    # Ask for next missing field — reset error counter since previous field was valid
    for field in _FIELDS:
        if field not in partial:
            _validation_errors[session_id] = 0
            return _FIELD_PROMPTS[lang][field]

    # All fields collected — check calendar availability before saving
    avail_err = await _check_calendar_availability(partial["preferred_date"], partial["preferred_time"])
    if avail_err:
        partial.pop("preferred_date", None)
        partial.pop("preferred_time", None)
        _pending_partial[session_id] = partial
        return _record_validation_error(session_id, _resolve(avail_err))

    # Save to Redis and notify George
    if on_progress:
        await on_progress(
            "All details collected — sending request to George…" if lang == 'en'
            else "Toate detaliile au fost colectate — trimit cererea lui George…"
        )

    meeting_uuid = str(uuid.uuid4())
    key = f"meeting:pending:{meeting_uuid}"
    meeting_data = {**partial, "session_id": session_id, "uuid": meeting_uuid}

    saved = await redis_set(key, meeting_data)

    buttons = [
        {"text": "✅ Approve", "callback_data": f"approve:{meeting_uuid}"},
        {"text": "❌ Reject",  "callback_data": f"reject:{meeting_uuid}"},
    ]
    tg_text = (
        "📅 <b>Meeting Request!</b>\n\n"
        f"Name:  {partial['visitor_name']}\n"
        f"Email: {partial['visitor_email']}\n"
        f"Date:  {partial['preferred_date']} at {partial['preferred_time']}\n"
        f"Topic: {partial['meeting_topic']}"
    )

    try:
        if on_telegram:
            await on_telegram("Sending meeting request to George…")
        await send_message_with_buttons(tg_text, buttons)
    except Exception:
        logger.exception("Failed to send Telegram notification for meeting request")
        _pending_partial.pop(session_id, None)
        return {
            'en': (
                "I wasn't able to forward your meeting request right now. "
                "Please reach out to George directly:\n\n"
                f"• Email: {CONTACT_EMAIL}\n"
                "• LinkedIn DM: linkedin.com/in/rusugeorge"
            ),
            'ro': (
                "Nu am putut trimite cererea de meeting acum. "
                "Te rog contactează-l pe George direct:\n\n"
                f"• Email: {CONTACT_EMAIL}\n"
                "• LinkedIn: linkedin.com/in/rusugeorge"
            ),
        }[lang]

    # Clear partial state and error counter
    _pending_partial.pop(session_id, None)
    _validation_errors.pop(session_id, None)

    if not saved:
        return {
            'en': (
                "Your details were sent to George via Telegram, but I couldn't save them to the system. "
                "To be safe, please also reach out directly:\n\n"
                f"• Email: {CONTACT_EMAIL}\n"
                "• LinkedIn DM: linkedin.com/in/rusugeorge"
            ),
            'ro': (
                "Detaliile au fost trimise lui George pe Telegram, dar nu le-am putut salva în sistem. "
                "Ca măsură de siguranță, contactează-l și direct:\n\n"
                f"• Email: {CONTACT_EMAIL}\n"
                "• LinkedIn: linkedin.com/in/rusugeorge"
            ),
        }[lang]

    return {
        'en': (
            f"Your meeting request has been sent to George for approval! "
            f"A confirmation (or alternative) will arrive at {partial['visitor_email']} shortly. "
            "Thank you!"
        ),
        'ro': (
            f"Cererea ta de meeting a fost trimisă lui George pentru aprobare! "
            f"O confirmare (sau alternativă) va ajunge la {partial['visitor_email']} în curând. "
            "Mulțumesc!"
        ),
    }[lang]
