import logging
import os
import re
from typing import Any, Callable, Coroutine, Optional

import httpx

from portfolio_config import ASSISTANT_NAME, CONTACT_EMAIL

logger = logging.getLogger(__name__)

_BASE_URL    = "https://api.telegram.org/bot{token}"
_QSTASH_URL  = "https://qstash.upstash.io/v2/publish"

# Session state for contact collection
_pending_contact: dict[str, dict] = {}
_validation_errors: dict[str, int] = {}

_VALIDATION_LIMIT = 3
_FIELDS_ORDER = ["name", "contact", "reason"]

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE_RE = re.compile(r"[\+\d][\d\s\-\(\)]{6,}")
_NAME_BLACKLIST = re.compile(
    r'\b(vreau|want|contact|reach|hello|hi|hey|call|meet|talk|schedule|discuss|colaborare|collaboration)\b',
    re.IGNORECASE
)
_RO_DETECT = re.compile(
    r'[ăâîșțĂÂÎȘȚ]|'
    r'\b(vreau|buna|salut|multumesc|mersi|da|nu|cum|unde|care|sunt|este|sau|daca|'
    r'pentru|despre|poti|vrei|aveti|puteti|merge|face|pune|lua|si|cu|un|o|al|ai|'
    r'ale|meu|tau|lui|lor|noi|voi|ei|ea|cand|cat|ce|cine|acum|azi|maine|bine)\b',
    re.IGNORECASE
)

# ── Bilingual strings ──────────────────────────────────────────────────────────

_QUESTIONS = {
    'en': {
        "name":    "What's your name?",
        "contact": "Type your email or phone number so George can reach you.",
        "reason":  "Briefly, what's the reason you'd like to connect? (e.g. job opportunity, collaboration, project idea…)",
    },
    'ro': {
        "name":    "Cum te numești?",
        "contact": "Scrie adresa de email sau numărul de telefon.",
        "reason":  "Pe scurt, de ce vrei să îl contactezi pe George? (ex: oportunitate de job, colaborare, idee de proiect…)",
    },
}

_VALIDATION_MSGS = {
    'en': {
        "name_invalid":     "Please enter just your name (first and last name).",
        "contact_invalid":  "That doesn't look like a valid email or phone number. Please try again.",
        "reason_short":     "Could you add a bit more detail? What are you looking for or offering?",
        "limit":            (
            "Too many invalid attempts. You can reach George directly:\n"
            f"✉ {CONTACT_EMAIL}\n"
            "💼 linkedin.com/in/rusugeorge\n"
            "📞 +40752383920"
        ),
    },
    'ro': {
        "name_invalid":     "Te rog introdu doar numele tău (prenume și nume).",
        "contact_invalid":  "Adresa de email sau numărul de telefon nu pare valid. Te rog încearcă din nou.",
        "reason_short":     "Poți adăuga ceva mai mult detaliu? Ce cauți sau oferi?",
        "limit":            (
            "Prea multe încercări invalide. Îl poți contacta pe George direct:\n"
            f"✉ {CONTACT_EMAIL}\n"
            "💼 linkedin.com/in/rusugeorge\n"
            "📞 +40752383920"
        ),
    },
}

_SUCCESS_MSG = {
    'en': lambda contact: (
        f"Done! George has been notified and will reach out to you at {contact} shortly.\n\n"
        "You can also contact him directly:\n"
        f"✉ {CONTACT_EMAIL}\n"
        "💼 linkedin.com/in/rusugeorge\n"
        "📞 +40752383920"
    ),
    'ro': lambda contact: (
        f"Gata! George a fost notificat și te va contacta la {contact} în curând.\n\n"
        "Îl poți contacta și direct:\n"
        f"✉ {CONTACT_EMAIL}\n"
        "💼 linkedin.com/in/rusugeorge\n"
        "📞 +40752383920"
    ),
}

_FAIL_MSG = {
    'en': f"Couldn't send the notification right now. You can reach George directly at {CONTACT_EMAIL} or +40752383920.",
    'ro': f"Nu s-a putut trimite notificarea. Îl poți contacta pe George direct la {CONTACT_EMAIL} sau +40752383920.",
}

# Used by orchestrator when __TELEGRAM_LIMIT__ is returned
_CONTACT_MSG = _VALIDATION_MSGS['en']['limit']


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_lang(text: str) -> str:
    return 'ro' if _RO_DETECT.search(text) else 'en'


def _token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


def _qstash_token() -> str:
    return os.getenv("QSTASH_TOKEN", "")


def _credentials_ok() -> bool:
    ok = bool(_token() and _chat_id())
    if not ok:
        logger.warning("Telegram credentials not set — skipping notification.")
    return ok


async def _post_via_qstash(tg_endpoint: str, payload: dict) -> bool:
    """Relay a Telegram API call through Upstash QStash to bypass network blocks."""
    dest = f"https://api.telegram.org/bot{_token()}/{tg_endpoint}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{_QSTASH_URL}/{dest}",
            headers={"Authorization": f"Bearer {_qstash_token()}"},
            json=payload,
        )
        resp.raise_for_status()
    return True


async def _post_direct(tg_endpoint: str, payload: dict) -> bool:
    """Call Telegram API directly."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"https://api.telegram.org/bot{_token()}/{tg_endpoint}",
            json=payload,
        )
        resp.raise_for_status()
    return True


async def _tg_post(tg_endpoint: str, payload: dict) -> bool:
    """Send via QStash if token available, otherwise direct."""
    if _qstash_token():
        return await _post_via_qstash(tg_endpoint, payload)
    return await _post_direct(tg_endpoint, payload)


async def send_message(text: str) -> bool:
    if not _credentials_ok():
        return False
    payload = {"chat_id": _chat_id(), "text": text, "parse_mode": "HTML"}
    await _tg_post("sendMessage", payload)
    return True


async def edit_message(chat_id: int, message_id: int, text: str) -> bool:
    if not _credentials_ok():
        return False
    payload = {
        "chat_id":    chat_id,
        "message_id": message_id,
        "text":       text,
        "parse_mode": "HTML",
    }
    await _tg_post("editMessageText", payload)
    return True


async def send_message_with_buttons(text: str, buttons: list[dict]) -> bool:
    if not _credentials_ok():
        return False
    inline_keyboard = [[{"text": b["text"], "callback_data": b["callback_data"]}] for b in buttons]
    payload = {
        "chat_id":      _chat_id(),
        "text":         text,
        "parse_mode":   "HTML",
        "reply_markup": {"inline_keyboard": inline_keyboard},
    }
    await _tg_post("sendMessage", payload)
    return True


def _validate_contact(value: str) -> bool:
    return bool(_EMAIL_RE.search(value) or _PHONE_RE.search(value))


def _next_missing(fields: dict) -> Optional[str]:
    for f in _FIELDS_ORDER:
        if not fields.get(f):
            return f
    return None


def _bump_error(session_id: str, lang: str, key: str) -> str:
    _validation_errors[session_id] = _validation_errors.get(session_id, 0) + 1
    if _validation_errors[session_id] >= _VALIDATION_LIMIT:
        _pending_contact.pop(session_id, None)
        _validation_errors.pop(session_id, None)
        return "__TELEGRAM_LIMIT__"
    return _VALIDATION_MSGS[lang][key]


async def run_telegram(
    query: str,
    session_id: str = "default",
    on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> str:
    fields = _pending_contact.setdefault(session_id, {})

    # Detect language on first message, persist for the session
    if "_lang" not in fields:
        fields["_lang"] = _detect_lang(query)
    lang = fields["_lang"]

    missing = _next_missing(fields)
    if missing:
        value = query.strip()

        if missing == "name":
            words = value.split()
            if len(words) > 4 or _NAME_BLACKLIST.search(value):
                return _bump_error(session_id, lang, "name_invalid")

        if missing == "contact" and not _validate_contact(value):
            return _bump_error(session_id, lang, "contact_invalid")

        if missing == "reason" and len(value) < 10:
            return _bump_error(session_id, lang, "reason_short")

        fields[missing] = value
        _validation_errors[session_id] = 0

    missing = _next_missing(fields)
    if missing:
        return _QUESTIONS[lang][missing]

    # All fields collected — send notification
    name    = fields.get("name", "—")
    contact = fields.get("contact", "—")
    reason  = fields.get("reason", "—")

    tg_text = (
        f"👤 <b>Contact Request — {ASSISTANT_NAME}</b>\n\n"
        f"Name: {name}\n"
        f"Contact: {contact}\n"
        f"Message: <i>{reason}</i>\n\n"
        f"📍 Via: {ASSISTANT_NAME}"
    )

    if on_progress:
        await on_progress("Sending your details to George…" if lang == 'en' else "Trimit datele tale lui George…")

    try:
        success = await send_message(tg_text)
    except Exception:
        logger.warning("Telegram send failed — contact info collected but notification not delivered.")
        success = False

    _pending_contact.pop(session_id, None)
    _validation_errors.pop(session_id, None)

    if success:
        return _SUCCESS_MSG[lang](contact)
    return _FAIL_MSG[lang]
