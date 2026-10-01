import asyncio
import hmac
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agents.calendar_agent import handle_approve, handle_reject
from agents.orchestrator import run_orchestration
from agents.rag_agent import init_rag

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_rag_ready = False
_sessions:   dict = {}   # session_id → list of {role, content}
_completed:  dict = {}   # session_id → set of completed actions ("email", "calendar")
_MAX_HISTORY = 10
_MAX_SESSIONS = 500      # oldest sessions are dropped beyond this, so memory stays bounded

# Per-visitor rate limits for /chat. The endpoint is public and every message costs an LLM call,
# so one client must not be able to drain the quota or drive the email agent in a loop.
_RATE_WINDOWS = ((300, 12), (86400, 60))   # (seconds, max messages): 12 per 5 minutes, 60 per day
_hits: dict = defaultdict(deque)           # client ip → timestamps of recent messages
# Backstop for the whole service. The per-visitor limit keys on an address taken from a header,
# which a determined client can vary; this cap holds no matter how many addresses are used.
_GLOBAL_PER_DAY = 600
_all_hits: deque = deque()                 # timestamps of every accepted message in the last 24 hours


def _client_ip(request: Request) -> str:
    """Visitor address. Hugging Face Spaces sits behind a proxy, so the first forwarded hop is the client."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.monotonic()
    while _all_hits and now - _all_hits[0] > 86400:
        _all_hits.popleft()
    if len(_all_hits) >= _GLOBAL_PER_DAY:
        return True
    hits = _hits[ip]
    longest = max(window for window, _ in _RATE_WINDOWS)
    while hits and now - hits[0] > longest:
        hits.popleft()
    for window, limit in _RATE_WINDOWS:
        if sum(1 for t in hits if now - t <= window) >= limit:
            return True
    hits.append(now)
    _all_hits.append(now)
    if len(_hits) > 5000:   # forget idle visitors
        for key in [k for k, v in _hits.items() if not v or now - v[-1] > longest]:
            del _hits[key]
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_ready
    logger.info("Initialising RAG knowledge base…")
    init_rag()
    _rag_ready = True
    logger.info("RAG ready — server is live.")
    yield


app = FastAPI(title="Agent Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit preflight handler — bypasses middleware for reliability
@app.options("/chat")
async def chat_preflight():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if _rag_ready:
        return Response(status_code=200)
    return Response(status_code=503)


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Receives Telegram callback_query (Approve / Reject buttons)."""
    # When TELEGRAM_WEBHOOK_SECRET is set, only requests carrying the same secret are accepted.
    # Telegram sends it in this header when the webhook is registered with `secret_token`.
    expected = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    if expected:
        received = request.headers.get("x-telegram-bot-api-secret-token", "")
        if not hmac.compare_digest(received, expected):
            return Response(status_code=403)

    body = await request.json()
    callback = body.get("callback_query")
    if not callback:
        return Response(status_code=200)

    callback_id   = callback.get("id")
    callback_data = callback.get("data", "")

    import httpx as _httpx
    import os as _os
    token = _os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Acknowledge the button press immediately (Telegram requires this within 10s)
    if token and callback_id:
        try:
            async with _httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                    json={"callback_query_id": callback_id, "text": "Processing…"},
                )
        except Exception:
            pass

    msg        = callback.get("message", {})
    chat_id    = msg.get("chat", {}).get("id")
    message_id = msg.get("message_id")

    if callback_data.startswith("approve:"):
        meeting_uuid = callback_data.split(":", 1)[1]
        await handle_approve(meeting_uuid, chat_id=chat_id, message_id=message_id)
    elif callback_data.startswith("reject:"):
        meeting_uuid = callback_data.split(":", 1)[1]
        await handle_reject(meeting_uuid, chat_id=chat_id, message_id=message_id)

    return Response(status_code=200)


_MAX_MSG_CHARS = 7000


@app.post("/chat")
async def chat(body: ChatRequest, request: Request):
    if len(body.message) > _MAX_MSG_CHARS:
        return Response(
            status_code=400,
            content=f"Message too long — max {_MAX_MSG_CHARS} characters.",
        )
    if _rate_limited(_client_ip(request)):
        return Response(
            status_code=429,
            content="Too many messages. Please wait a few minutes, or write to george@flowentic.com.",
            headers={"Retry-After": "300"},
        )

    queue: asyncio.Queue = asyncio.Queue()

    # Maintain conversation history per session
    sid = body.session_id[:64]
    if sid not in _sessions:
        while len(_sessions) >= _MAX_SESSIONS:
            oldest = next(iter(_sessions))
            _sessions.pop(oldest, None)
            _completed.pop(oldest, None)
        _sessions[sid] = []
    history = _sessions[sid]

    async def _run():
        try:
            await run_orchestration(
                body.message, queue,
                session_id=sid,
                history=history,
                completed_actions=_completed.get(sid, set()),
            )
        except Exception as exc:
            await queue.put({"type": "error", "agent": "orchestrator", "text": str(exc)})
        finally:
            await queue.put(None)

    asyncio.create_task(_run())

    history.append({"role": "user", "content": body.message})

    async def _stream():
        final_text = None
        while True:
            event = await queue.get()
            if event is None:
                break
            if event.get("type") == "_action_complete":
                # Internal signal — track completion, don't forward to client
                _completed.setdefault(sid, set()).add(event["agent"])
                continue
            if event.get("type") == "final_response":
                final_text = event.get("text", "")
            yield {"data": json.dumps(event)}
        # Save assistant response to history
        if final_text:
            history.append({"role": "assistant", "content": final_text})
        # Keep only last _MAX_HISTORY exchanges
        if len(history) > _MAX_HISTORY * 2:
            _sessions[sid] = history[-(  _MAX_HISTORY * 2):]

    return EventSourceResponse(_stream())
