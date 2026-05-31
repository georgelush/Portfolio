import asyncio
import json
import logging
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
async def chat(body: ChatRequest):
    if len(body.message) > _MAX_MSG_CHARS:
        return Response(
            status_code=400,
            content=f"Message too long — max {_MAX_MSG_CHARS} characters.",
        )

    queue: asyncio.Queue = asyncio.Queue()

    # Maintain conversation history per session
    sid = body.session_id
    if sid not in _sessions:
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
