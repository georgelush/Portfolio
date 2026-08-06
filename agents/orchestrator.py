import asyncio
import contextvars
import logging
import os
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agents.calendar_agent import run_calendar
from agents.email_agent import run_email
from agents.job_match_agent import run_job_match
from agents.rag_agent import run_rag
from agents.telegram_agent import run_telegram

logger = logging.getLogger(__name__)

# Context-var holds the active SSE queue — safe under concurrent asyncio tasks
_queue_var: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "sse_queue", default=None
)

# Context-var holds already-completed actions for this session (e.g. {"email", "calendar"})
_completed_var: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "completed_actions", default=frozenset()
)

_BLOCKED_MESSAGES = {
    "email": (
        "George's CV was already sent this session — please check your inbox (and spam folder).\n\n"
        "If you didn't receive it, you can reach George directly:\n"
        "📄 Download CV: https://georgelush.github.io/Portfolio/images/RusuGeorgeCV.pdf\n"
        "✉ Email: georgel1988@gmail.com\n"
        "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
        "📞 Phone: +40752383920"
    ),
    "calendar": (
        "A meeting request was already submitted this session — George will confirm via email shortly.\n\n"
        "If you haven't heard back, reach out directly:\n"
        "✉ Email: georgel1988@gmail.com\n"
        "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
        "📞 Phone: +40752383920"
    ),
}

_ROUTING_MESSAGES = {
    "rag":          "Looking this up in George's background & projects…",
    "email":        "Getting the CV ready to send to you…",
    "calendar":     "Setting up a meeting with George…",
    "job_match": "Analysing this role against George's profile…",
    "telegram":     "Sending George a notification…",
    "off_topic":    "This doesn't seem related to George…",
}

_ROUTING_PROMPT = """You are the routing layer for George AI Assistant, representing George Rusu.
Classify the visitor's intent and respond with EXACTLY ONE agent name from this list:

  rag           — questions about George: skills, projects, experience, background, certifications
  email         — visitor wants to receive the CV or contact George by email
  calendar      — visitor wants to SCHEDULE a specific meeting, call, or interview at a date and time
  job_match     — visitor shares a job posting or role description (in ANY language/format), or asks for a fit analysis or cover letter
  telegram      — visitor wants George to contact THEM back (leaving contact details, no specific date/time)
  off_topic     — anything unrelated to George or the above intents

ROUTING RULES:
- "schedule a call", "book a meeting", "set up an interview", "meet on Thursday" → calendar
- "contact me", "reach out to me", "I'm interested, get in touch" → telegram
- When ambiguous between calendar and telegram, prefer calendar if a date/time is mentioned, telegram if not.

JOB MATCH DETECTION — route to job_match when the message:
- Contains a job posting URL (LinkedIn, Indeed, company careers page, etc.)
- Pastes a job description, even partially — look for role duties, requirements, qualifications, benefits, "we are looking for", "ideal candidate", "apply now", or similar hiring language
- Starts with phrases like "Despre job", "Job description", "Position:", "Role:", "We're hiring"
- Is a long message (>200 chars) that reads like a job ad, even if mixed with Romanian text
- Asks "does George fit this role?" or "write a cover letter"

Reply with ONLY the agent name in lowercase. No explanation. Default to "rag" if unclear but related to George.

IMPORTANT: Use the conversation history to understand context. If the previous assistant message asked for name/email/date/time/topic for a meeting, route the current reply to "calendar". If the assistant was collecting contact info (name/email/phone/reason), route to "telegram". If the assistant was helping with CV/email, route to "email".

{history_block}
Current visitor message: {message}"""


# ── State ─────────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    user_message: str
    session_id: str
    history: list
    route: str
    agent_result: str
    final_response: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _emit(event_type: str, text: str, agent: str = "orchestrator") -> None:
    q = _queue_var.get()
    if q is not None:
        await q.put({"type": event_type, "agent": agent, "text": text})


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def _routing_node(state: OrchestratorState) -> dict:
    await _emit("orchestrator_thinking", "Reading your message…")

    session_id = state.get("session_id", "default")

    # Hard override: active calendar collection — keep routing to calendar until done
    from agents.calendar_agent import _pending_partial
    from agents.telegram_agent import _pending_contact
    if _pending_partial.get(session_id):
        await _emit("orchestrator_thinking", _ROUTING_MESSAGES["calendar"])
        return {"route": "calendar"}

    # Hard override: active telegram contact collection
    if _pending_contact.get(session_id):
        await _emit("orchestrator_thinking", _ROUTING_MESSAGES["telegram"])
        return {"route": "telegram"}

    # Build history block for routing context (last 6 messages max)
    history = state.get("history", [])
    recent = history[-6:] if len(history) > 6 else history
    if recent:
        lines = []
        for h in recent:
            role = "Visitor" if h["role"] == "user" else "Assistant"
            lines.append(f"{role}: {h['content'][:200]}")
        history_block = "Recent conversation:\n" + "\n".join(lines) + "\n"
    else:
        history_block = ""

    llm = _get_llm()
    resp = await llm.ainvoke([
        HumanMessage(content=_ROUTING_PROMPT.format(
            message=state["user_message"],
            history_block=history_block,
        ))
    ])
    route = resp.content.strip().lower().strip('"').strip("'")

    if route not in _ROUTING_MESSAGES:
        route = "rag"

    # Block already-completed actions for this session
    completed = _completed_var.get()
    if route in completed:
        lang = _detect_lang(state["user_message"])
        _BLOCKED_BILINGUAL = {
            "email": {
                'en': (
                    "George's CV was already sent this session — please check your inbox (and spam folder).\n\n"
                    "If you didn't receive it, you can reach George directly:\n"
                    "📄 Download CV: https://georgelush.github.io/Portfolio/images/RusuGeorgeCV.pdf\n"
                    "✉ Email: georgel1988@gmail.com\n"
                    "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
                    "📞 Phone: +40752383920"
                ),
                'ro': (
                    "CV-ul lui George a fost deja trimis în această sesiune — verifică inbox-ul (și spam).\n\n"
                    "Dacă nu l-ai primit, îl poți contacta direct:\n"
                    "📄 Descarcă CV: https://georgelush.github.io/Portfolio/images/RusuGeorgeCV.pdf\n"
                    "✉ Email: georgel1988@gmail.com\n"
                    "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
                    "📞 Telefon: +40752383920"
                ),
            },
            "calendar": {
                'en': (
                    "A meeting request was already submitted this session — George will confirm via email shortly.\n\n"
                    "If you haven't heard back, reach out directly:\n"
                    "✉ Email: georgel1988@gmail.com\n"
                    "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
                    "📞 Phone: +40752383920"
                ),
                'ro': (
                    "O cerere de meeting a fost deja trimisă în această sesiune — George va confirma pe email în curând.\n\n"
                    "Dacă nu ai primit răspuns, contactează-l direct:\n"
                    "✉ Email: georgel1988@gmail.com\n"
                    "💼 LinkedIn: linkedin.com/in/rusugeorge\n"
                    "📞 Telefon: +40752383920"
                ),
            },
        }
        blocked = _BLOCKED_BILINGUAL.get(route, {})
        msg = blocked.get(lang, blocked.get('en', "This action was already completed this session."))
        await _emit("orchestrator_thinking", msg)
        return {"route": "synthesize", "agent_result": msg}

    await _emit("orchestrator_thinking", _ROUTING_MESSAGES[route])
    return {"route": route}


async def _rag_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Searching George's experience & projects…", agent="rag")

    async def _prog(text: str) -> None:
        await _emit("agent_working", text, agent="rag")

    result = await run_rag(state["user_message"], on_progress=_prog)
    await _emit("agent_done", "Got what I needed.", agent="rag")
    return {"agent_result": result}


async def _email_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Getting the CV ready…", agent="email")

    async def _prog(text: str) -> None:
        await _emit("agent_working", text, agent="email")

    _tg_started = False

    async def _on_telegram(text: str) -> None:
        nonlocal _tg_started
        if not _tg_started:
            await _emit("agent_start", "Notifying George…", agent="telegram")
            _tg_started = True

    result = await run_email(state["user_message"], on_progress=_prog, on_telegram=_on_telegram)
    if _tg_started:
        await _emit("agent_done", "George notified.", agent="telegram")
    await _emit("agent_done", "Done.", agent="email")
    return {"agent_result": result}


async def _calendar_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Checking George's availability…", agent="calendar")

    async def _prog(text: str) -> None:
        await _emit("agent_working", text, agent="calendar")

    _tg_started = False

    async def _on_telegram(text: str) -> None:
        nonlocal _tg_started
        if not _tg_started:
            await _emit("agent_start", "Notifying George…", agent="telegram")
            _tg_started = True

    session_id = state.get("session_id", "default")
    result = await run_calendar(
        state["user_message"],
        session_id=session_id,
        on_progress=_prog,
        on_telegram=_on_telegram,
    )
    await _emit("agent_done", "Done.", agent="calendar")
    if _tg_started:
        await _emit("agent_done", "George notified.", agent="telegram")

    # Validation limit hit — signal frontend to exhaust budget
    from agents.calendar_agent import _pending_partial as _cal_partial, _CONTACT_MSG
    if result == "__CALENDAR_LIMIT__":
        await _emit("calendar_limit", "Validation limit reached.", agent="calendar")
        result = _CONTACT_MSG
        return {"agent_result": result}

    # Signal frontend to keep calendar node glowing if still collecting fields
    if _cal_partial.get(session_id):
        await _emit("calendar_collecting", "Waiting for your response…", agent="calendar")

    return {"agent_result": result}


async def _job_match_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Matching this role to George's profile…", agent="job_match")

    async def _prog(text: str) -> None:
        await _emit("agent_working", text, agent="job_match")

    _tg_started = False

    async def _on_telegram(text: str) -> None:
        nonlocal _tg_started
        if not _tg_started:
            await _emit("agent_start", "Notifying George…", agent="telegram")
            _tg_started = True

    result = await run_job_match(state["user_message"], on_progress=_prog, on_telegram=_on_telegram)
    await _emit("agent_done", "Analysis complete.", agent="job_match")
    if _tg_started:
        await _emit("agent_done", "George notified.", agent="telegram")
    return {"agent_result": result}


async def _telegram_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Collecting your contact details…", agent="telegram")

    async def _prog(text: str) -> None:
        await _emit("agent_working", text, agent="telegram")

    session_id = state.get("session_id", "default")
    result = await run_telegram(state["user_message"], session_id=session_id, on_progress=_prog)

    from agents.telegram_agent import _pending_contact, _CONTACT_MSG as _TG_CONTACT_MSG
    if result == "__TELEGRAM_LIMIT__":
        await _emit("telegram_limit", "Validation limit reached.", agent="telegram")
        return {"agent_result": _TG_CONTACT_MSG}

    if _pending_contact.get(session_id):
        await _emit("telegram_collecting", "Waiting for your response…", agent="telegram")
    else:
        await _emit("agent_done", "George has been notified.", agent="telegram")

    return {"agent_result": result}


def _detect_lang(text: str) -> str:
    import re as _re2
    _ro = _re2.compile(
        r'[ăâîșțĂÂÎȘȚ]|'
        r'\b(vreau|buna|salut|multumesc|mersi|da|nu|cum|unde|care|sunt|este|sau|daca|'
        r'pentru|despre|poti|vrei|aveti|puteti|merge|face|pune|lua|si|cu|un|o|al|ai|'
        r'ale|meu|tau|lui|lor|noi|voi|ei|ea|cand|cat|ce|cine|acum|azi|maine|bine)\b',
        _re2.IGNORECASE
    )
    return 'ro' if _ro.search(text) else 'en'


async def _off_topic_node(state: OrchestratorState) -> dict:
    await _emit("agent_start", "Out of scope.", agent="off_topic")
    lang = _detect_lang(state["user_message"])
    result = {
        'en': (
            "I'm George AI Assistant — I can only help with:\n"
            "• Questions about George's skills, projects, and experience\n"
            "• Sending his CV to your inbox\n"
            "• Booking a call or meeting\n"
            "• Job-match analysis (paste a JD or URL)\n\n"
            "Try asking: \"What has George built?\" or \"Schedule a call.\""
        ),
        'ro': (
            "Sunt George AI Assistant — pot ajuta cu:\n"
            "• Întrebări despre skills-urile, proiectele și experiența lui George\n"
            "• Trimiterea CV-ului în inbox-ul tău\n"
            "• Programarea unui apel sau meeting\n"
            "• Analiza compatibilității cu un job (lipești JD sau URL)\n\n"
            "Încearcă: \"Ce a construit George?\" sau \"Vreau un meeting.\""
        ),
    }[lang]
    await _emit("agent_done", "Redirected.", agent="off_topic")
    return {"agent_result": result}


async def _synthesize_node(state: OrchestratorState) -> dict:
    await _emit("orchestrator_synthesizing", "Putting it all together…")
    return {"final_response": state["agent_result"]}


# ── Compile graph once at import time ─────────────────────────────────────────

def _build_graph():
    builder = StateGraph(OrchestratorState)

    builder.add_node("route", _routing_node)
    builder.add_node("rag", _rag_node)
    builder.add_node("email", _email_node)
    builder.add_node("calendar", _calendar_node)
    builder.add_node("job_match", _job_match_node)
    builder.add_node("telegram", _telegram_node)
    builder.add_node("off_topic", _off_topic_node)
    builder.add_node("synthesize", _synthesize_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "rag":          "rag",
            "email":        "email",
            "calendar":     "calendar",
            "job_match": "job_match",
            "telegram":     "telegram",
            "off_topic":    "off_topic",
            "synthesize":   "synthesize",   # direct short-circuit for blocked actions
        },
    )
    for node in ("rag", "email", "calendar", "job_match", "telegram", "off_topic"):
        builder.add_edge(node, "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


_graph = _build_graph()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_orchestration(
    message: str,
    queue: asyncio.Queue,
    session_id: str = "default",
    history: list = None,
    completed_actions: set = None,
) -> None:
    token_q = _queue_var.set(queue)
    token_c = _completed_var.set(frozenset(completed_actions or set()))
    try:
        result = await _graph.ainvoke({
            "user_message": message,
            "session_id":   session_id,
            "history":      history or [],
            "route":        "",
            "agent_result": "",
            "final_response": "",
        })
        await queue.put({
            "type": "final_response",
            "agent": "orchestrator",
            "text": result["final_response"],
        })
        # Emit internal completion signals so app.py can track them
        route       = result.get("route", "")
        agent_result = result.get("agent_result", "")
        if route == "email" and "has been sent to" in agent_result:
            await queue.put({"type": "_action_complete", "agent": "email", "text": ""})
        elif route == "calendar" and (
            "sent to George for approval" in agent_result
            or "trimisă lui George pentru aprobare" in agent_result
        ):
            await queue.put({"type": "_action_complete", "agent": "calendar", "text": ""})
    except Exception as exc:
        logger.exception("Orchestration error")
        await queue.put({
            "type": "error",
            "agent": "orchestrator",
            "text": f"Something went wrong: {exc}",
        })
    finally:
        _queue_var.reset(token_q)
        _completed_var.reset(token_c)
