"""Job Match Analyzer — Phase 5.

Recruiter-focused: paste a JD URL or raw text → instant fit report for George Rusu.
Auto-detects URL vs raw text. Uses Groq for analysis, compares against portfolio_data.txt.
Sends Telegram notification to George after every analysis.
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from agents.telegram_agent import send_message

logger = logging.getLogger(__name__)

_PORTFOLIO_DATA = Path(__file__).parent.parent / "knowledge" / "portfolio_data.txt"
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_LINKEDIN_JOB_RE = re.compile(r"linkedin\.com/jobs/view/(\d+)", re.IGNORECASE)

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_url(text: str) -> bool:
    return bool(_URL_RE.search(text.strip()))


async def _fetch_linkedin(job_id: str) -> str:
    """Fetch LinkedIn job via the unauthenticated guest API endpoint."""
    guest_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers=_SCRAPE_HEADERS) as client:
        resp = await client.get(guest_url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Job description lives in show-more-less-html__markup or the criteria list
    for selector in [
        ".show-more-less-html__markup",
        ".description__text",
        "[class*='description']",
        "main",
    ]:
        block = soup.select_one(selector)
        if block:
            text = block.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return text[:8000]

    return soup.get_text(separator="\n", strip=True)[:8000]


async def _fetch_jd_from_url(url: str) -> str:
    """Fetch and extract readable text from a job posting URL."""
    # LinkedIn special path — use the unauthenticated guest API
    li_match = _LINKEDIN_JOB_RE.search(url)
    if li_match:
        return await _fetch_linkedin(li_match.group(1))

    async with httpx.AsyncClient(timeout=15, follow_redirects=True, headers=_SCRAPE_HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    for selector in ["[class*='job']", "[class*='description']", "main", "article", "#content"]:
        block = soup.select_one(selector)
        if block:
            text = block.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text[:8000]

    return soup.get_text(separator="\n", strip=True)[:8000]


def _load_portfolio() -> str:
    if _PORTFOLIO_DATA.exists():
        return _PORTFOLIO_DATA.read_text(encoding="utf-8")[:25000]
    return ""


def _get_groq_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


_ANALYSIS_PROMPT = """You are analyzing a job description to evaluate how well George Rusu fits this role.
You are speaking directly to the recruiter who posted this job.

## George's Background (portfolio_data.txt excerpt):
{portfolio}

## Job Description:
{jd}

## Your task:
1. Extract the job title and company name.
2. Calculate a fit score (0–100%). Be realistic — a score above 90% is rare and requires near-perfect alignment on every requirement including bonus ones.
3. Identify the top 3 SPECIFIC strengths where George directly matches the requirements.
   Each strength must map to a specific requirement mentioned in the JD.
4. Identify 1 honest gap or area where George's profile is weaker or less proven relative to this specific role.
   It could be a missing technology, limited scale of experience, or a bonus requirement not covered.
   IMPORTANT: Do NOT flag something as a gap if George has a shipped project that directly demonstrates that skill — check the portfolio carefully before claiming a gap.
   Do NOT write "No significant gap identified." — always provide a real, specific gap that is not already covered by his projects.
   Then add context on how George addresses or can address it.

## Output format (follow EXACTLY — no extra text before or after):
JOB_TITLE: <job title>
COMPANY: <company name>
FIT_SCORE: <number 0-100>
STRENGTH_1: <strength> — maps directly to your requirement for <specific JD requirement>
STRENGTH_2: <strength> — maps directly to your requirement for <specific JD requirement>
STRENGTH_3: <strength> — maps directly to your requirement for <specific JD requirement>
GAP: <specific gap> — context: <how George addresses or can address this>"""


def _parse_analysis(raw: str, jd_snippet: str, url: Optional[str]) -> dict:
    """Parse structured LLM output into a result dict."""
    result = {
        "job_title": "this role",
        "company":   "the company",
        "fit_score": 0,
        "strengths": [],
        "gap":       None,
        "jd_snippet": jd_snippet[:100],
        "url":       url,
    }

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("JOB_TITLE:"):
            result["job_title"] = line.split(":", 1)[1].strip()
        elif line.startswith("COMPANY:"):
            result["company"] = line.split(":", 1)[1].strip()
        elif line.startswith("FIT_SCORE:"):
            try:
                result["fit_score"] = int(re.search(r"\d+", line).group())
            except Exception:
                pass
        elif line.startswith("STRENGTH_"):
            result["strengths"].append(line.split(":", 1)[1].strip())
        elif line.startswith("GAP:"):
            result["gap"] = line.split(":", 1)[1].strip()

    return result


def _format_output(parsed: dict) -> str:
    strengths = "\n".join(f"- {s}" for s in parsed["strengths"][:3])
    gap_line  = parsed.get("gap") or "No significant gap identified."

    return (
        f"📊 Fit Score: {parsed['fit_score']}%\n\n"
        f"Based on your job description, here is George's fit for {parsed['job_title']}"
        f" at {parsed['company']}:\n\n"
        f"✅ Where George delivers:\n{strengths}\n\n"
        f"⚠️ One gap to consider:\n- {gap_line}\n\n"
        "Interested? You can request George's CV or schedule a call with him directly from this chat."
    )


async def run_job_match(
    query: str,
    on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    on_telegram: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
) -> str:
    url: Optional[str] = None
    jd_text: str

    if _is_url(query):
        url_match = _URL_RE.search(query)
        url = url_match.group(0)
        if on_progress:
            await on_progress(f"Fetching job description from {url}…")
        try:
            jd_text = await _fetch_jd_from_url(url)
        except Exception as exc:
            logger.warning("URL fetch failed: %s", exc)
            if "linkedin.com" in url.lower():
                return (
                    "LinkedIn requires a login to access job pages and blocks automated requests.\n\n"
                    "Please copy the job description text from the LinkedIn page and paste it here — "
                    "I'll analyse it instantly."
                )
            return (
                "I couldn't load that URL (the site may block automated access).\n\n"
                "Please copy the job description text and paste it directly here — "
                "I'll analyse it instantly."
            )
    else:
        jd_text = query
        if on_progress:
            await on_progress("Analysing the job description you provided…")

    if not jd_text or len(jd_text) < 50:
        return "The job description seems too short. Please paste more of the JD text."

    portfolio = _load_portfolio()
    if not portfolio:
        return "Portfolio data not found — cannot run analysis."

    if on_progress:
        await on_progress("Comparing George's profile against the role requirements…")

    llm = _get_groq_llm()
    prompt = _ANALYSIS_PROMPT.format(
        portfolio=portfolio[:20000],
        jd=jd_text[:4000],
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_output = resp.content.strip()
    except Exception as exc:
        logger.exception("Groq analysis failed")
        return f"Analysis failed: {exc}"

    parsed  = _parse_analysis(raw_output, jd_snippet=jd_text[:100], url=url)
    output  = _format_output(parsed)

    # Notify George via Telegram
    if on_telegram:
        await on_telegram("Notifying George…")
    tg_text = (
        f"💼 <b>Recruiter analyzed role:</b> {parsed['job_title']} at {parsed['company']}\n"
        f"Fit Score: {parsed['fit_score']}%\n"
        f"JD snippet: <i>{parsed['jd_snippet']}</i>"
    )
    if url:
        tg_text += f"\nURL: {url}"
    await send_message(tg_text)

    if on_progress:
        await on_progress("Analysis complete — George notified.")

    return output
