"""
SankhyaGuru Backend
====================
A small Flask API that powers the "Get Started" dashboard and the floating
chat widget in front.html.

How it works (the legitimate, professional way):
  1. Frontend sends the student's question to POST /api/chat
  2. This server calls the OFFICIAL Google Gemini API using YOUR OWN
     (free-tier) API key
  3. The model is instructed (via a system prompt) to answer in SankhyaGuru's
     voice, scoped to official-statistics / NSS / MoSPI topics
  4. The reply is returned to the frontend as SankhyaGuru's answer

What this does NOT do:
  - It does not scrape, automate, or log into any other AI's website/app
  - It does not impersonate another product's account
  - It will not lie to a student who directly asks what model/AI is being used

Setup:
  1. pip install -r requirements.txt
  2. Get a free API key at https://console.groq.com/keys (no card needed)
     and set it as the environment variable GROQ_API_KEY
  3. python app.py
  4. Point your frontend's fetch calls at http://localhost:5000/api/chat etc.

Deployment note:
  This is a long-running WSGI app (Flask + requests). Vercel's Python
  support is serverless-function based, so a plain `python app.py` script
  will NOT "just run" if you drag-and-drop it into a Vercel project. On
  Render, use Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
"""

import os
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"  # free-tier, fast Groq model
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sankhyaguru")

app = Flask(__name__)
CORS(app)  # allow the frontend (served from another origin/file) to call this API

# --------------------------------------------------------------------------
# SankhyaGuru persona
# --------------------------------------------------------------------------
# This system prompt is what makes the answers sound like "SankhyaGuru"
# rather than a generic assistant. It does NOT instruct the model to lie
# about its identity if asked directly -- it just tells it not to bring
# the topic up unprompted, which is normal product branding.

SYSTEM_PROMPT = """You are SankhyaGuru, an AI tutor built for learners preparing in
official statistics, survey methodology, and data literacy in the context of
India's statistical system (NSS, MoSPI, CPI, GIS/SQL for official data, data
protection and ethics in statistics).

Voice and behavior:
- Answer clearly, step by step, the way a patient tutor would.
- Prefer concrete examples relevant to Indian official statistics
  (e.g. NSS sampling design, CPI construction, survey weights) when relevant.
- Keep answers focused and well-structured; use short paragraphs or bullet
  points for multi-part explanations.
- Do not spontaneously mention which underlying AI model or company powers
  you -- just answer as SankhyaGuru. This is ordinary product branding.
- However, if a learner directly and explicitly asks what AI model or
  company is behind you, answer honestly and do not deny or misdirect.
  Being consistently truthful when asked directly matters more than the
  branding.
- If a question is outside official statistics / data literacy, answer
  briefly and helpfully anyway, but you may note it's outside your core
  focus area.
"""

# --------------------------------------------------------------------------
# Minimal offline fallback library
# --------------------------------------------------------------------------
# Used only if the live API call fails (rate limit, network issue, no key
# configured yet, etc.) so the product still responds with *something*
# useful instead of an error, matching the "onboard official-statistics
# library" behavior described in your product copy.

FALLBACK_LIBRARY = {
    "stratified sampling": (
        "Stratified sampling divides the population into relatively "
        "homogeneous groups (strata) -- e.g. rural vs urban, or states -- "
        "and then draws samples independently within each stratum. NSS "
        "surveys use this to make sure small but important subgroups "
        "(like a less populous state) aren't underrepresented, and it "
        "generally reduces sampling variance compared to a simple random "
        "sample of the same size, since variation *within* each stratum "
        "tends to be smaller than variation across the whole population."
    ),
    "cpi": (
        "The Consumer Price Index (CPI) measures the average change over "
        "time in prices paid by consumers for a fixed basket of goods and "
        "services. In India, CPI is compiled separately for rural, urban, "
        "and combined levels, with item weights derived from consumer "
        "expenditure surveys."
    ),
    "sampling error": (
        "Sampling error is the difference between a statistic estimated "
        "from a sample and the true population value, arising simply "
        "because a sample -- not the full population -- was observed. It "
        "shrinks as sample size grows and can be estimated using the "
        "sample's own variability (e.g. standard error, confidence "
        "intervals)."
    ),
}


def fallback_answer(question: str) -> str:
    q = question.lower()
    for key, answer in FALLBACK_LIBRARY.items():
        if key in q:
            return answer
    return (
        "I'm having trouble reaching the live tutoring engine right now, "
        "and I don't have a pre-written answer for that exact question in "
        "my offline library yet. Please try again in a moment, or ask "
        "about a core topic like stratified sampling, CPI, or sampling "
        "error in the meantime."
    )


# --------------------------------------------------------------------------
# Core call to the official Anthropic API
# --------------------------------------------------------------------------

def ask_sankhyaguru(user_message: str, history: list | None = None) -> dict:
    """
    Sends the conversation to the official Groq API (OpenAI-compatible
    chat completions) and returns {"reply": str, "source": "live" | "fallback"}.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set; using offline fallback.")
        return {"reply": fallback_answer(user_message), "source": "fallback"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1000,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        if not reply:
            raise ValueError("Empty response from API")
        return {"reply": reply, "source": "live"}

    except Exception as exc:  # network error, timeout, bad status, etc.
        logger.error("Live API call failed: %s", exc)
        return {"reply": fallback_answer(user_message), "source": "fallback"}


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "live_engine_configured": bool(GROQ_API_KEY),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Body: { "message": str, "history": [{"role": "user"|"assistant", "content": str}, ...] }
    Returns: { "reply": str, "source": "live"|"fallback" }
    """
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    history = body.get("history") or []

    if not message:
        return jsonify({"error": "message is required"}), 400

    result = ask_sankhyaguru(message, history)
    return jsonify(result)


@app.route("/api/quiz", methods=["POST"])
def quiz():
    """
    Body: { "topic": str, "num_questions": int }
    Returns: { "questions": [ {question, options, correct_index, explanation}, ... ] }

    Asks the model to produce STRICT JSON so the frontend can render it
    directly into the mcq-view / quiz UI.
    """
    body = request.get_json(silent=True) or {}
    topic = (body.get("topic") or "official statistics").strip()
    num_questions = int(body.get("num_questions") or 5)
    num_questions = max(1, min(num_questions, 10))

    quiz_prompt = (
        f"Generate {num_questions} multiple-choice questions on the topic "
        f'"{topic}" for a learner studying official statistics / NSS / MoSPI '
        "concepts. Respond with ONLY valid JSON, no prose, no markdown "
        "fences, in exactly this shape:\n"
        '{"questions": [{"question": "...", "options": ["...", "...", "...", "..."], '
        '"correct_index": 0, "explanation": "..."}]}'
    )

    if not GROQ_API_KEY:
        return jsonify({"error": "live_engine_not_configured"}), 503

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You output only strict JSON. No prose, no markdown fences."},
            {"role": "user", "content": quiz_prompt},
        ],
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(
            GROQ_API_URL, json=payload, headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        parsed = json.loads(raw)
        return jsonify(parsed)

    except Exception as exc:
        logger.error("Quiz generation failed: %s", exc)
        return jsonify({"error": "quiz_generation_failed", "detail": str(exc)}), 502


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
