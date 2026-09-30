import os
import json
import hashlib
import hmac
import httpx
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alx-feedback-engine.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ────────────────────────────────────────────────────────────────────
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

# ── LLM provider keys ─────────────────────────────────────────────────────────
# Primary:  Hugging Face Serverless Inference API (free, no model download,
#           runs entirely on HF's servers — safe on Render's memory limits)
# Fallback: Groq (ultra-fast, free tier, Llama 3 70B) — used automatically if
#           Hugging Face errors out or is slow to respond
HF_API_KEY   = os.getenv("HF_API_KEY")    # huggingface.co → Settings → Access Tokens
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # console.groq.com → API Keys

HF_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
GROQ_MODEL = "llama3-70b-8192"

ZOOM_ACCOUNT_ID    = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID     = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")

PROGRAM_EMAIL_MAP = {
    "aice@alxafrica.com":           "AiCE",
    "vaprogram@alxafrica.com":      "Virtual Assistant",
    "alxfoundations@alxafrica.com": "Professional Foundations",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ZOOM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_zoom_access_token() -> str:
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    r = httpx.post(url, auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET),
                   headers={"Content-Type": "application/x-www-form-urlencoded"})
    r.raise_for_status()
    return r.json()["access_token"]


def zoom_get(path: str, token: str, params: dict = None) -> dict:
    r = httpx.get(f"https://api.zoom.us/v2{path}",
                  headers={"Authorization": f"Bearer {token}"},
                  params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def detect_program(host_email: str, topic: str, cohost_emails: list[str] = None) -> str:
    program = PROGRAM_EMAIL_MAP.get((host_email or "").strip().lower())
    if not program and cohost_emails:
        for c in cohost_emails:
            program = PROGRAM_EMAIL_MAP.get((c or "").strip().lower())
            if program:
                print(f"[Zoom] Program from co-host: {c} → {program}")
                break
    if not program:
        t = topic.lower()
        if any(k in t for k in ["aice", "ai career", "tambali", "karibu"]):
            program = "AiCE"
        elif any(k in t for k in ["virtual assistant", "va c", "va cohort", "deep dive"]):
            program = "Virtual Assistant"
        elif any(k in t for k in ["professional foundations", "pro found"]):
            program = "Professional Foundations"
        else:
            program = "AiCE"
            print(f"[Zoom] WARNING: Could not detect program for host='{host_email}' topic='{topic}'. Defaulted to AiCE.")
    return program


def map_answer(question: str, answer: str) -> dict:
    """Map a Zoom poll/survey question+answer to survey_events columns."""
    q = question.lower().strip()
    a = (answer or "").strip()

    if "how would you rate today" in q or "rate today's session" in q:
        try:
            val = int(str(a).strip()[0])
            if 1 <= val <= 5:
                return {"session_quality_csat": val}
        except Exception:
            pass

    elif "understand the learning outcome" in q or "did you understand" in q:
        return {"understood_outcomes": a.lower() in ["yes", "true", "1"]}

    elif "one thing that would" in q and "useful" in q:
        return {"improvement_suggestion_text": a if a and a.lower() != "n/a" else None}

    elif "module or topic" in q or "most challenging" in q or "find most challeng" in q:
        return {"challenging_topic_text": a if a and a.lower() != "n/a" else None}

    return {}


def process_pending_job(job: dict):
    """
    Processes one pending Zoom job from zoom_pending_jobs.
    Fetches full participant report + polls + survey from Zoom API
    (called ~1 hour after meeting ends so all data is finalised),
    then upserts rows into survey_events.
    """
    meeting_id   = job["meeting_id"]
    meeting_type = job["meeting_type"]   # 'meetings' or 'webinars'
    topic        = job["topic"]
    host_email   = job["host_email"] or ""
    event_name_date = job["event_name_date"]
    event_type   = job["event_type"]
    job_id       = job["id"]

    print(f"[Collector] Processing job {job_id}: {event_name_date}")

    # Mark as processing to prevent double-runs
    supabase.table("zoom_pending_jobs").update({
        "status": "processing"
    }).eq("id", job_id).execute()

    try:
        token = get_zoom_access_token()

        # ── 1. Participants (attendance duration + co-host detection) ─────────
        participant_map: dict[str, int] = {}  # email → total minutes
        cohost_emails:   list[str]      = []

        try:
            resp = zoom_get(f"/report/{meeting_type}/{meeting_id}/participants", token)
            # Handle paginated responses
            participants = resp.get("participants", [])
            next_token = resp.get("next_page_token", "")
            while next_token:
                resp = zoom_get(f"/report/{meeting_type}/{meeting_id}/participants",
                                token, params={"next_page_token": next_token})
                participants += resp.get("participants", [])
                next_token = resp.get("next_page_token", "")

            for p in participants:
                email    = (p.get("user_email") or "").strip().lower()
                duration = p.get("duration", 0)   # seconds
                role     = p.get("role", 0)
                if email:
                    participant_map[email] = participant_map.get(email, 0) + round(duration / 60)
                    if role == 2:
                        cohost_emails.append(email)
        except Exception as e:
            print(f"[Collector] Warning — participants fetch failed: {e}")

        # ── 2. Detect program (now with co-host info available) ───────────────
        program = job.get("program") or detect_program(host_email, topic, cohost_emails)

        # Block unrecognised sessions — neither host nor co-host is a program account
        if not job.get("program") and not any(
            PROGRAM_EMAIL_MAP.get((e or "").lower()) for e in [host_email] + cohost_emails
        ):
            print(f"[Collector] Blocked job {job_id} — unrecognised host and co-hosts")
            supabase.table("zoom_pending_jobs").update({
                "status": "failed",
                "error_message": f"Unrecognised host '{host_email}' — not a program account",
                "processed_at": datetime.utcnow().isoformat(),
            }).eq("id", job_id).execute()
            return

        print(f"[Collector] Program: {program} | Participants: {len(participant_map)}")

        # ── 3. Polls ──────────────────────────────────────────────────────────
        poll_map: dict[str, dict] = {}   # email → {question: answer}
        try:
            polls_resp = zoom_get(f"/report/{meeting_type}/{meeting_id}/polls", token)
            for block in polls_resp.get("questions", []):
                email = (block.get("email") or "").strip().lower()
                if not email:
                    continue
                poll_map.setdefault(email, {})
                for qa in block.get("question_details", []):
                    poll_map[email][qa.get("question", "")] = qa.get("answer", "")
        except Exception as e:
            print(f"[Collector] Warning — polls fetch failed: {e}")

        # ── 4. Survey ─────────────────────────────────────────────────────────
        # Zoom post-webinar surveys can be anonymous OR identified.
        # If anonymous, email will be empty — we store with placeholder email.
        survey_map:       dict[str, dict] = {}   # email → {question: answer}
        anonymous_surveys: list[dict]     = []   # for anonymous responses

        try:
            survey_resp = zoom_get(f"/report/{meeting_type}/{meeting_id}/survey", token)
            for block in survey_resp.get("questions", []):
                email = (block.get("email") or "").strip().lower()
                answers = {}
                for qa in block.get("question_details", []):
                    answers[qa.get("question", "")] = qa.get("answer", "")

                if email:
                    survey_map.setdefault(email, {}).update(answers)
                else:
                    # Anonymous response — store separately
                    anonymous_surveys.append(answers)
        except Exception as e:
            print(f"[Collector] Warning — survey fetch failed: {e}")

        # ── 5. Build rows ─────────────────────────────────────────────────────
        all_emails = set(participant_map.keys()) | set(poll_map.keys()) | set(survey_map.keys())

        rows_to_upsert = []

        for email in all_emails:
            combined = {}
            combined.update(poll_map.get(email, {}))
            combined.update(survey_map.get(email, {}))  # survey overrides poll if duplicate Q

            row = {
                "learner_email":               email,
                "program":                     program,
                "event_type":                  event_type,
                "event_name_date":             event_name_date,
                "attendance_duration_mins":    participant_map.get(email),
                "session_quality_csat":        None,
                "understood_outcomes":         None,
                "improvement_suggestion_text": None,
                "challenging_topic_text":      None,
            }
            for question, answer in combined.items():
                row.update(map_answer(question, answer))
            rows_to_upsert.append(row)

        # Anonymous survey responses — placeholder email per response
        for i, answers in enumerate(anonymous_surveys, 1):
            row = {
                "learner_email":               f"anon.survey.{meeting_id}.{i}@zoom.placeholder",
                "program":                     program,
                "event_type":                  event_type,
                "event_name_date":             event_name_date,
                "attendance_duration_mins":    None,
                "session_quality_csat":        None,
                "understood_outcomes":         None,
                "improvement_suggestion_text": None,
                "challenging_topic_text":      None,
            }
            for question, answer in answers.items():
                row.update(map_answer(question, answer))
            rows_to_upsert.append(row)

        if not rows_to_upsert:
            print(f"[Collector] No data found for job {job_id}. Marking done anyway.")
            supabase.table("zoom_pending_jobs").update({
                "status": "done",
                "error_message": "No participant or survey data found",
                "processed_at": datetime.utcnow().isoformat(),
            }).eq("id", job_id).execute()
            return

        # ── 6. Upsert into survey_events ──────────────────────────────────────
        supabase.table("survey_events").upsert(
            rows_to_upsert, on_conflict="learner_email,event_name_date"
        ).execute()

        # ── 7. Mark job done ──────────────────────────────────────────────────
        supabase.table("zoom_pending_jobs").update({
            "status": "done",
            "processed_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()

        print(f"[Collector] ✅ Job {job_id} done — {len(rows_to_upsert)} rows for: {event_name_date}")

    except Exception as e:
        print(f"[Collector] ❌ Job {job_id} failed: {e}")
        supabase.table("zoom_pending_jobs").update({
            "status": "failed",
            "error_message": str(e),
            "processed_at": datetime.utcnow().isoformat(),
        }).eq("id", job_id).execute()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — HOURLY COLLECTOR ENDPOINT
#  Render Cron Job calls GET /collect-zoom-data every hour.
#  It picks up all jobs that have been pending for at least 60 minutes,
#  giving Zoom's report API plenty of time to fully populate.
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/collect-zoom-data")
async def collect_zoom_data(background_tasks: BackgroundTasks):
    """
    Called by Render's cron job every hour.
    Picks up all pending jobs older than 60 minutes and processes them.
    Returns immediately — processing runs in background.
    """
    # Fetch pending jobs that were created at least 60 minutes ago
    result = supabase.table("zoom_pending_jobs") \
        .select("*") \
        .eq("status", "pending") \
        .lt("created_at", (datetime.utcnow().replace(microsecond=0).isoformat() + "Z")) \
        .execute()

    jobs = result.data or []

    # Filter to only jobs older than 60 minutes in Python
    # (Supabase free tier doesn't support interval arithmetic easily)
    from datetime import timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=60)
    eligible = []
    for job in jobs:
        try:
            created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
            if created <= cutoff:
                eligible.append(job)
        except Exception:
            pass

    if not eligible:
        print("[Collector] No eligible pending jobs found.")
        return {"status": "ok", "jobs_queued": 0}

    print(f"[Collector] Found {len(eligible)} job(s) to process")
    for job in eligible:
        background_tasks.add_task(process_pending_job, job)

    return {"status": "ok", "jobs_queued": len(eligible),
            "meetings": [j["event_name_date"] for j in eligible]}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — AI SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class SummaryRequest(BaseModel):
    program: str
    activeTab: str
    startDate: str
    endDate: str
    activeEvent: str
    reportPeriod: str


def get_job_key(req: SummaryRequest) -> str:
    if req.activeTab in ['community', 'support']:
        return f"{req.program}|{req.activeTab}|{req.activeEvent}"
    return f"{req.program}|{req.activeTab}|{req.reportPeriod}"


PROMPT_TEMPLATE = """You are an expert Data Analyst for a professional skills training program.
Analyze the following session data and learner feedback.
Identify the 3 to 4 most important insights or themes.

Return the result STRICTLY as a valid JSON array of objects with NO markdown, NO backticks, NO extra text.

Each object must have exactly these keys:
- "theme_title": A short 2-4 word title for the insight
- "summary_text": A single sentence summarising the insight, referencing specific numbers where available
- "response_count": An integer — number of responses this insight is based on
- "question_short": A short category label (e.g. "CSAT", "Learning Outcomes", "Improvement", "Positive Feedback")

Data to analyze:
{analysis_input}
"""


def call_huggingface(prompt: str) -> str:
    """Call Hugging Face Serverless Inference API — routes to HF's cloud,
    no model download, no Render memory pressure."""
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY not set")

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": HF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
        "stream": False,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_groq(prompt: str) -> str:
    """Fallback: Groq inference API (Llama 3 70B). Ultra-fast, free tier.
    Used automatically when Hugging Face is slow or unavailable.
    Get a free key at console.groq.com"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    response = httpx.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_summary(prompt: str) -> str:
    """Tries Hugging Face first, falls back to Groq automatically.
    Returns the raw text response from whichever provider succeeds."""
    if HF_API_KEY:
        try:
            print("[LLM] Trying Hugging Face (Mistral-7B)...")
            result = call_huggingface(prompt)
            print("[LLM] \u2705 Hugging Face responded successfully")
            return result
        except Exception as hf_err:
            print(f"[LLM] \u26a0\ufe0f  Hugging Face failed: {hf_err} \u2014 falling back to Groq")

    if GROQ_API_KEY:
        try:
            print("[LLM] Trying Groq (Llama 3 70B)...")
            result = call_groq(prompt)
            print("[LLM] \u2705 Groq responded successfully")
            return result
        except Exception as groq_err:
            print(f"[LLM] \u274c Groq also failed: {groq_err}")
            raise groq_err

    raise ValueError("No LLM provider available \u2014 set HF_API_KEY or GROQ_API_KEY in environment variables")


def process_ai_summary(req: SummaryRequest):
    job_key = get_job_key(req)
    try:
        raw_text = ""
        structured_context = ""

        if req.activeTab in ['onboarding', 'eop']:
            table = 'survey_onboarding' if req.activeTab == 'onboarding' else 'survey_eop'
            response = supabase.table(table).select("*") \
                .eq('program', req.program) \
                .gte('created_at', req.startDate) \
                .lte('created_at', req.endDate) \
                .execute()
            for row in response.data:
                for col in ['unclear_aspects_text', 'additional_feedback_text',
                            'missing_info_text', 'additional_support_resources_text']:
                    if row.get(col):
                        raw_text += f"Feedback: {row[col]}\n"

        elif req.activeTab in ['community', 'support']:
            response = supabase.table('survey_events').select("*") \
                .eq('program', req.program) \
                .eq('event_name_date', req.activeEvent) \
                .execute()
            rows = response.data or []

            total_rows   = len(rows)
            csat_responses = [r['session_quality_csat'] for r in rows if r.get('session_quality_csat') is not None]
            csat_total   = len(csat_responses)
            csat_high    = sum(1 for v in csat_responses if v >= 4)
            csat_low     = sum(1 for v in csat_responses if v <= 2)
            csat_pct     = round((csat_high / csat_total * 100), 1) if csat_total else 0
            avg_csat     = round(sum(csat_responses) / csat_total, 2) if csat_total else 0

            outcomes_responses = [r['understood_outcomes'] for r in rows if r.get('understood_outcomes') is not None]
            outcomes_total = len(outcomes_responses)
            outcomes_yes   = sum(1 for v in outcomes_responses if v is True)
            outcomes_pct   = round((outcomes_yes / outcomes_total * 100), 1) if outcomes_total else None

            for row in rows:
                for col in ['improvement_suggestion_text', 'challenging_topic_text']:
                    if row.get(col):
                        raw_text += f"Feedback: {row[col]}\n"

            structured_context = f"""
SESSION METRICS SUMMARY ({req.activeEvent}):
- Total attendees: {total_rows}
- CSAT responses: {csat_total} out of {total_rows} attendees
- High satisfaction (4-5 rating): {csat_high} responses ({csat_pct}%)
- Low satisfaction (1-2 rating): {csat_low} responses
- Average CSAT score: {avg_csat} / 5.0
"""
            if outcomes_pct is not None:
                structured_context += f"- Understood learning outcomes: {outcomes_yes}/{outcomes_total} ({outcomes_pct}%)\n"
            if raw_text.strip():
                structured_context += f"\nOPEN-ENDED FEEDBACK RESPONSES:\n{raw_text[:15000]}"

        if not structured_context.strip() and len(raw_text.strip()) < 10:
            print(f"[{job_key}] Not enough data found.")
            supabase.table("ai_summary_jobs").update({
                "status": "failed",
                "error_message": "Not enough learner feedback data found for this period."
            }).eq("job_key", job_key).execute()
            return

        analysis_input = structured_context if req.activeTab in ['community', 'support'] else raw_text[:20000]
        prompt = PROMPT_TEMPLATE.format(analysis_input=analysis_input)
        response_text = generate_summary(prompt)
        if '```' in response_text:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        json_start    = response_text.find('[')
        json_end      = response_text.rfind(']') + 1
        parsed_themes = json.loads(response_text[json_start:json_end])

        insert_rows = []
        for theme in parsed_themes:
            insert_rows.append({
                "program":         req.program,
                "tab_name":        req.activeTab,
                "question_short":  theme.get("question_short", "General Feedback"),
                "theme_title":     theme.get("theme_title"),
                "response_count":  theme.get("response_count", 1),
                "summary_text":    theme.get("summary_text"),
                "report_period":   req.reportPeriod,
                "event_name_date": req.activeEvent if req.activeTab in ['community', 'support'] else None,
            })

        supabase.table("ai_thematic_summaries").insert(insert_rows).execute()
        supabase.table("ai_summary_jobs").update({
            "status": "done", "error_message": None
        }).eq("job_key", job_key).execute()
        print(f"[{job_key}] ✅ Summary saved.")

    except Exception as e:
        print(f"[{job_key}] ❌ Error: {e}")
        try:
            supabase.table("ai_summary_jobs").update({
                "status": "failed", "error_message": str(e)
            }).eq("job_key", job_key).execute()
        except Exception:
            pass


@app.post("/summarize")
async def trigger_summary(req: SummaryRequest, background_tasks: BackgroundTasks):
    job_key = get_job_key(req)
    supabase.table("ai_summary_jobs").upsert({
        "job_key":         job_key,
        "program":         req.program,
        "tab_name":        req.activeTab,
        "report_period":   req.reportPeriod,
        "event_name_date": req.activeEvent if req.activeTab in ['community', 'support'] else None,
        "status":          "busy",
        "error_message":   None,
    }, on_conflict="job_key").execute()
    background_tasks.add_task(process_ai_summary, req)
    return {"status": "processing", "message": "AI is reading and summarising feedback in the background."}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "service": "ALX Feedback Engine"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.head("/")
async def root_head():
    return {}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
