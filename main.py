import os
import json
import time
import hashlib
import hmac
import httpx
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://alx-feedback-engine.vercel.app"],  # Replace with your Vercel URL e.g. ["https://your-app.vercel.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Clients ────────────────────────────────────────────────────────────────────
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Zoom credentials (Server-to-Server OAuth app) ──────────────────────────────
ZOOM_ACCOUNT_ID     = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID      = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET  = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_SECRET_TOKEN   = os.getenv("ZOOM_SECRET_TOKEN")   # Event Subscriptions → Secret Token


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — ZOOM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_zoom_access_token() -> str:
    """Exchange Server-to-Server OAuth credentials for a bearer token."""
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    response = httpx.post(
        url,
        auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    return response.json()["access_token"]


def zoom_get(path: str, token: str, params: dict = None) -> dict:
    """Generic authenticated GET to the Zoom API."""
    r = httpx.get(
        f"https://api.zoom.us/v2{path}",
        headers={"Authorization": f"Bearer {token}"},
        params=params or {},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def format_event_name_date(topic: str, start_time: str) -> str:
    """
    Formats event_name_date to match your existing convention:
    e.g. "VA C15 Grace Week Support Session - May 6, 2026"
    """
    try:
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        date_str = dt.strftime("%B %-d, %Y")   # "May 6, 2026"  (Linux)
    except Exception:
        date_str = start_time[:10]
    return f"{topic} - {date_str}"


def detect_program_and_event_type(topic: str) -> tuple[str, str]:
    """
    Infer `program` and `event_type` from the meeting topic.
    Adjust the keywords below to match your actual naming conventions.
    """
    topic_lower = topic.lower()

    # ── Program detection ─────────────────────────────────────────────────────
    if any(k in topic_lower for k in ["aice", "ai career", "ai community", "ai cohort", "tambali", "karibu", "portfolio office"]):
        program = "AiCE"
    elif any(k in topic_lower for k in ["virtual assistant", "va c", "va cohort", "va program"]):
        program = "Virtual Assistant"
    elif any(k in topic_lower for k in ["professional foundations", "pro found", "profound"]):
        program = "Professional Foundations"
    else:
        program = "AiCE"  # default fallback — adjust if needed

    # ── Event type detection ──────────────────────────────────────────────────
    if any(k in topic_lower for k in ["office hour", "support session", "project clinic", "program team", "webinar"]):
        event_type = "Program Team"
    elif any(k in topic_lower for k in ["mentorship", "mentor", "technical"]):
        event_type = "Technical Mentorship"
    else:
        event_type = "Community Event"

    return program, event_type


def collect_zoom_data(meeting_id: str, meeting_type: str = "meeting"):
    """
    Main Zoom data collector. Called in background after meeting.ended webhook.
    Fetches: participants (attendance_duration_mins) + poll results + survey results.
    Inserts one row per attendee into survey_events.
    """
    print(f"[Zoom] Starting data collection for {meeting_type} {meeting_id}")
    try:
        token = get_zoom_access_token()

        # ── 1. Get meeting details (topic + start time) ────────────────────────
        endpoint = "webinars" if meeting_type == "webinar" else "meetings"
        details = zoom_get(f"/{endpoint}/{meeting_id}", token)
        topic      = details.get("topic", "Unnamed Session")
        start_time = details.get("start_time", datetime.utcnow().isoformat())

        event_name_date = format_event_name_date(topic, start_time)
        program, event_type = detect_program_and_event_type(topic)

        print(f"[Zoom] Event: {event_name_date} | Program: {program} | Type: {event_type}")

        # ── 2. Get participant report (attendance duration per email) ──────────
        participant_map: dict[str, int] = {}   # email → duration in minutes
        part_endpoint = "webinars" if meeting_type == "webinar" else "meetings"
        try:
            participants_resp = zoom_get(f"/report/{part_endpoint}/{meeting_id}/participants", token)
            for p in participants_resp.get("participants", []):
                email = (p.get("user_email") or "").strip().lower()
                duration = p.get("duration", 0)  # in seconds from report API
                if email:
                    # Zoom report API returns duration in seconds
                    participant_map[email] = round(duration / 60)
        except Exception as e:
            print(f"[Zoom] Warning: Could not fetch participants: {e}")

        # ── 3. Get poll results ────────────────────────────────────────────────
        # poll_map: email → {question_text: answer_text}
        poll_map: dict[str, dict] = {}
        try:
            poll_endpoint = "webinars" if meeting_type == "webinar" else "meetings"
            polls_resp = zoom_get(f"/report/{poll_endpoint}/{meeting_id}/polls", token)
            for question_block in polls_resp.get("questions", []):
                email = (question_block.get("email") or "").strip().lower()
                if not email:
                    continue
                if email not in poll_map:
                    poll_map[email] = {}
                for qa in question_block.get("question_details", []):
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    poll_map[email][q] = a
        except Exception as e:
            print(f"[Zoom] Warning: Could not fetch polls: {e}")

        # ── 4. Get survey results ──────────────────────────────────────────────
        # survey_map: email → {question_text: answer_text}
        survey_map: dict[str, dict] = {}
        try:
            survey_endpoint = "webinars" if meeting_type == "webinar" else "meetings"
            survey_resp = zoom_get(f"/report/{survey_endpoint}/{meeting_id}/survey", token)
            for question_block in survey_resp.get("questions", []):
                email = (question_block.get("email") or "").strip().lower()
                if not email:
                    continue
                if email not in survey_map:
                    survey_map[email] = {}
                for qa in question_block.get("question_details", []):
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    survey_map[email][q] = a
        except Exception as e:
            print(f"[Zoom] Warning: Could not fetch survey: {e}")

        # ── 5. Merge all data sources and map to survey_events columns ─────────
        all_emails = set(participant_map.keys()) | set(poll_map.keys()) | set(survey_map.keys())

        if not all_emails:
            print(f"[Zoom] No participant data found for {meeting_id}. Skipping insert.")
            return

        rows_to_insert = []
        for email in all_emails:
            # Combine poll + survey answers for this respondent
            combined = {}
            combined.update(poll_map.get(email, {}))
            combined.update(survey_map.get(email, {}))  # survey answers override if duplicate key

            row = {
                "learner_email":   email,
                "program":         program,
                "event_type":      event_type,
                "event_name_date": event_name_date,
                "attendance_duration_mins": participant_map.get(email),
                # These will be populated below from poll/survey answers
                "session_quality_csat":        None,
                "understood_outcomes":         None,
                "improvement_suggestion_text": None,
                "challenging_topic_text":      None,
            }

            # ── Map question answers to your exact column names ────────────────
            # The keys below are SUBSTRINGS of your actual Zoom question text.
            # Edit them to match your real poll/survey question wording.
            for question_text, answer in combined.items():
                q = question_text.lower()

                # session_quality_csat  — expects integer 1–5
                if any(k in q for k in ["rate this session", "session quality", "how would you rate", "overall session"]):
                    try:
                        row["session_quality_csat"] = int(str(answer).strip()[0])
                    except Exception:
                        pass

                # understood_outcomes  — expects boolean
                elif any(k in q for k in ["understood", "learning outcome", "understand the objective", "clear on what"]):
                    row["understood_outcomes"] = str(answer).lower() in ["yes", "true", "1", "definitely", "absolutely"]

                # improvement_suggestion_text  — open text
                elif any(k in q for k in ["improve", "suggestion", "could be better", "what would make"]):
                    row["improvement_suggestion_text"] = str(answer).strip() if answer else None

                # challenging_topic_text  — open text
                elif any(k in q for k in ["challeng", "difficult", "hard to understand", "confus"]):
                    row["challenging_topic_text"] = str(answer).strip() if answer else None

            rows_to_insert.append(row)

        # ── 6. Upsert into Supabase (on learner_email + event_name_date) ───────
        # This prevents duplicates if the webhook fires twice
        supabase.table("survey_events").upsert(
            rows_to_insert,
            on_conflict="learner_email,event_name_date"
        ).execute()

        print(f"[Zoom] ✅ Inserted/updated {len(rows_to_insert)} rows for: {event_name_date}")

    except Exception as e:
        print(f"[Zoom] ❌ Error collecting data for meeting {meeting_id}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ZOOM WEBHOOK ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives all Zoom event webhooks.
    Handles:
      - URL validation challenge (required by Zoom on first setup)
      - meeting.ended  → triggers data collection
      - webinar.ended  → triggers data collection
    """
    body_bytes = await request.body()
    payload = json.loads(body_bytes)

    event = payload.get("event")

    # ── Step 1: Handle Zoom's URL validation challenge ─────────────────────────
    # Zoom sends this once when you add the webhook URL to verify it's reachable.
    if event == "endpoint.url_validation":
        plain_token = payload["payload"]["plainToken"]
        encrypted   = hmac.new(
            ZOOM_SECRET_TOKEN.encode("utf-8"),
            plain_token.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return {"plainToken": plain_token, "encryptedToken": encrypted}

    # ── Step 2: Verify webhook signature (security) ────────────────────────────
    # Zoom signs every request — reject anything that doesn't match.
    ts        = request.headers.get("x-zm-request-timestamp", "")
    signature = request.headers.get("x-zm-signature", "")
    message   = f"v0:{ts}:{body_bytes.decode('utf-8')}"
    expected  = "v0=" + hmac.new(
        ZOOM_SECRET_TOKEN.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid Zoom webhook signature")

    # ── Step 3: Reject stale webhooks (older than 5 minutes) ──────────────────
    if abs(time.time() - int(ts)) > 300:
        raise HTTPException(status_code=400, detail="Webhook timestamp too old")

    # ── Step 4: Route to data collector ───────────────────────────────────────
    if event == "meeting.ended":
        meeting_id = str(payload["payload"]["object"]["id"])
        background_tasks.add_task(collect_zoom_data, meeting_id, "meeting")
        print(f"[Zoom] meeting.ended received for {meeting_id}")

    elif event == "webinar.ended":
        webinar_id = str(payload["payload"]["object"]["id"])
        background_tasks.add_task(collect_zoom_data, webinar_id, "webinar")
        print(f"[Zoom] webinar.ended received for {webinar_id}")

    # Always return 200 immediately so Zoom doesn't retry
    return {"status": "received"}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — AI SUMMARY (unchanged from before)
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


def process_ai_summary(req: SummaryRequest):
    job_key = get_job_key(req)
    try:
        raw_text = ""

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
            for row in response.data:
                for col in ['improvement_suggestion_text', 'challenging_topic_text']:
                    if row.get(col):
                        raw_text += f"Feedback: {row[col]}\n"

        if len(raw_text.strip()) < 10:
            print(f"[{job_key}] Not enough feedback text found.")
            supabase.table("ai_summary_jobs").update({
                "status": "failed",
                "error_message": "Not enough learner feedback text found for this period."
            }).eq("job_key", job_key).execute()
            return

        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
You are an expert Data Analyst for a professional skills training program.
Analyze the following learner feedback responses.
Identify the 3 to 4 most prominent themes.

Return the result STRICTLY as a valid JSON array of objects with NO markdown, NO backticks, NO extra text.

Each object must have exactly these keys:
- "theme_title": A short 2-4 word title for the theme (e.g. "Platform Navigation Issues")
- "summary_text": A single sentence summarising what learners are saying about this theme
- "response_count": An integer — your best estimate of how many responses mention this theme
- "question_short": A short category label (e.g. "Improvement", "Positive Feedback", "Support Request")

Feedback to analyze:
{raw_text[:20000]}
"""
        result = model.generate_content(prompt)
        response_text = result.text.strip()
        if '```' in response_text:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        json_start = response_text.find('[')
        json_end   = response_text.rfind(']') + 1
        parsed_themes = json.loads(response_text[json_start:json_end])

        insert_rows = []
        for theme in parsed_themes:
            insert_rows.append({
                "program":        req.program,
                "tab_name":       req.activeTab,
                "question_short": theme.get("question_short", "General Feedback"),
                "theme_title":    theme.get("theme_title"),
                "response_count": theme.get("response_count", 1),
                "summary_text":   theme.get("summary_text"),
                "report_period":  req.reportPeriod,
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
        "job_key":          job_key,
        "program":          req.program,
        "tab_name":         req.activeTab,
        "report_period":    req.reportPeriod,
        "event_name_date":  req.activeEvent if req.activeTab in ['community', 'support'] else None,
        "status":           "busy",
        "error_message":    None,
    }, on_conflict="job_key").execute()
    background_tasks.add_task(process_ai_summary, req)
    return {"status": "processing", "message": "AI is reading and summarising feedback in the background."}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
