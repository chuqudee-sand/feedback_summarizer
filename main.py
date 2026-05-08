import os
import json
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel URL in production e.g. ["https://your-app.vercel.app"]
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class SummaryRequest(BaseModel):
    program: str
    activeTab: str
    startDate: str
    endDate: str
    activeEvent: str
    reportPeriod: str


def get_job_key(req: SummaryRequest) -> str:
    """Creates a unique key for this specific summarization job."""
    if req.activeTab in ['community', 'support']:
        return f"{req.program}|{req.activeTab}|{req.activeEvent}"
    return f"{req.program}|{req.activeTab}|{req.reportPeriod}"


def process_ai_summary(req: SummaryRequest):
    """Background task: fetch feedback → call Gemini → save to Supabase → update job status."""
    job_key = get_job_key(req)

    try:
        raw_text = ""

        # 1. Fetch open-ended feedback text from Supabase
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
            print(f"[{job_key}] Not enough feedback text found. Marking job as failed.")
            supabase.table("ai_summary_jobs").update({
                "status": "failed",
                "error_message": "Not enough learner feedback text found for this period."
            }).eq("job_key", job_key).execute()
            return

        # 2. Call Gemini (using gemini-2.0-flash — the correct current model)
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

        # Robust JSON extraction — strip markdown if Gemini includes it
        if '```' in response_text:
            response_text = response_text.replace('```json', '').replace('```', '').strip()

        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        json_string = response_text[json_start:json_end]
        parsed_themes = json.loads(json_string)

        # 3. Build insert rows for ai_thematic_summaries
        insert_rows = []
        for theme in parsed_themes:
            insert_rows.append({
                "program": req.program,
                "tab_name": req.activeTab,
                "question_short": theme.get("question_short", "General Feedback"),
                "theme_title": theme.get("theme_title"),
                "response_count": theme.get("response_count", 1),
                "summary_text": theme.get("summary_text"),
                "report_period": req.reportPeriod,
                "event_name_date": req.activeEvent if req.activeTab in ['community', 'support'] else None
            })

        supabase.table("ai_thematic_summaries").insert(insert_rows).execute()

        # 4. Mark job as done
        supabase.table("ai_summary_jobs").update({
            "status": "done",
            "error_message": None
        }).eq("job_key", job_key).execute()

        print(f"[{job_key}] ✅ Summary saved successfully.")

    except Exception as e:
        print(f"[{job_key}] ❌ Error: {str(e)}")
        try:
            supabase.table("ai_summary_jobs").update({
                "status": "failed",
                "error_message": str(e)
            }).eq("job_key", job_key).execute()
        except Exception as inner:
            print(f"[{job_key}] Could not update job status: {str(inner)}")


@app.post("/summarize")
async def trigger_summary(req: SummaryRequest, background_tasks: BackgroundTasks):
    """
    Immediately returns 200 and fires the AI job in the background.
    Also writes a 'busy' row to ai_summary_jobs so the frontend can show a spinner.
    """
    job_key = get_job_key(req)

    # Upsert a 'busy' status into Supabase so the frontend knows work is in progress
    supabase.table("ai_summary_jobs").upsert({
        "job_key": job_key,
        "program": req.program,
        "tab_name": req.activeTab,
        "report_period": req.reportPeriod,
        "event_name_date": req.activeEvent if req.activeTab in ['community', 'support'] else None,
        "status": "busy",
        "error_message": None
    }, on_conflict="job_key").execute()

    background_tasks.add_task(process_ai_summary, req)

    return {"status": "processing", "message": "AI is reading and summarising feedback in the background."}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
