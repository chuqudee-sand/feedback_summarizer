import os
import json
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS so your Vercel frontend can trigger this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clients
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class SummaryRequest(BaseModel):
    program: str
    activeTab: str
    startDate: str
    endDate: str
    activeEvent: str
    reportPeriod: str

def process_ai_summary(req: SummaryRequest):
    """The heavy lifting logic moved from page.tsx to Python"""
    try:
        raw_text = ""
        
        # 1. Fetch Data based on Tab (using your _text column logic)
        if req.activeTab in ['onboarding', 'eop']:
            table = 'survey_onboarding' if req.activeTab == 'onboarding' else 'survey_eop'
            response = supabase.table(table).select("*")\
                .eq('program', req.program)\
                .gte('created_at', req.startDate)\
                .lte('created_at', req.endDate)\
                .execute()
            
            for row in response.data:
                # Target the specific open-ended columns you identified
                for col in ['unclear_aspects_text', 'additional_feedback_text', 'missing_info_text', 'additional_support_resources_text']:
                    if row.get(col):
                        raw_text += f"Feedback: {row[col]}\n"

        elif req.activeTab in ['community', 'support']:
            response = supabase.table('survey_events').select("*")\
                .eq('program', req.program)\
                .eq('event_name_date', req.activeEvent)\
                .execute()
            
            for row in response.data:
                # Target event-specific open-ended columns
                for col in ['improvement_suggestion_text', 'challenging_topic_text']:
                    if row.get(col):
                        raw_text += f"Feedback: {row[col]}\n"

        if len(raw_text.strip()) < 10:
            return

        # 2. Call Gemini 1.5 Flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert Data Analyst for an educational program. Analyze the following learner feedback.
        Identify the 3 to 4 most prominent themes. Return the result strictly as a JSON array of objects.
        Each object must have these exactly matching keys:
        - "theme_title": A short 2-4 word title for the theme.
        - "summary_text": A 1-sentence summary of what learners are saying.
        - "response_count": Your estimated number of mentions for this theme (integer).
        - "question_short": A short category like "General Feedback" or "Improvement".

        Feedback to analyze:
        {raw_text}
        """

        result = model.generate_content(prompt)
        # Clean the response in case Gemini includes markdown backticks
        clean_json = result.text.replace('```json', '').replace('```', '').strip()
        parsed_themes = json.loads(clean_json)

        # 3. Prepare rows for Supabase
        insert_rows = []
        for theme in parsed_themes:
            insert_rows.append({
                "program": req.program,
                "tab_name": req.activeTab,
                "question_short": theme.get("question_short", "General"),
                "theme_title": theme.get("theme_title"),
                "response_count": theme.get("response_count", 1),
                "summary_text": theme.get("summary_text"),
                "report_period": req.reportPeriod,
                "event_name_date": req.activeEvent if req.activeTab in ['community', 'support'] else None
            })

        # 4. Save to Supabase
        supabase.table("ai_thematic_summaries").insert(insert_rows).execute()
        print(f"Successfully saved summary for {req.program}")

    except Exception as e:
        print(f"Error processing AI summary: {str(e)}")

@app.post("/summarize")
async def trigger_summary(req: SummaryRequest, background_tasks: BackgroundTasks):
    # We add the task to background so the API returns 200 immediately
    background_tasks.add_task(process_ai_summary, req)
    return {"status": "processing", "message": "The AI is working in the background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
