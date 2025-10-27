# app.py (Full Flask Script for Render Deployment)
# This script processes data from Google Sheets, summarizes qualitative responses using Gemini API,
# and returns summaries and recommendations. Deploy on Render with gunicorn.

from flask import Flask, request, jsonify
import pandas as pd
import requests
import os  # Added for environment variables (e.g., API key)

app = Flask(__name__)

# Gemini API configuration
# Use environment variable for security (set in Render Dashboard > Environment Variables)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')  # Fallback for testing
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"

# Questions and short names
QUESTIONS = [
    {
        "full": "Briefly explain your rating. What aspects of the program influenced your score the most?",
        "short": "program satisfaction",
        "col": "Briefly explain your rating. What aspects of the program influenced your score the most?"
    },
    {
        "full": "Optional: What additional suggestions would you recommend to enhance the overall learning experience?",
        "short": "learning experience",
        "col": "Optional: What additional suggestions would you recommend to enhance the overall learning experience?"
    },
    {
        "full": "OPTIONAL: What additional support or resources would you like to see introduced to enhance your experience in the program further? [open-ended]",
        "short": "program support",
        "col": "OPTIONAL: What additional support or resources would you like to see introduced to enhance your experience in the program further? [open-ended]"
    },
    {
        "full": "Please describe in detail what challenges you experienced",
        "short": "payment process",
        "col": "Please describe in detail what challenges you experienced"
    }
]

def call_gemini_api(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def parse_summary_output(output, cohort, question, short_name):
    themes = []
    lines = output.split("\n")
    current_theme = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Theme:"):
            if current_theme:
                themes.append([cohort, question, current_theme["theme"], current_theme["summary"], current_theme["samples"].strip()])
            current_theme = {"theme": line.replace("Theme:", "").strip(), "summary": "", "samples": ""}
        elif line.startswith("Summarised feedback:") and current_theme:
            current_theme["summary"] = line.replace("Summarised feedback:", "").strip()
        elif line.startswith("Sample Responses:") and current_theme:
            continue
        elif line.startswith('"') and current_theme:
            current_theme["samples"] += line + "\n"
    
    if current_theme and (current_theme["summary"] or current_theme["samples"]):
        themes.append([cohort, question, current_theme["theme"], current_theme["summary"], current_theme["samples"].strip()])
    
    return themes

def parse_rec_output(output, cohort):
    lines = [line.strip() for line in output.split("\n") if line.strip().startswith(tuple(f"{i}." for i in range(1, 10)))]
    return [[cohort, line] for line in lines]

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "rows" not in data or "headers" not in data:
            return jsonify({"error": "Invalid data format"}), 400
        
        # Convert data to DataFrame
        df = pd.DataFrame(data["rows"], columns=data["headers"])
        
        # Group by cohort
        cohorts = df.groupby("Cohort")
        summary_output = []
        recommendations_output = []
        
        for cohort, group in cohorts:
            # Process each question
            for q in QUESTIONS:
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue
                
                prompt = f"""Summarize the following responses to the question "{q['full']}" from cohort "{cohort}". Identify 3-5 main themes. For each theme, provide:
- Theme name
- Summarised feedback: Start with "X respondents similarly mentioned:" where X is the number of similar responses, followed by a concise summary.
- Sample Responses: 3-4 quoted snippets from responses, each on a new line.

Output in a structured list, one theme per block, like:
Theme: [name]
Summarised feedback: [X respondents similarly mentioned: text]
Sample Responses:
"[sample1]"
"[sample2]"
etc.

Responses:
{'\n---\n'.join(responses)}"""
                
                summary = call_gemini_api(prompt)
                themes = parse_summary_output(summary, cohort, q["full"], q["short"])
                summary_output.extend(themes)
            
            # Generate recommendations
            all_responses = []
            for q in QUESTIONS:
                all_responses.extend(group[q["col"]].dropna().tolist())
            
            if all_responses:
                rec_prompt = f"""Based on all open-ended responses from cohort "{cohort}" across questions on program satisfaction, learning experience, program support, and payment process, generate 3-4 actionable recommendations to improve the AiCE program. Each recommendation should be concise and specific. Output as a numbered list:
1. [rec1]
2. [rec2]
etc."""
                
                recs = call_gemini_api(rec_prompt)
                rec_lines = parse_rec_output(recs, cohort)
                recommendations_output.extend(rec_lines)
        
        return jsonify({
            "summary": summary_output,
            "recommendations": recommendations_output
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Debug for local testing; Render uses gunicorn
