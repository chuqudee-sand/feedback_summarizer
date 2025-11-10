from flask import Flask, request, jsonify
import pandas as pd
import os
from google import genai

app = Flask(__name__)

# Configure Gemini API client with API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Define questions and short names to process
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

# JSON schema describing expected structured output format from Gemini
summary_schema = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "summary": {"type": "string"},
                    "samples": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["theme", "summary", "samples"]
            }
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["themes", "recommendations"]
}

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "rows" not in data or "headers" not in data:
            return jsonify({"error": "Invalid data format"}), 400
        
        df = pd.DataFrame(data["rows"], columns=data["headers"])
        cohorts = df.groupby("Cohort")
        
        summary_output = []
        recommendations_output = []
        
        for cohort, group in cohorts:
            for q in QUESTIONS:
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue
                
                prompt_text = f"""Summarize the following responses to the question \"{q['full']}\" from cohort \"{cohort}\". Identify 3-5 main themes. For each theme, provide:
- theme name
- summary specifying how many respondents mentioned it
- 3-4 sample responses
                
Output strictly in this JSON format matching the schema:

{summary_schema}

Responses:
{'\n---\n'.join(responses)}
"""
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[{"text": prompt_text}],
                    response_mime_type="application/json",
                    response_json_schema=summary_schema
                )
                
                json_resp = response.json
                
                # Add structured themes to summary output
                for theme in json_resp.get("themes", []):
                    summary_output.append([
                        cohort, 
                        q["full"], 
                        q["short"],
                        theme["theme"],
                        theme["summary"],
                        "\n".join(theme["samples"])
                    ])
            
            # Recommendations based on all responses
            all_responses = []
            for q in QUESTIONS:
                all_responses.extend(group[q["col"]].dropna().tolist())
            
            if all_responses:
                rec_prompt = f"""Generate 3-4 concise, actionable recommendations to improve the AiCE program based on the following open-ended feedback from cohort \"{cohort}\":

{'\n---\n'.join(all_responses)}

Output as a numbered JSON list (array of strings).
"""
                rec_schema = {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 4
                }
                
                rec_resp = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[{"text": rec_prompt}],
                    response_mime_type="application/json",
                    response_json_schema=rec_schema
                )
                recommendations = rec_resp.json
                
                for rec in recommendations:
                    recommendations_output.append([cohort, rec])
        
        return jsonify({
            "summary": summary_output,
            "recommendations": recommendations_output
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
