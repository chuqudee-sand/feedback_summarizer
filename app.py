from flask import Flask, request, jsonify
import pandas as pd
import os
import json
from google import genai
from pydantic import BaseModel, ValidationError
from typing import List

app = Flask(__name__)

# Initialize Gemini client with API key from env variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Define full questions with exact header names on your sheet
QUESTIONS = [
    {
        "full": "Briefly explain your rating. What aspects of the program influenced your score the most?",
        "short": "program satisfaction",
        "col": "Briefly explain your rating. What aspects of the program influenced your score the most?"
    },
    {
        "full": "Optional: Please share any additional comments or suggestions you have for improving the AiCE program.",
        "short": "learning experience",
        "col": "Optional: Please share any additional comments or suggestions you have for improving the AiCE program."
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

# Pydantic models for validating structured JSON output
class Theme(BaseModel):
    theme: str
    summary: str
    samples: List[str]

class SummaryResponse(BaseModel):
    themes: List[Theme]
    recommendations: List[str]

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "rows" not in data or "headers" not in data:
            return jsonify({"error": "Invalid data format"}), 400

        df = pd.DataFrame(data["rows"], columns=data["headers"])

        summary_output = []
        recommendations_output = []

        cohorts = df.groupby("Cohort")

        for cohort, group in cohorts:
            for q in QUESTIONS:
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue

                # Clear and concise prompt asking for structured JSON string output
                prompt_text = f"""
You are an expert qualitative analyst summarizing survey data from cohort '{cohort}' for the question:

\"{q['full']}\"

Identify 3-5 main themes. For each, provide:
- theme title
- summary including approx respondent count
- 3-4 verbatim sample responses

Output ONLY a JSON string with this schema:
{{
  "themes": [
    {{
      "theme": "string",
      "summary": "string",
      "samples": ["string"]
    }}
  ],
  "recommendations": ["string"]
}}

Responses:
{'\n---\n'.join(responses)}
"""

                # Call Gemini API without response_mime_type (fallback for compatibility)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[{"text": prompt_text}]
                )
                raw_text = response.text.strip()

                # Try parse JSON string output manually
                try:
                    parsed_json = json.loads(raw_text)
                    summary_resp = SummaryResponse.parse_obj(parsed_json)

                    for theme in summary_resp.themes:
                        summary_output.append([
                            cohort,
                            q["full"],
                            q["short"],
                            theme.theme,
                            theme.summary,
                            "\n".join(theme.samples)
                        ])

                    for rec in summary_resp.recommendations:
                        recommendations_output.append([cohort, rec])

                except (json.JSONDecodeError, ValidationError) as e:
                    summary_output.append([
                        cohort,
                        q["full"],
                        q["short"],
                        "Parse Error",
                        f"Failed to parse JSON output: {e}",
                        raw_text[:200]  # Truncate for brevity
                    ])

        return jsonify({"summary": summary_output, "recommendations": recommendations_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

