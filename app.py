from flask import Flask, request, jsonify
import pandas as pd
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from typing import List

app = Flask(__name__)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

QUESTIONS = [
    {
        "full": "Briefly explain your rating. What aspects of the program influenced your score the most?",
        "short": "program satisfaction",
        "col": "Briefly explain your rating. What aspects of the program influenced your score the most?"
    },
    {
        "full": "OPTIONAL: Please share any additional comments or suggestions you have for improving the AiCE program.",
        "short": "learning experience",
        "col": "OPTIONAL: Please share any additional comments or suggestions you have for improving the AiCE program."
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

class Theme(BaseModel):
    theme: str
    summary: str
    samples: List[str]

class SummaryResponse(BaseModel):
    themes: List[Theme]
    recommendations: List[str]

MAX_RESPONSES_PER_CHUNK = 20

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_summary_prompt(cohort, question_full, responses):
    prompt = f"""
You are an expert qualitative analyst summarizing survey responses from cohort '{cohort}' for the question:

\"{question_full}\"

Identify 3 to 5 main themes. For each theme, provide:
- A concise theme title
- A summary including approximately how many respondents mentioned it
- 3 to 4 representative sample quotes verbatim

Output **only** a valid JSON string matching this schema exactly (no extra text):

{{
  "themes": [
    {{
      "theme": "string",
      "summary": "string",
      "samples": ["string"]
    }}
  ],
  "recommendations": []
}}

Responses (separate each by "---"):

{'\n---\n'.join(responses)}
"""
    return prompt

def generate_recommendations_prompt(cohort, all_feedback):
    prompt = f"""
Based on the following open-ended feedback from cohort '{cohort}', generate 3 to 4 numbered, concise, actionable recommendations to improve the AiCE program.

Return ONLY the recommendations as a JSON array of strings (e.g. ["rec 1", "rec 2", ...]) without any other commentary.

Feedback (each separated by ---):

{'\n---\n'.join(all_feedback)}
"""
    return prompt

def call_gemini_api(prompt_text):
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.7,
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Part.from_text(text=prompt_text)],
        config=config,
    )
    return response.text.strip()

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "rows" not in data or "headers" not in data:
            return jsonify({"error": "Invalid input data format"}), 400

        df = pd.DataFrame(data["rows"], columns=data["headers"])
        cohorts = df.groupby("Cohort")

        summary_output = []
        recommendations_output = []

        for cohort, group in cohorts:
            all_text_for_recs = []

            for q in QUESTIONS:
                if q["col"] not in group.columns:
                    continue
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue
                all_text_for_recs.extend(responses)

                chunks = list(chunk_list(responses, MAX_RESPONSES_PER_CHUNK))
                themes_accum = []

                for chunk in chunks:
                    prompt_text = generate_summary_prompt(cohort, q["full"], chunk)
                    raw_text = call_gemini_api(prompt_text)

                    try:
                        parsed_json = json.loads(raw_text)
                        summary_resp = SummaryResponse.parse_obj(parsed_json)
                        themes_accum.extend(summary_resp.themes)
                    except (json.JSONDecodeError, ValidationError) as e:
                        summary_output.append([
                            cohort, q["full"], q["short"],
                            "Parse Error",
                            f"Failed to parse JSON output: {e}",
                            raw_text[:200]
                        ])
                        break

                for theme in themes_accum[:3]:
                    summary_output.append([
                        cohort,
                        q["full"],
                        q["short"],
                        theme.theme,
                        theme.summary,
                        "\n".join(theme.samples)
                    ])

            if all_text_for_recs:
                rec_prompt = generate_recommendations_prompt(cohort, all_text_for_recs)
                raw_rec_text = call_gemini_api(rec_prompt)

                try:
                    recs = json.loads(raw_rec_text)
                    if not isinstance(recs, list):
                        raise ValueError("Recommendations response is not a list")
                    for rec in recs:
                        recommendations_output.append([cohort, rec])
                except Exception as e:
                    recommendations_output.append([cohort, f"Failed to parse recommendations: {e}"])

        return jsonify({"summary": summary_output, "recommendations": recommendations_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

