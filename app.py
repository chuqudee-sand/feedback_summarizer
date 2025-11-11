from flask import Flask, request, jsonify
import pandas as pd
import json
import os
from openai import OpenAI
from typing import List

app = Flask(__name__)

# Load DeepSeek API key from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

# Initialize OpenAI client using DeepSeek's base URL for API compatibility
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# Full questions list (must match exact Google Sheet columns)
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
                if q["col"] not in group.columns:
                    continue
                
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue

                prompt = f"""
You are an expert qualitative analyst summarizing survey responses from cohort '{cohort}' for the question:

\"{q['full']}\"

Identify 3 to 5 main themes. For each theme, provide:
- A concise theme title
- A summary including approximately how many respondents mentioned it
- 3 to 4 representative sample quotes verbatim

Respond ONLY with a valid JSON string matching this schema:

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

                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are an expert qualitative analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )

                raw_text = completion.choices[0].message.content.strip()

                try:
                    parsed = json.loads(raw_text)
                    themes = parsed.get("themes", [])
                    recs = parsed.get("recommendations", [])

                    for theme in themes:
                        summary_output.append([
                            cohort,
                            q["full"],
                            q["short"],
                            theme.get("theme", ""),
                            theme.get("summary", ""),
                            "\n".join(theme.get("samples", []))
                        ])

                    for rec in recs:
                        recommendations_output.append([cohort, rec])

                except json.JSONDecodeError:
                    summary_output.append([
                        cohort,
                        q["full"],
                        q["short"],
                        "Parse Error",
                        "Failed to parse JSON output",
                        raw_text[:200]
                    ])

        return jsonify({
            "summary": summary_output,
            "recommendations": recommendations_output
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

