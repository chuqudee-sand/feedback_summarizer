from flask import Flask, request, jsonify
import pandas as pd
import json
import os
import re
from google import genai
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

def parse_plain_text_summary(text):
    # This regex matches blocks starting with Theme: followed by summary lines until next Theme or end
    pattern = r"(Theme:.*?)(?=(?:Theme:|$))"
    matches = re.findall(pattern, text, flags=re.DOTALL|re.IGNORECASE)
    summaries = []
    for block in matches:
        clean_text = block.strip().replace("\n", " ")
        summaries.append(clean_text)
    return "\n\n".join(summaries)

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "rows" not in data or "headers" not in data:
            return jsonify({"error": "Invalid data format"}), 400

        df = pd.DataFrame(data["rows"], columns=data["headers"])

        summary_output = []

        cohorts = df.groupby("Cohort")

        for cohort, group in cohorts:
            for q in QUESTIONS:
                if q["col"] not in group.columns:
                    continue
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue

                prompt_text = f"""
You are an expert qualitative analyst summarizing survey responses from cohort '{cohort}' for the question:

\"{q['full']}\"

Identify 3-5 main themes. For each theme, provide:
- The approximate number of participants or responses mentioning it, stated explicitly at the start of the summary (e.g., '35 participants expressed...')
- A concise theme title, introduced after 'Theme:'
- A well-structured summary starting with 'summary:' that elaborates on the theme with the count included

Please output in plain text using the format:

Theme: [theme title]
summary: [summary text including number of responses]

Responses:
{'\n---\n'.join(responses)}
"""

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[{"text": prompt_text}]
                )
                raw_text = response.text.strip()

                summary_text = parse_plain_text_summary(raw_text)

                if not summary_text:
                    summary_text = "No summary generated."

                summary_output.append([
                    cohort,
                    q["full"],
                    q["short"],
                    summary_text
                ])

        return jsonify({"summary": summary_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


