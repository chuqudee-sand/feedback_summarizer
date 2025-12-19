from flask import Flask, request, jsonify
import pandas as pd
import json
import os
import re
from google import genai

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def parse_plain_text_summary(text):
    pattern = r"(Theme:.*?)(?=(?:Theme:|$))"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
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

        question_short_map = data.get("questionShortMap", {})
        headers = data["headers"]
        cohort_col = None
        for h in headers:
            if "cohort" in h.lower():
                cohort_col = h
                break
        if cohort_col is None:
            return jsonify({"error": "Cohort column not found"}), 400

        df = pd.DataFrame(data["rows"], columns=headers)
        summary_output = []

        cohorts = df.groupby(cohort_col)

        for cohort, group in cohorts:
            for col in headers:
                if col == cohort_col:
                    continue
                if col not in group.columns:
                    continue

                responses = group[col].dropna().tolist()
                if not responses:
                    continue

                question_short = question_short_map.get(col, col)

                prompt_text = f"""
You are an expert qualitative analyst summarizing survey responses from cohort '{cohort}' for the question:

\"{col}\"

Identify 3-5 main themes. For each theme, provide:
- The approximate number of participants or responses mentioning it, stated explicitly at the start of the summary (e.g., '35 learners expressed...')
- A concise theme title, introduced after 'Theme:'
- A well-structured summary starting with 'summary:' that elaborates on the theme with the count included

Please output in plain text using the format:

Theme: [theme title]
summary: [summary text including number of responses]

Responses:
{'\n---\n'.join(responses)}
"""

                # 🔄 AUTOMATIC RETRY LOGIC FOR QUOTA ERRORS
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[{"text": prompt_text}]
                        )
                        raw_text = response.text.strip()
                        break  # Success! Exit retry loop
                        
                    except Exception as api_error:
                        error_str = str(api_error).upper()
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "QUOTA" in error_str:
                            wait_time = (2 ** attempt) + 3  # 5s, 11s, 19s exponential backoff
                            print(f"Quota hit for {cohort}/{col}. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                            if attempt == max_retries - 1:
                                print(f"Failed after {max_retries} retries for {cohort}/{col}")
                                raw_text = "API quota exhausted after retries"
                        else:
                            # Non-quota error - re-raise immediately
                            raise api_error

                summary_text = parse_plain_text_summary(raw_text)
                if not summary_text:
                    summary_text = "No summary generated."

                summary_output.append([
                    cohort,
                    col,
                    question_short,
                    summary_text
                ])

        return jsonify({"summary": summary_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)





