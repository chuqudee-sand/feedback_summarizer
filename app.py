from flask import Flask, request, jsonify
import pandas as pd
from google import genai
from google.genai import types
import os

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

Return your summary in plain text only (no JSON).

Responses (each separated by "---"):

{'\n---\n'.join(responses)}
"""
    return prompt

def generate_recommendations_prompt(cohort, all_feedback):
    prompt = f"""
Based on the following open-ended feedback from cohort '{cohort}', generate 3 to 4 numbered, concise, actionable recommendations to improve the AiCE program.

Return your recommendations in plain text only (no JSON).

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

        summary_texts = []
        recommendation_texts = []

        for cohort, group in cohorts:
            all_text_for_recs = []

            for q in QUESTIONS:
                if q["col"] not in group.columns:
                    continue
                responses = group[q["col"]].dropna().tolist()
                if not responses:
                    continue
                all_text_for_recs.extend(responses)

                # Chunk inputs
                chunks = list(chunk_list(responses, MAX_RESPONSES_PER_CHUNK))
                for chunk in chunks:
                    prompt_text = generate_summary_prompt(cohort, q["full"], chunk)
                    raw_text = call_gemini_api(prompt_text)
                    summary_texts.append({
                        "cohort": cohort,
                        "question": q["full"],
                        "question_short": q["short"],
                        "summary": raw_text
                    })

            if all_text_for_recs:
                rec_prompt = generate_recommendations_prompt(cohort, all_text_for_recs)
                raw_rec_text = call_gemini_api(rec_prompt)
                recommendation_texts.append({
                    "cohort": cohort,
                    "recommendations": raw_rec_text
                })

        return jsonify({"summary": summary_texts, "recommendations": recommendation_texts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

