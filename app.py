from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import torch

# Flask app
app = Flask(__name__, static_folder="static")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSON (ENSURE UTF-8)
with open("college_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract Q&A list
qa_list = data.get("qa_list", [])

questions = []
answers = []

# Convert qa_list into two arrays
for item in qa_list:
    questions.append(item["question"])
    answers.append(item["answer"])

# Encode all questions
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot API
@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_query = request.json.get("question", "")
        if user_query.strip() == "":
            return jsonify({"answer": "Please type something!"})

        # Encode user question
        user_emb = model.encode(user_query, convert_to_tensor=True)

        # Compare with all questions
        scores = util.cos_sim(user_emb, question_embeddings)[0]

        # Find best match
        best_idx = int(torch.argmax(scores))
        answer = answers[best_idx]

        return jsonify({"answer": answer})

    except Exception as e:
        print("Error:", e)
        return jsonify({"answer": "⚠️ Something went wrong!"})

# Run server
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

