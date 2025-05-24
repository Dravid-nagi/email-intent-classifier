# app.py
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load classifier and SBERT model
clf = joblib.load("llm_classifier.joblib")
sbert_model = SentenceTransformer("sbert_model/")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email_text", "")

    if not email_text:
        return jsonify({"error": "Missing email_text in request"}), 400

    # Get embedding and predict
    embedding = sbert_model.encode([email_text])
    prediction = clf.predict(embedding)[0]
    confidence = clf.predict_proba(embedding).max()

    return jsonify({
        "predicted_intent": prediction,
        "confidence_score": round(float(confidence), 4)
    })

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)

