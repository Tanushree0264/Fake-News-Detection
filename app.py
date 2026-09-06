from flask import Flask, render_template, request
import pickle
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    text = ""
    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if text:
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            confidence = round(float(max(probabilities)) * 100, 2)
            result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    return render_template("index.html", result=result, confidence=confidence, text=text)

if __name__ == "__main__":
    app.run(debug=True)
