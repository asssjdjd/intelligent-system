from flask import Flask, render_template, request
import joblib, json, pandas as pd
import tensorflow as tf
import os

app = Flask(__name__)

# --- Load model ---
MODEL_PATH = r"C:\DATA\best_model.pkl"   # đổi sang .h5 nếu DL
KB_PATH = r"C:\DATA\kb_healthGuide.json"

if MODEL_PATH.endswith(".pkl"):
    model = joblib.load(MODEL_PATH)
    model_type = "sklearn"
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    model_type = "dl"

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

# --- Predict ---
def predict_health(job, age, height, weight):
    sample = pd.DataFrame([[job, age, height, weight]],
                          columns=['job','age','height_cm','weight_kg'])
    if model_type == "sklearn":
        pred = model.predict(sample)[0]
    else:  # deep learning model
        # cần encode lại như khi training
        return "normal", "⚠️ Deploy DL model cần encode giống lúc training"
    return pred, kb[pred]

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    job = age = height = weight = ""

    if request.method == "POST":
        job = request.form["job"]
        age = request.form["age"]
        height = request.form["height"]
        weight = request.form["weight"]

        if job and age and height and weight:
            pred, guide = predict_health(job, int(age), float(height), float(weight))
            result = {"pred": pred, "guide": guide}

    return render_template("index.html",
                           result=result,
                           job=job, age=age, height=height, weight=weight)

if __name__ == "__main__":
    app.run(debug=True)
