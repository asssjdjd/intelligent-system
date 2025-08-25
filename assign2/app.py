from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # Lấy dữ liệu từ form
        glucose = float(request.form["glucose"])
        bmi = float(request.form["bmi"])
        age = float(request.form["age"])
        bp = float(request.form["bp"])
        insulin = float(request.form["insulin"])

        # Chuyển thành numpy array
        features = np.array([[glucose, bmi, age, bp, insulin]])

        # Dự đoán
        prediction = model.predict(features)[0]
        result = "Có khả năng bị tiểu đường" if prediction == 1 else "Không bị tiểu đường"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
