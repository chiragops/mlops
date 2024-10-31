import joblib
from flask import Flask, request, jsonify,render_template
from prometheus_client import Counter, generate_latest
import os

request_count = Counter("request_count", "Number of requests received")
error_count = Counter("error_count", "Number of errors")

def increment_request_count():
    request_count.inc()

def increment_error_count():
    error_count.inc()

app = Flask(__name__, template_folder='/app/templates')

model_path = "/app/models/diabetes_regression_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Make sure main.py creates it in the expected location.")

# Load the trained model
with open(model_path, "rb") as f:
    model = joblib.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Get the nine input features from the form
        try:
            input_data = [
                float(request.form["Pregnancies"]),
                float(request.form["Glucose"]),
                float(request.form["BloodPressure"]),
                float(request.form["SkinThickness"]),
                float(request.form["Insulin"]),
                float(request.form["BMI"]),
                float(request.form["DiabetesPedigreeFunction"]),
                float(request.form["Age"]),
               
            ]
            
            # Predict using the model
            prediction = model.predict([input_data])[0]
        except ValueError:
            prediction = "Invalid input. Please enter numeric values only."

    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict():
    increment_request_count()
    try:
        data = request.get_json()
        input_data = data["input_data"]
        prediction = model.predict([input_data])[0]
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        increment_error_count()
        return jsonify({"error": str(e)}), 400

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)