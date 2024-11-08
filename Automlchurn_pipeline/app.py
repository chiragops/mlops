# app.py

from flask import Flask, request, render_template,jsonify
import pandas as pd
from model_pred import load_and_predict  # Modular function to load and predict
#from model_trainer import train_model
#from model_save import save_model
import os
import h2o

app = Flask(__name__,template_folder='/app/templates')
h2o.init()

# Model path inside the container
MODEL_PATH = "/app/models/trained_model.zip"
#model = train_model()
#model_path = save_model(model)
@app.route('/')
def index():
    # Render the HTML form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data and store it in a dictionary
    form_data = {
        'gender': request.form.get('gender'),
        'SeniorCitizen': int(request.form.get('SeniorCitizen')),
        'Partner': request.form.get('Partner'),
        'Dependents': request.form.get('Dependents'),
        'tenure': int(request.form.get('tenure')),
        'PhoneService': request.form.get('PhoneService'),
        'MultipleLines': request.form.get('MultipleLines'),
        'InternetService': request.form.get('InternetService'),
        'OnlineSecurity': request.form.get('OnlineSecurity'),
        'OnlineBackup': request.form.get('OnlineBackup'),
        'DeviceProtection': request.form.get('DeviceProtection'),
        'TechSupport': request.form.get('TechSupport'),
        'StreamingTV': request.form.get('StreamingTV'),
        'StreamingMovies': request.form.get('StreamingMovies'),
        'Contract': request.form.get('Contract'),
        'PaperlessBilling': request.form.get('PaperlessBilling'),
        'PaymentMethod': request.form.get('PaymentMethod'),
        'MonthlyCharges': float(request.form.get('MonthlyCharges')),
        'TotalCharges': float(request.form.get('TotalCharges')),
    }

    # Convert the form data to a DataFrame
    input_data = pd.DataFrame([form_data])

    # Use the prediction function to get the result
    result = load_and_predict(model_path=MODEL_PATH, input_data=input_data)
    #return render_template('index.html', result=f"Customer Churn Prediction: {result}")
    return jsonify({"Prediction": result})
if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
