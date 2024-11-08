# predict_model.py

import h2o
import pandas as pd

def load_and_predict(model_path, input_data):
    h2o.init()
    model = h2o.load_model(model_path)
    h2o_data = h2o.H2OFrame(input_data)
    prediction = model.predict(h2o_data).as_data_frame().iloc[0, 0]
    return "Yes" if prediction == "Yes" else "No"

if __name__ == "__main__":
    sample_data = {
        'gender': ["Male"],
        'SeniorCitizen': [0],
        'Partner': ["No"],
        'Dependents': ["No"],
        'tenure': [34],
        'PhoneService': ["Yes"],
        'MultipleLines': ["No"],
        'InternetService': ["DSL"],
        'OnlineSecurity': ["Yes"],
        'OnlineBackup': ["No"],
        'DeviceProtection': ["Yes"],
        'TechSupport': ["No"],
        'StreamingTV': ["No"],
        'StreamingMovies': ["No"],
        'Contract': ["One year"],
        'PaperlessBilling': ["No"],
        'PaymentMethod': ["Mailed check"],
        'MonthlyCharges': [56.95],
        'TotalCharges': [1889.5]
    }

    input_data = pd.DataFrame(sample_data)
    result = load_and_predict(input_data=input_data)
    print(f"Customer Churn Prediction: {result}")