# scripts/main.py

import os
import joblib
from Data_loader import load_and_split_data
from model_trainer import train_model
from model_eval import evaluate_model

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    print("Model training and evaluation completed. Run `python app/scripts/model_deploy.py` to start the API server.")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "/app/models/diabetes_regression_model.pkl")

if __name__ == "__main__":
    main()

    #while True:
        #pass