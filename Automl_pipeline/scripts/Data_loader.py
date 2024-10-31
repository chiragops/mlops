# scripts/data_loader.py

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.2, random_state=42):
    # Load the Diabetes dataset
    #data = load_diabetes(as_frame=True)
    df =  pd.read_csv('/app/data/diabetes.csv')

    # Separate features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test