import h2o
from data_loader import load_and_preprocess_data

#def evaluate_model(model):
    #_, test, features, target = load_and_preprocess_data()
    #performance = model.model_performance(test_data=test)
    #return performance

def evaluate_saved_model(model_path="/app/models/trained_model.zip"):
    # Initialize H2O and load data
    h2o.init()
    _, test, features, target = load_and_preprocess_data()

    # Load the saved model
    model = h2o.load_model(model_path)

    # Evaluate model performance on test data
    performance = model.model_performance(test_data=test)
    print("Model Performance:\n", performance)

if __name__ == "__main__":
    evaluate_saved_model()
