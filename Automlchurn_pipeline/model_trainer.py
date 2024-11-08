# model_trainer.py
import h2o
from h2o.automl import H2OAutoML
from data_loader import load_and_preprocess_data
import os

#def train_model():
    #train, _, features, target = load_and_preprocess_data()
    #aml = H2OAutoML(max_models=20, seed=1)
    #aml.train(x=features, y=target, training_frame=train)
    #return aml.leader

def train_and_save_model():
    # Load data
    train, test, features, target = load_and_preprocess_data()

    # Train model using H2O AutoML
    aml = H2OAutoML(max_models=5, seed=1234)
    aml.train(x=features, y=target, training_frame=train)

    # Save the best model locally in /models
    model_dir = "models"  # Local path to save models
    os.makedirs(model_dir, exist_ok=True)
    best_model = aml.leader
    model_path = h2o.save_model(model=best_model, path=model_dir, force=True)
    final_model_path = os.path.join(model_dir, "trained_model.zip")

# Rename to a consistent filename
    os.rename(model_path, final_model_path)
    print(f"Best model saved at: {final_model_path}")
    #print(f"Best model saved at: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
