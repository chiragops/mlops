# scripts/model_trainer.py

from tpot import TPOTRegressor

def train_model(X_train, y_train, generations=5, population_size=50, random_state=42):
    # Initialize TPOT with settings
    tpot = TPOTRegressor(generations=generations, population_size=population_size, verbosity=2, random_state=random_state, n_jobs=-1)
    
    # Train the model
    tpot.fit(X_train, y_train)
    
    # Export the pipeline as a Python script
    tpot.export('scripts/best_model_pipeline.py')
    
    return tpot.fitted_pipeline_  # Return the fitted pipeline