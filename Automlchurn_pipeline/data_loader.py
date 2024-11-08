import h2o

def load_and_preprocess_data(data_path="/app/data/churn_data.csv"):
    h2o.init()
    data = h2o.import_file(data_path)
    target = "Churn"
    features = [col for col in data.columns if col != target and col != 'customerID']
    data[target] = data[target].asfactor()
    train, test = data.split_frame(ratios=[0.8])
    return train, test, features, target
