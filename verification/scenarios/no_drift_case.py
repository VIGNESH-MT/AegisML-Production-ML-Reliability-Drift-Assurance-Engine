import numpy as np

def generate_no_drift_data():
    np.random.seed(42)
    train = np.random.normal(0, 1, 10000)
    production = np.random.normal(0, 1, 10000)
    return train, production