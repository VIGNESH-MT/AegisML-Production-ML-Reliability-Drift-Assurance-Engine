import numpy as np

def generate_mild_drift_data():
    np.random.seed(42)
    train = np.random.normal(0, 1, 10000)
    production = np.random.normal(0.5, 1, 10000)
    return train, production