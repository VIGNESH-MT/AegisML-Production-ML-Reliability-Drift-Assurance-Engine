import numpy as np

def generate_small_sample_data():
    np.random.seed(42)
    train = np.random.normal(0, 1, 20)
    production = np.random.normal(0, 1, 5)
    return train, production