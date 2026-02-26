import numpy as np

def generate_bias_data():
    np.random.seed(42)

    # Protected attribute (0 and 1 groups)
    protected_attr = np.random.binomial(1, 0.5, 10000)

    # Predictions intentionally biased
    y_pred = np.where(protected_attr == 1,
                      np.random.binomial(1, 0.7, 10000),
                      np.random.binomial(1, 0.3, 10000))

    return y_pred, protected_attr