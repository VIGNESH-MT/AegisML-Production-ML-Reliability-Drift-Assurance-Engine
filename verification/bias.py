import numpy as np

def statistical_parity_difference(y_pred, protected_attr):
    """
    Computes Statistical Parity Difference.
    SPD = P(Y=1 | A=1) - P(Y=1 | A=0)
    """

    group_1 = y_pred[protected_attr == 1]
    group_0 = y_pred[protected_attr == 0]

    if len(group_1) == 0 or len(group_0) == 0:
        return 0.0

    rate_1 = np.mean(group_1)
    rate_0 = np.mean(group_0)

    return round(rate_1 - rate_0, 5)


def disparate_impact_ratio(y_pred, protected_attr):
    """
    Computes Disparate Impact Ratio.
    DIR = P(Y=1 | A=1) / P(Y=1 | A=0)
    """

    group_1 = y_pred[protected_attr == 1]
    group_0 = y_pred[protected_attr == 0]

    if len(group_1) == 0 or len(group_0) == 0:
        return 1.0

    rate_1 = np.mean(group_1)
    rate_0 = np.mean(group_0)

    if rate_0 == 0:
        return 1.0

    return round(rate_1 / rate_0, 5)