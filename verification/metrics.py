import numpy as np
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, bins=10):
    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents) *
        np.log((expected_percents + 1e-8) / (actual_percents + 1e-8))
    )

    return psi


def calculate_ks(expected, actual):
    statistic, p_value = ks_2samp(expected, actual)
    return statistic, p_value