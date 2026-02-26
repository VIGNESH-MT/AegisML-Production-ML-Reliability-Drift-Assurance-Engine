def calculate_risk_score(psi, p_value):
    """
    Enterprise-style calibrated risk scoring.

    psi: Population Stability Index
    p_value: KS test p-value

    Returns:
        Risk score between 0–100
    """

    # Drift impact (scaled)
    drift_component = min(psi * 100, 100)

    # KS impact (lower p-value = higher risk)
    ks_component = (1 - p_value) * 100

    # Weighted combination
    risk = 0.6 * drift_component + 0.4 * ks_component

    # Clamp between 0–100
    risk = max(0, min(risk, 100))

    return round(risk, 2)