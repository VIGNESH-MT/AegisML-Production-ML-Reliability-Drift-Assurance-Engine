def calculate_risk_score(psi, p_value, bias_score=0):
    """
    Combined enterprise risk model.
    """

    drift_component = min(psi * 100, 100)
    ks_component = (1 - p_value) * 100

    # Bias scaled to 0–100
    bias_component = min(abs(bias_score) * 100, 100)

    risk = (
        0.5 * drift_component +
        0.3 * ks_component +
        0.2 * bias_component
    )

    return round(max(0, min(risk, 100)), 2)