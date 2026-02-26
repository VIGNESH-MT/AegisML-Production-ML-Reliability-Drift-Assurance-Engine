def map_risk_tier(score):
    if score < 30:
        return "LOW", "Routine Monitoring"
    elif score < 70:
        return "MEDIUM", "Engineering Review Required"
    else:
        return "HIGH", "Immediate Escalation Required"