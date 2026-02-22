import numpy as np

RISK_LEVELS = ["Low","Medium","High","Critical"]
T = {"ece_low":0.05,"ece_medium":0.10,"ece_high":0.15,
     "brier_low":0.10,"brier_medium":0.17,"brier_high":0.22,
     "drift_low":0.05,"drift_medium":0.10,"drift_high":0.20,
     "accuracy_low":0.90,"accuracy_medium":0.80,"accuracy_high":0.70}

def _score(value, lo, med, hi, higher_is_worse=True):
    if higher_is_worse:
        return 0 if value<=lo else 1 if value<=med else 2 if value<=hi else 3
    else:
        return 0 if value>=lo else 1 if value>=med else 2 if value>=hi else 3

def compute_risk_score(ece, brier_score, accuracy, drift_score=None):
    scores = {
        "calibration_ece": _score(ece, T["ece_low"], T["ece_medium"], T["ece_high"]),
        "brier_score": _score(brier_score, T["brier_low"], T["brier_medium"], T["brier_high"]),
        "accuracy": _score(accuracy, T["accuracy_low"], T["accuracy_medium"], T["accuracy_high"], False),
    }
    if drift_score is not None:
        scores["distribution_drift"] = _score(drift_score, T["drift_low"], T["drift_medium"], T["drift_high"])
    weights = {"calibration_ece":0.35,"brier_score":0.25,"accuracy":0.20,"distribution_drift":0.20}
    total_w = sum(weights[k] for k in scores)
    weighted = sum(scores[k]*weights[k] for k in scores)/total_w
    overall_score = min(int(np.ceil(weighted)),3)
    risk_level = RISK_LEVELS[overall_score]
    recs = _recommendations(scores, ece, brier_score, accuracy, drift_score)
    verdict = _verdict(risk_level)
    return {"overall_risk_level": risk_level, "overall_score": overall_score,
            "weighted_score": round(weighted,3),
            "component_scores": {k:{"score":v,"level":RISK_LEVELS[v]} for k,v in scores.items()},
            "recommendations": recs, "deployment_verdict": verdict}

def _recommendations(scores, ece, brier, accuracy, drift):
    recs = []
    if scores["calibration_ece"]>=2:
        recs.append(f"⚠ Calibration is poor (ECE={ece:.4f}). Apply temperature scaling before deployment.")
    elif scores["calibration_ece"]==1:
        recs.append(f"ℹ Calibration is moderate (ECE={ece:.4f}). Monitor confidence scores in production.")
    if scores["brier_score"]>=2:
        recs.append(f"⚠ Brier score ({brier:.4f}) indicates poor probabilistic predictions.")
    if scores["accuracy"]>=2:
        recs.append(f"⚠ Accuracy ({accuracy:.2%}) is below acceptable thresholds.")
    if drift is not None:
        ds = scores.get("distribution_drift",0)
        if ds>=2: recs.append(f"🚨 Significant distribution shift (PSI={drift:.4f}). Consider retraining.")
        elif ds==1: recs.append(f"ℹ Moderate distribution shift (PSI={drift:.4f}). Set up drift alerts.")
    if not recs:
        recs.append("✅ Model meets all reliability thresholds. Safe for deployment.")
    recs.append("📋 Log all predictions, set up monitoring dashboards, schedule quarterly review.")
    return recs

def _verdict(risk_level):
    verdicts = {
        "Low": "Model demonstrates strong reliability. Deployment is recommended with standard monitoring.",
        "Medium": "Model shows moderate concerns. Deployment feasible with caution — implement confidence thresholding and drift alerts.",
        "High": "Model exhibits significant reliability concerns. Deployment NOT recommended without remediation.",
        "Critical": "CRITICAL RISK — Model should NOT be deployed. Multiple reliability signals are severely degraded.",
    }
    return verdicts[risk_level]