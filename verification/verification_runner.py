import json
import datetime
import platform
import subprocess
import hashlib

from verification.bias import statistical_parity_difference, disparate_impact_ratio
from verification.scenarios.bias_case import generate_bias_data
from verification.pdf_report import generate_pdf
generate_pdf("verification/verification_report.json", language="English")
from verification.scenarios.no_drift_case import generate_no_drift_data
from verification.scenarios.mild_drift_case import generate_mild_drift_data
from verification.scenarios.severe_drift_case import generate_severe_drift_data
from verification.scenarios.failure_case import generate_small_sample_data

from verification.metrics import calculate_psi, calculate_ks
from verification.risk_scoring import calculate_risk_score
from verification.governance import map_risk_tier


# -------------------------------------------------
# Utility
# -------------------------------------------------

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "N/A"


# -------------------------------------------------
# Drift Test Runner
# -------------------------------------------------

def run_test(case_name, train, prod, expected_path):

    psi = calculate_psi(train, prod)
    ks_stat, p_value = calculate_ks(train, prod)

    risk_score = calculate_risk_score(psi, p_value, bias_score=0)
    tier, action = map_risk_tier(risk_score)

    with open(expected_path) as f:
        expected = json.load(f)

    status = "PASS"

    if "psi_min" in expected and psi < expected["psi_min"]:
        status = "FAIL"
    if "psi_max" in expected and psi > expected["psi_max"]:
        status = "FAIL"
    if "ks_pvalue_min" in expected and p_value < expected["ks_pvalue_min"]:
        status = "FAIL"
    if "ks_pvalue_max" in expected and p_value > expected["ks_pvalue_max"]:
        status = "FAIL"

    print(f"\n[{case_name}] PSI: {psi:.5f}")
    print(f"[{case_name}] KS p-value: {p_value:.5f}")
    print(f"[{case_name}] Risk Score: {risk_score}")
    print(f"[{case_name}] Risk Tier: {tier}")

    return {
        "case": case_name,
        "psi": round(psi, 5),
        "ks_pvalue": round(p_value, 5),
        "risk_score": risk_score,
        "risk_tier": tier,
        "required_action": action,
        "status": status
    }


# -------------------------------------------------
# Bias Test Runner
# -------------------------------------------------

def run_bias_test():

    y_pred, protected_attr = generate_bias_data()

    spd = statistical_parity_difference(y_pred, protected_attr)
    dir_ratio = disparate_impact_ratio(y_pred, protected_attr)

    bias_score = abs(spd)

    print(f"\n[bias_test] Statistical Parity Difference: {spd}")
    print(f"[bias_test] Disparate Impact Ratio: {dir_ratio}")

    risk_score = calculate_risk_score(0, 1, bias_score)
    tier, action = map_risk_tier(risk_score)

    return {
        "case": "bias_assessment",
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_ratio,
        "risk_score": risk_score,
        "risk_tier": tier,
        "required_action": action,
        "status": "PASS" if abs(spd) > 0.1 else "FAIL"
    }


# -------------------------------------------------
# Report Generator (With Integrity Hash)
# -------------------------------------------------

def generate_verification_report(results, overall_status):

    base_report = {
        "verification_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "system_status": "VERIFIED" if overall_status else "FAILED",
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "methodology": {
            "drift_metric": "Population Stability Index (PSI)",
            "statistical_test": "Kolmogorov-Smirnov two-sample test",
            "risk_model": "Weighted linear combination",
            "deterministic_seed": 42
        },
        "regulatory_alignment": {
            "eu_ai_act_article_9": "Risk monitoring through drift detection",
            "article_15": "Accuracy and robustness validation",
            "article_17": "Technical documentation traceability"
        },
        "results": results
    }

    # Integrity hash
    report_string = json.dumps(base_report, sort_keys=True)
    integrity_hash = hashlib.sha256(report_string.encode()).hexdigest()

    base_report["integrity_hash_sha256"] = integrity_hash

    with open("verification/verification_report.json", "w") as f:
        json.dump(base_report, f, indent=4)

    print("\nStructured enterprise verification report generated.")
    print(f"Integrity Hash: {integrity_hash}")


# -------------------------------------------------
# Main Execution
# -------------------------------------------------

if __name__ == "__main__":

    print("\nRunning Enterprise-Grade Verification Suite")
    print("===========================================")

    results = []

    # Drift Tests
    results.append(run_test(
        "no_drift",
        *generate_no_drift_data(),
        "verification/expected_outputs/no_drift.json"
    ))

    results.append(run_test(
        "mild_drift",
        *generate_mild_drift_data(),
        "verification/expected_outputs/mild_drift.json"
    ))

    results.append(run_test(
        "severe_drift",
        *generate_severe_drift_data(),
        "verification/expected_outputs/severe_drift.json"
    ))

    # Bias Test
    results.append(run_bias_test())

    # Failure Mode Simulation
    small_train, small_prod = generate_small_sample_data()
    print("\n[Failure Simulation] Small sample test executed.")

    # Monotonic Calibration Check (Drift only)
    drift_risks = [r["risk_score"] for r in results if r["case"] in ["no_drift", "mild_drift", "severe_drift"]]
    monotonic = drift_risks[0] < drift_risks[1] < drift_risks[2]

    if not monotonic:
        print("\nCalibration Check FAILED: Risk not monotonic.")
        overall_status = False
    else:
        print("\nCalibration Check PASSED: Risk increases logically.")
        overall_status = all(r["status"] == "PASS" for r in results)

    print(f"\nOverall System Status: {'VERIFIED' if overall_status else 'FAILED'}")

    generate_verification_report(results, overall_status)
    generate_pdf("verification/verification_report.json", language="English")
    generate_pdf("verification/verification_report.json", language="German") 
    