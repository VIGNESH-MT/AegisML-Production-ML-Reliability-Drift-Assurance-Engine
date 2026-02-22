"""
test_core.py
------------
Unit tests for the ML Reliability Auditor core modules.

Run with:  python -m pytest test_core.py -v
       or: python test_core.py
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from reliability.metrics import compute_metrics, format_confusion_matrix
from reliability.calibration import compute_ece, reliability_diagram, confidence_histogram
from reliability.drift import (
    compute_prediction_drift,
    compute_feature_drift,
    drift_heatmap,
)
from reliability.risk import compute_risk_score
from reliability.report import generate_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perfect_preds(n=200):
    """Perfect predictions — high accuracy, well calibrated."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    # Near-perfect probs
    y_prob = np.where(y_true == 1, np.random.uniform(0.85, 0.99, n), np.random.uniform(0.01, 0.15, n))
    return y_true, y_prob


def _random_preds(n=200):
    """Random predictions."""
    np.random.seed(7)
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.uniform(0, 1, n)
    return y_true, y_prob


def _overconfident_preds(n=200):
    """Model that is systematically overconfident."""
    np.random.seed(13)
    y_true = np.random.randint(0, 2, n)
    y_prob = np.where(y_true == 1,
                      np.random.uniform(0.75, 0.98, n),
                      np.random.uniform(0.35, 0.65, n))  # Should be low, but isn't
    return y_true, y_prob


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    y_true, y_prob = _perfect_preds()
    result = compute_metrics(y_true, y_prob)
    assert result["accuracy"] > 0.90, f"Expected accuracy > 0.90, got {result['accuracy']}"
    assert result["f1"] > 0.85
    assert result["brier_score"] < 0.10
    assert len(result["confusion_matrix"]) == 2
    print(f"  [PASS] Perfect preds — accuracy={result['accuracy']:.3f}, brier={result['brier_score']:.4f}")


def test_compute_metrics_random():
    y_true, y_prob = _random_preds()
    result = compute_metrics(y_true, y_prob)
    assert 0.3 < result["accuracy"] < 0.7
    assert result["brier_score"] > 0.15
    print(f"  [PASS] Random preds — accuracy={result['accuracy']:.3f}, brier={result['brier_score']:.4f}")


def test_format_confusion_matrix():
    cm = [[90, 10], [5, 95]]
    s = format_confusion_matrix(cm)
    assert "90" in s and "95" in s
    print("  [PASS] Confusion matrix formatting")


# ---------------------------------------------------------------------------
# calibration.py tests
# ---------------------------------------------------------------------------

def test_ece_perfect():
    y_true, y_prob = _perfect_preds()
    result = compute_ece(y_true, y_prob)
    assert result["ece"] < 0.12, f"Expected low ECE for near-perfect model, got {result['ece']}"
    assert result["mce"] < 0.20
    assert len(result["bins"]) == 10
    print(f"  [PASS] ECE perfect — ece={result['ece']:.4f}")


def test_ece_overconfident():
    y_true, y_prob = _overconfident_preds()
    result = compute_ece(y_true, y_prob)
    # Overconfident model should have moderate/high ECE
    assert result["overconfidence_gap"] > 0, "Expected positive overconfidence gap"
    print(f"  [PASS] ECE overconfident — ece={result['ece']:.4f}, gap={result['overconfidence_gap']:.4f}")


def test_reliability_diagram_bytes():
    y_true, y_prob = _perfect_preds()
    img_bytes = reliability_diagram(y_true, y_prob, return_bytes=True)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 1000  # Should be a real image
    print(f"  [PASS] Reliability diagram bytes — {len(img_bytes):,} bytes")


def test_confidence_histogram_bytes():
    _, y_prob = _perfect_preds()
    img_bytes = confidence_histogram(y_prob, return_bytes=True)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 1000
    print(f"  [PASS] Confidence histogram bytes — {len(img_bytes):,} bytes")


# ---------------------------------------------------------------------------
# drift.py tests
# ---------------------------------------------------------------------------

def test_prediction_drift_no_drift():
    np.random.seed(1)
    probs = np.random.beta(2, 2, 200)
    result = compute_prediction_drift(probs, probs + np.random.normal(0, 0.01, 200).clip(-0.05, 0.05))
    assert result["psi"] < 0.10, f"Minimal drift expected, got PSI={result['psi']}"
    print(f"  [PASS] Prediction drift (no shift) — psi={result['psi']:.4f}, kl={result['kl_divergence']:.4f}")


def test_prediction_drift_significant():
    np.random.seed(2)
    ref_probs = np.random.beta(2, 5, 200)    # Mostly low probs
    prod_probs = np.random.beta(5, 2, 200)   # Mostly high probs (big shift)
    result = compute_prediction_drift(ref_probs, prod_probs)
    assert result["psi"] > 0.1, f"Expected significant drift, got PSI={result['psi']}"
    print(f"  [PASS] Prediction drift (significant) — psi={result['psi']:.4f}, kl={result['kl_divergence']:.4f}")


def test_feature_drift():
    np.random.seed(3)
    ref_df = pd.DataFrame({
        "f1": np.random.normal(0, 1, 100),
        "f2": np.random.normal(5, 2, 100),
        "f3": np.random.normal(-1, 0.5, 100),
    })
    # Significant shift in f1, moderate in f2
    prod_df = pd.DataFrame({
        "f1": np.random.normal(3, 1.5, 100),   # Big shift
        "f2": np.random.normal(5.5, 2, 100),    # Small shift
        "f3": np.random.normal(-1, 0.5, 100),   # No shift
    })
    result = compute_feature_drift(ref_df, prod_df)
    assert result["n_features"] == 3
    assert "f1" in result["features"]
    assert result["features"]["f1"]["psi"] > result["features"]["f3"]["psi"]
    print(f"  [PASS] Feature drift — f1_psi={result['features']['f1']['psi']:.4f}, "
          f"f3_psi={result['features']['f3']['psi']:.4f}")


def test_drift_heatmap_bytes():
    np.random.seed(4)
    ref_df = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
    prod_df = pd.DataFrame({"a": np.random.normal(2, 1, 100), "b": np.random.normal(0.1, 1, 100)})
    drift = compute_feature_drift(ref_df, prod_df)
    img_bytes = drift_heatmap(drift, return_bytes=True)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 500
    print(f"  [PASS] Drift heatmap bytes — {len(img_bytes):,} bytes")


# ---------------------------------------------------------------------------
# risk.py tests
# ---------------------------------------------------------------------------

def test_risk_low():
    risk = compute_risk_score(ece=0.02, brier_score=0.05, accuracy=0.95, drift_score=0.03)
    assert risk["overall_risk_level"] == "Low"
    assert risk["overall_score"] == 0
    print(f"  [PASS] Risk low — level={risk['overall_risk_level']}, score={risk['overall_score']}")


def test_risk_high():
    risk = compute_risk_score(ece=0.18, brier_score=0.25, accuracy=0.62, drift_score=0.35)
    assert risk["overall_risk_level"] in ("High", "Critical")
    assert risk["overall_score"] >= 2
    print(f"  [PASS] Risk high — level={risk['overall_risk_level']}, score={risk['overall_score']}")


def test_risk_no_drift():
    risk = compute_risk_score(ece=0.08, brier_score=0.14, accuracy=0.84, drift_score=None)
    assert risk["overall_risk_level"] in ("Low", "Medium")
    assert isinstance(risk["recommendations"], list)
    assert len(risk["recommendations"]) >= 1
    print(f"  [PASS] Risk no drift — level={risk['overall_risk_level']}, recs={len(risk['recommendations'])}")


def test_risk_verdict_present():
    risk = compute_risk_score(ece=0.05, brier_score=0.10, accuracy=0.88)
    assert isinstance(risk["deployment_verdict"], str)
    assert len(risk["deployment_verdict"]) > 10
    print("  [PASS] Risk verdict present")


# ---------------------------------------------------------------------------
# report.py tests
# ---------------------------------------------------------------------------

def test_generate_report():
    y_true, y_prob = _perfect_preds(300)
    metrics = compute_metrics(y_true, y_prob)
    calibration = compute_ece(y_true, y_prob)
    risk = compute_risk_score(
        ece=calibration["ece"],
        brier_score=metrics["brier_score"],
        accuracy=metrics["accuracy"],
        drift_score=0.07,
    )

    rel_bytes = reliability_diagram(y_true, y_prob, return_bytes=True)
    hist_bytes = confidence_histogram(y_prob, return_bytes=True)

    np.random.seed(5)
    ref_df = pd.DataFrame({f"feat{i}": np.random.normal(i, 1, 100) for i in range(4)})
    prod_df = pd.DataFrame({f"feat{i}": np.random.normal(i + 0.5, 1.2, 100) for i in range(4)})
    drift = compute_feature_drift(ref_df, prod_df)
    pred_drift = compute_prediction_drift(y_prob, y_prob + np.random.normal(0, 0.05, len(y_prob)))
    drift_chart = drift_heatmap(drift, return_bytes=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        out_path = f.name

    result_path = generate_report(
        output_path=out_path,
        model_name="TestModel v1.0",
        metrics=metrics,
        calibration=calibration,
        risk=risk,
        drift=drift,
        prediction_drift=pred_drift,
        reliability_diagram_bytes=rel_bytes,
        confidence_hist_bytes=hist_bytes,
        drift_chart_bytes=drift_chart,
        n_samples=len(y_true),
    )

    assert os.path.exists(result_path)
    file_size = os.path.getsize(result_path)
    assert file_size > 50_000, f"PDF too small: {file_size} bytes"
    print(f"  [PASS] PDF report generated — {file_size:,} bytes at {result_path}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # metrics
        ("compute_metrics (perfect)", test_compute_metrics_perfect),
        ("compute_metrics (random)", test_compute_metrics_random),
        ("format_confusion_matrix", test_format_confusion_matrix),
        # calibration
        ("ECE (perfect)", test_ece_perfect),
        ("ECE (overconfident)", test_ece_overconfident),
        ("reliability_diagram bytes", test_reliability_diagram_bytes),
        ("confidence_histogram bytes", test_confidence_histogram_bytes),
        # drift
        ("prediction_drift (no drift)", test_prediction_drift_no_drift),
        ("prediction_drift (significant)", test_prediction_drift_significant),
        ("feature_drift", test_feature_drift),
        ("drift_heatmap bytes", test_drift_heatmap_bytes),
        # risk
        ("risk (low)", test_risk_low),
        ("risk (high)", test_risk_high),
        ("risk (no drift)", test_risk_no_drift),
        ("risk verdict", test_risk_verdict_present),
        # report
        ("generate_report PDF", test_generate_report),
    ]

    passed = 0
    failed = 0
    print("\n" + "=" * 60)
    print("  ML RELIABILITY AUDITOR — TEST SUITE")
    print("=" * 60)

    for name, test_fn in tests:
        print(f"\n▶ {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()