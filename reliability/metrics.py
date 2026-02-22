import numpy as np
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss, confusion_matrix

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def format_confusion_matrix(cm):
    cm = np.array(cm)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    return (f"                Predicted 0   Predicted 1\n"
            f"  Actual 0          {tn:<6}        {fp:<6}\n"
            f"  Actual 1          {fn:<6}        {tp:<6}")