import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

def _psi_single(ref, prod, n_bins=10):
    ref = np.asarray(ref, dtype=float); prod = np.asarray(prod, dtype=float)
    quantiles = np.unique(np.percentile(ref, np.linspace(0,100,n_bins+1)))
    if len(quantiles) < 2: return 0.0
    eps = 1e-8
    ref_pct = np.histogram(ref,bins=quantiles)[0] / (len(ref)+eps) + eps
    prod_pct = np.histogram(prod,bins=quantiles)[0] / (len(prod)+eps) + eps
    return round(max(float(np.sum((prod_pct-ref_pct)*np.log(prod_pct/ref_pct))),0.0),4)

def _kl_divergence(ref_probs, prod_probs, n_bins=20):
    bins = np.linspace(0,1,n_bins+1); eps = 1e-8
    ref_dist = np.histogram(ref_probs,bins=bins)[0]/(len(ref_probs)+eps)+eps
    prod_dist = np.histogram(prod_probs,bins=bins)[0]/(len(prod_probs)+eps)+eps
    return round(float(entropy(prod_dist,ref_dist)),4)

def compute_prediction_drift(ref_probs, prod_probs):
    kl = _kl_divergence(ref_probs, prod_probs)
    psi = _psi_single(ref_probs, prod_probs)
    return {"kl_divergence": kl, "psi": psi,
            "kl_interpretation": "Negligible" if kl<0.05 else "Moderate" if kl<0.2 else "High divergence",
            "psi_interpretation": _interpret_psi(psi)}

def compute_feature_drift(ref_df, prod_df):
    ref_df = ref_df.select_dtypes(include=[np.number])
    prod_df = prod_df.select_dtypes(include=[np.number])
    common_cols = [c for c in ref_df.columns if c in prod_df.columns]
    if not common_cols: return {"features":{}, "overall_drift_score":0.0, "n_features":0}
    feature_stats = {}; psi_values = []
    for col in common_cols:
        ref_col = ref_df[col].dropna().values; prod_col = prod_df[col].dropna().values
        if len(ref_col)==0 or len(prod_col)==0: continue
        psi = _psi_single(ref_col, prod_col)
        mean_shift = float(prod_col.mean()-ref_col.mean())
        feature_stats[col] = {
            "psi": psi, "mean_ref": round(float(ref_col.mean()),4),
            "mean_prod": round(float(prod_col.mean()),4), "mean_shift": round(mean_shift,4),
            "std_ref": round(float(ref_col.std()),4), "std_prod": round(float(prod_col.std()),4),
            "std_shift": round(float(prod_col.std()-ref_col.std()),4),
            "relative_mean_shift_pct": round(abs(mean_shift)/(abs(ref_col.mean())+1e-8)*100,2),
            "psi_interpretation": _interpret_psi(psi)}
        psi_values.append(psi)
    overall = round(float(np.mean(psi_values)) if psi_values else 0.0,4)
    return {"features": feature_stats, "overall_drift_score": overall,
            "overall_interpretation": _interpret_psi(overall), "n_features": len(feature_stats)}

def drift_heatmap(feature_drift, title="Feature Drift (PSI)", return_bytes=False):
    features = feature_drift.get("features",{})
    if not features: return None
    names = list(features.keys()); psi_vals = [features[f]["psi"] for f in names]
    colors = ["#2ecc71" if p<0.1 else "#f39c12" if p<0.2 else "#e74c3c" for p in psi_vals]
    fig, ax = plt.subplots(figsize=(max(7,len(names)*0.9),5))
    ax.set_facecolor("#f9f9f9")
    bars = ax.bar(names, psi_vals, color=colors, edgecolor="white", alpha=0.9)
    ax.axhline(y=0.1,color="#f39c12",linestyle="--",linewidth=1.2,label="Moderate (0.1)")
    ax.axhline(y=0.2,color="#e74c3c",linestyle="--",linewidth=1.2,label="Significant (0.2)")
    for bar,val in zip(bars,psi_vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f"{val:.3f}",ha="center",fontsize=9)
    ax.set_xlabel("Feature",fontsize=12); ax.set_ylabel("PSI",fontsize=12)
    ax.set_title(title,fontsize=13); ax.legend(fontsize=9)
    ax.grid(True,axis="y",linestyle="--",alpha=0.4); plt.xticks(rotation=30,ha="right"); plt.tight_layout()
    if return_bytes:
        buf = BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight")
        plt.close(fig); buf.seek(0); return buf.read()
    plt.show(); plt.close(fig)

def _interpret_psi(psi):
    if psi<0.1: return "No significant shift"
    elif psi<0.2: return "Moderate shift — monitor closely"
    else: return "Significant shift — investigate immediately"