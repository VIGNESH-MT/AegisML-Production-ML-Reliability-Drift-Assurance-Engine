"""
app.py — ML Reliability & Distribution Shift Auditor
Elite UI Redesign: Tier-1 Enterprise AI Platform (Obsidian & Cyan)
"""

import io, os, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from reliability.metrics import compute_metrics, format_confusion_matrix
from reliability.calibration import compute_ece, reliability_diagram, confidence_histogram
from reliability.drift import compute_prediction_drift, compute_feature_drift, drift_heatmap
from reliability.risk import compute_risk_score
from reliability.report import generate_report

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Auditor | Elite",
    page_icon="❖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# TIER-1 ENTERPRISE CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════════════
   ROOT VARIABLES (Obsidian & Neon)
══════════════════════════════════════════════════ */
:root {
  --bg-base:        #050507;
  --bg-glass:       rgba(13, 14, 18, 0.6);
  --bg-glass-hover: rgba(20, 22, 28, 0.8);
  --border-light:   rgba(255, 255, 255, 0.06);
  --border-glow:    rgba(0, 240, 255, 0.3);
  
  --text-main:      #FFFFFF;
  --text-muted:     #A0AEC0;
  --text-dim:       #4A5568;
  
  --accent-cyan:    #00F0FF;
  --accent-violet:  #8A2BE2;
  
  --risk-low:       #10B981;
  --risk-med:       #F59E0B;
  --risk-high:      #EF4444;
  --risk-crit:      #E11D48;
}

/* ══════════════════════════════════════════════════
   GLOBAL RESET & BACKGROUND EFFECTS
══════════════════════════════════════════════════ */
.stApp {
  background-color: var(--bg-base) !important;
  background-image: 
    radial-gradient(circle at 15% 50%, rgba(138, 43, 226, 0.04) 0%, transparent 50%),
    radial-gradient(circle at 85% 30%, rgba(0, 240, 255, 0.04) 0%, transparent 50%);
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text-main) !important;
}

/* ══════════════════════════════════════════════════
   SIDEBAR (Sleek Panel)
══════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: rgba(8, 9, 12, 0.95) !important;
  backdrop-filter: blur(20px) !important;
  border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
  color: var(--text-muted) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stFileUploader"] {
  background: rgba(255, 255, 255, 0.02) !important;
  border: 1px solid var(--border-light) !important;
  border-radius: 8px !important;
  color: var(--text-main) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  transition: all 0.3s ease;
}

[data-testid="stSidebar"] input:focus,
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 12px rgba(0, 240, 255, 0.1) !important;
}

/* ══════════════════════════════════════════════════
   MAIN CONTENT & TYPOGRAPHY
══════════════════════════════════════════════════ */
.main .block-container {
  padding-top: 2rem !important;
  max-width: 1400px;
}

h1, h2, h3, h4 {
  font-family: 'Outfit', sans-serif !important;
  color: var(--text-main) !important;
}

/* ══════════════════════════════════════════════════
   PREMIUM GLASSMORPHISM CARDS
══════════════════════════════════════════════════ */
.elite-card {
  background: var(--bg-glass);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-light);
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
  transition: transform 0.3s ease, background 0.3s ease;
}
.elite-card:hover {
  background: var(--bg-glass-hover);
  border-color: rgba(255, 255, 255, 0.1);
}

/* ══════════════════════════════════════════════════
   BUTTONS (Call to Action)
══════════════════════════════════════════════════ */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00F0FF 0%, #0080FF 100%) !important;
  color: #000000 !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 28px !important;
  box-shadow: 0 4px 14px 0 rgba(0, 240, 255, 0.3) !important;
  transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px 0 rgba(0, 240, 255, 0.5) !important;
}

/* ══════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════ */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
  margin: 3rem 0 !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM HTML COMPONENTS (The Elite Look)
# ─────────────────────────────────────────────────────────────────────────────

def hero_header():
    st.markdown("""
    <div style="padding: 3rem 0 3rem; position: relative;">
      <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #00F0FF;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        text-transform: uppercase;
      ">
        <span style="display:inline-block; width:30px; height:1px; background:#00F0FF;"></span>
        PLATFORM INTELLIGENCE
      </div>
      <h1 style="
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin: 0 0 1rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #A0AEC0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      ">Model Reliability<br/>& Distribution Auditor</h1>
      <p style="
        font-size: 1.1rem;
        color: #A0AEC0;
        margin: 0;
        max-width: 650px;
        line-height: 1.6;
        font-weight: 400;
      ">
        Evaluate calibration fidelity, detect covariate shift, and generate
        institutional-grade risk assessments with sub-millisecond precision.
      </p>
    </div>
    """, unsafe_allow_html=True)

def risk_badge(risk_level):
    colors = {
        "Low":      ("#10B981", "rgba(16, 185, 129, 0.05)", "rgba(16, 185, 129, 0.2)"),
        "Medium":   ("#F59E0B", "rgba(245, 158, 11, 0.05)", "rgba(245, 158, 11, 0.2)"),
        "High":     ("#EF4444", "rgba(239, 68, 68, 0.05)", "rgba(239, 68, 68, 0.2)"),
        "Critical": ("#E11D48", "rgba(225, 29, 72, 0.05)", "rgba(225, 29, 72, 0.3)"),
    }
    verdicts = {
        "Low":      "PRODUCTION DEPLOYMENT AUTHORIZED",
        "Medium":   "DEPLOYMENT APPROVED WITH OBSERVATION",
        "High":     "DEPLOYMENT BLOCKED: ELEVATED RISK",
        "Critical": "CRITICAL SYSTEM RISK DETECTED",
    }
    c, bg, border = colors.get(risk_level, colors["High"])
    verdict = verdicts.get(risk_level, "")

    st.markdown(f"""
    <div style="
      background: linear-gradient(90deg, {bg} 0%, rgba(0,0,0,0) 100%);
      border: 1px solid {border};
      border-left: 4px solid {c};
      border-radius: 12px;
      padding: 28px 32px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin: 1rem 0 3rem;
      position: relative;
      overflow: hidden;
    ">
      <div style="position:absolute; top:-50px; left:-50px; width:150px; height:150px; background:{c}; filter:blur(80px); opacity:0.15;"></div>
      
      <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: {c};
        letter-spacing: 0.1em;
        text-transform: uppercase;
      ">❖ VERDICT: {risk_level} RISK</div>
      <div style="
        font-family: 'Outfit', sans-serif;
        font-size: 1.6rem;
        font-weight: 500;
        color: #FFFFFF;
        letter-spacing: -0.01em;
      ">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(title, subtitle=""):
    sub_html = f'<p style="font-size:0.95rem; color:#A0AEC0; margin:0; font-weight:400;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin: 3.5rem 0 2rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 1.2rem;">
      <h2 style="
        font-size: 1.5rem;
        font-weight: 500;
        color: #FFFFFF;
        margin: 0 0 8px;
        letter-spacing: -0.02em;
      ">{title}</h2>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def stat_card(label, value, sublabel="", color="#FFFFFF"):
    sub_html = f"""<div style="font-size:0.8rem; color:#4A5568; font-family:'Plus Jakarta Sans';">{sublabel}</div>""" if sublabel else ""
    st.markdown(f"""
    <div class="elite-card" style="padding: 20px 24px; position: relative; overflow: hidden;">
      <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, {color}80, transparent);"></div>
      
      <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #A0AEC0;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
      ">{label}</div>
      <div style="
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 400;
        color: {color};
        line-height: 1;
        margin-bottom: 8px;
        text-shadow: 0 0 24px {color}40;
      ">{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def component_risk_row(name, score, level):
    colors = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
    c = colors.get(level, "#A0AEC0")
    pct = (score / 3) * 100
    st.markdown(f"""
    <div style="
      display: flex; align-items: center; gap: 20px;
      padding: 14px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    ">
      <div style="
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.9rem;
        color: #CBD5E1;
        min-width: 180px;
        font-weight: 500;
      ">{name.replace('_',' ').title()}</div>
      
      <div style="flex:1; height:4px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden;">
        <div style="
          width:{pct}%; height:100%;
          background: {c};
          border-radius: 4px;
          box-shadow: 0 0 10px {c}80;
        "></div>
      </div>
      
      <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: {c};
        min-width: 80px;
        text-align: right;
        font-weight: 600;
      ">{level}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="padding: 1.5rem 0;">
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#00F0FF; letter-spacing:0.1em; margin-bottom:8px;">❖ SYSTEM ONLINE</div>
  <div style="font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:600; color:#FFFFFF;">Auditor Engine</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br/>", unsafe_allow_html=True)
model_name = st.sidebar.text_input("Model Identifier", value="Production-v2.0-Omega")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
n_bins    = st.sidebar.slider("Calibration Bins",   5,   20,  10)

st.sidebar.markdown("<br/>", unsafe_allow_html=True)
prod_file      = st.sidebar.file_uploader("Production Predictions", type=["csv"])
ref_file       = st.sidebar.file_uploader("Reference Features",     type=["csv"])
prod_feat_file = st.sidebar.file_uploader("Production Features",    type=["csv"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

hero_header()

if prod_file is None:
    section_header("Awaiting Telemetry", "Upload model predictions to initialize the auditing sequence.")
    st.markdown("""
    <div class="elite-card" style="font-family: 'JetBrains Mono', monospace;">
      <div style="color:#00F0FF; font-size:0.75rem; margin-bottom:16px; letter-spacing:0.1em;">EXPECTED FORMAT (PRODUCTION_DATA.CSV)</div>
      <div style="color:#A0AEC0; font-size:0.85rem; margin-bottom:8px;">y_true,y_prob</div>
      <div style="color:#FFFFFF; font-size:0.85rem; line-height:2;">
        1,0.92<br/>0,0.15<br/>1,0.87<br/>
        <span style="color:#4A5568;">...</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

try:
    prod_df = pd.read_csv(prod_file)
    assert "y_true" in prod_df.columns and "y_prob" in prod_df.columns
    y_true = prod_df["y_true"].values.astype(int)
    y_prob = prod_df["y_prob"].values.astype(float)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
except Exception as e:
    st.error(f"Error loading production data: {e}")
    st.stop()

ref_df = None
prod_feat_df = None
if ref_file:
    try: ref_df = pd.read_csv(ref_file)
    except: st.warning("Could not load reference features.")
if prod_feat_file:
    try: prod_feat_df = pd.read_csv(prod_feat_file)
    except: st.warning("Could not load production features.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Executing high-precision reliability metrics..."):
    metrics     = compute_metrics(y_true, y_prob, threshold=threshold)
    calibration = compute_ece(y_true, y_prob, n_bins=n_bins)

    pred_drift = None
    feature_drift = None

    if ref_df is not None and "y_prob" in ref_df.columns:
        pred_drift = compute_prediction_drift(ref_df["y_prob"].values.astype(float), y_prob)

    if ref_df is not None and prod_feat_df is not None:
        feature_drift = compute_feature_drift(ref_df, prod_feat_df)
    elif ref_df is not None and all(c not in ref_df.columns for c in ["y_true","y_prob"]):
        extra_prod = prod_df.drop(columns=["y_true","y_prob"], errors="ignore")
        if len(extra_prod.columns) > 0:
            feature_drift = compute_feature_drift(ref_df, extra_prod)

    drift_score = (feature_drift["overall_drift_score"] if feature_drift
                   else pred_drift["psi"] if pred_drift else None)

    risk = compute_risk_score(
        ece=calibration["ece"],
        brier_score=metrics["brier_score"],
        accuracy=metrics["accuracy"],
        drift_score=drift_score,
    )

    # Maintain backend logic for PDF generation
    rel_bytes   = reliability_diagram(y_true, y_prob, n_bins=n_bins, return_bytes=True)
    hist_bytes  = confidence_histogram(y_prob, return_bytes=True)
    drift_bytes = None
    if feature_drift and feature_drift["n_features"] > 0:
        drift_bytes = drift_heatmap(feature_drift, return_bytes=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

risk_level = risk["overall_risk_level"]

# ── Risk Badge ──
risk_badge(risk_level)

# ── Key Metrics ──
section_header("Core Telemetry", "Primary performance indicators and baseline health")

c1, c2, c3, c4, c5 = st.columns(5)
risk_colors = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
rc = risk_colors.get(risk_level, "#FFFFFF")

with c1: stat_card("ACCURACY",    f"{metrics['accuracy']:.1%}",   "Classification", "#FFFFFF")
with c2: stat_card("F1 SCORE",    f"{metrics['f1']:.3f}",         "Weighted",       "#FFFFFF")
with c3: stat_card("BRIER SCORE", f"{metrics['brier_score']:.4f}", "Probabilistic",  "#EF4444" if metrics['brier_score']>0.17 else "#FFFFFF")
with c4: stat_card("ECE",         f"{calibration['ece']:.4f}",     "Calibration",    "#EF4444" if calibration['ece']>0.10 else "#FFFFFF")
with c5: stat_card("DRIFT PSI",   f"{drift_score:.4f}" if drift_score else "N/A", "Distribution", rc)

# ── Interactive Calibration Analysis (Plotly Premium) ──
section_header("Calibration Dynamics", "Interactive reliability analysis and probability density")

col_left, col_right = st.columns([1.6, 1])

with col_left:
    st.markdown('<div class="elite-card" style="padding: 24px 24px 0 24px;">', unsafe_allow_html=True)
    
    # Calculate interactive calibration curve data
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # 1. Premium Plotly Reliability Diagram
    fig_rel = go.Figure()
    
    # Perfect calibration line (subtle dashed)
    fig_rel.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode='lines', 
        name='Ideal', 
        line=dict(dash='dash', color='#4A5568', width=1.5),
        hoverinfo='skip'
    ))
    
    # Model calibration line (Glowing Cyan with Area Fill)
    fig_rel.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, 
        mode='lines+markers', 
        name='Model',
        line=dict(color='#00F0FF', width=3),
        marker=dict(size=10, color='#050507', line=dict(color='#00F0FF', width=2.5)), 
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.08)',
        hovertemplate='<b>Predicted:</b> %{x:.3f}<br><b>Actual:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig_rel.update_layout(
        title=dict(text="Reliability Curve", font=dict(color='#FFFFFF', size=16, family="Outfit")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A0AEC0', family="Plus Jakarta Sans"),
        xaxis=dict(title="Mean Predicted Probability", gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)', range=[0, 1]),
        yaxis=dict(title="Fraction of Positives", gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)', range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=340,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(13, 14, 18, 0.9)", font_size=13, font_family="Plus Jakarta Sans", bordercolor="rgba(0, 240, 255, 0.3)")
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    # 2. Premium Plotly Confidence Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=y_prob, 
        nbinsx=n_bins, 
        marker_color='rgba(138, 43, 226, 0.6)', 
        marker_line_color='#8A2BE2', 
        marker_line_width=1.5,
        name='Predictions',
        hovertemplate='<b>Probability Bin:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    fig_hist.update_layout(
        title=dict(text="Confidence Density", font=dict(color='#FFFFFF', size=16, family="Outfit")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A0AEC0', family="Plus Jakarta Sans"),
        xaxis=dict(title="Predicted Probability", gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)', range=[0, 1]),
        yaxis=dict(title="Frequency", gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)'),
        margin=dict(l=40, r=20, t=50, b=40),
        height=220,
        bargap=0.15,
        hoverlabel=dict(bgcolor="rgba(13, 14, 18, 0.9)", font_size=13, font_family="Plus Jakarta Sans", bordercolor="rgba(138, 43, 226, 0.5)")
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Diagnostics Card
    st.markdown("""
    <div class="elite-card" style="margin-bottom:20px;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#00F0FF; margin-bottom:24px; letter-spacing:0.1em;">❖ DIAGNOSTIC VECTORS</div>
    """, unsafe_allow_html=True)

    cal_stats = [
        ("Expected Cal. Error (ECE)", f"{calibration['ece']:.4f}"),
        ("Maximum Cal. Error (MCE)",  f"{calibration['mce']:.4f}"),
        ("Overconfidence Gap",        f"{calibration['overconfidence_gap']:.4f}"),
        ("Analysis Bin Count",        str(calibration['n_bins'])),
    ]
    for label, value in cal_stats:
        oc = "#EF4444" if (label.startswith("Expected") and calibration['ece']>0.10) else "#FFFFFF"
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
          <span style="font-size:0.9rem; color:#CBD5E1; font-weight:500;">{label}</span>
          <span style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:600; color:{oc};">{value}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Risk Factors Card
    st.markdown("""
    <div class="elite-card">
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#00F0FF; margin-bottom:20px; letter-spacing:0.1em;">❖ RISK CONTEXT</div>
    """, unsafe_allow_html=True)
    for comp, info in risk["component_scores"].items():
        component_risk_row(comp, info["score"], info["level"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── Drift ──
if drift_bytes or pred_drift:
    section_header("Distribution Shift", "Covariate shift detection and Population Stability Index (PSI)")
    d1, d2 = st.columns([1.6, 1])
    with d1:
        if drift_bytes:
            st.markdown('<div class="elite-card">', unsafe_allow_html=True)
            st.image(drift_bytes, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with d2:
        if pred_drift or feature_drift:
            st.markdown("""
            <div class="elite-card">
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#00F0FF; margin-bottom:20px; letter-spacing:0.1em;">❖ SHIFT SIGNALS</div>
            """, unsafe_allow_html=True)
            if pred_drift:
                for label, value in [("KL Divergence", f"{pred_drift['kl_divergence']:.4f}"),
                                     ("Prediction PSI", f"{pred_drift['psi']:.4f}")]:
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                      <span style="font-size:0.9rem; color:#CBD5E1; font-weight:500;">{label}</span>
                      <span style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:600; color:#FFFFFF;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            if feature_drift:
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                  <span style="font-size:0.9rem; color:#CBD5E1; font-weight:500;">Feature PSI</span>
                  <span style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:600; color:#FFFFFF;">{feature_drift['overall_drift_score']:.4f}</span>
                </div>
                <div style="font-size:0.85rem; color:#A0AEC0; padding-top:16px; line-height:1.6;">
                  {feature_drift['overall_interpretation']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ── PDF Report ──
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="
  text-align: center;
  max-width: 700px;
  margin: 0 auto 3rem;
">
  <h3 style="font-weight:600; font-size:2rem; color:#FFFFFF; margin-bottom:16px; letter-spacing:-0.02em;">Export Intelligence Package</h3>
  <p style="font-size:1.05rem; color:#A0AEC0; line-height:1.7; margin-bottom:32px;">
    Compile a cryptographic-grade, institutional risk assessment PDF. Includes interactive calibration plots, isolated drift metrics, and formal deployment recommendations.
  </p>
</div>
""", unsafe_allow_html=True)

col_btn = st.columns([1, 1, 1])
with col_btn[1]:
    if st.button("GENERATE INSTITUTIONAL REPORT", type="primary", use_container_width=True):
        with st.spinner("Compiling cryptographic intelligence document..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                pdf_path = f.name
            
            # Using your backend logic untouched
            generate_report(
                output_path=pdf_path,
                model_name=model_name,
                metrics=metrics,
                calibration=calibration,
                risk=risk,
                drift=feature_drift,
                prediction_drift=pred_drift,
                reliability_diagram_bytes=rel_bytes,     
                confidence_hist_bytes=hist_bytes,        
                drift_chart_bytes=drift_bytes,
                n_samples=len(y_true),
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            os.unlink(pdf_path)

        st.success("Intelligence Package Compiled Successfully.")
        st.download_button(
            label="DOWNLOAD SECURE PDF",
            data=pdf_bytes,
            file_name=f"{model_name.replace(' ','_')}_audit.pdf",
            mime="application/pdf",
            use_container_width=True,
        )