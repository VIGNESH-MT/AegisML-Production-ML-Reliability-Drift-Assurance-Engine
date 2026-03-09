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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AegisML | Sentinel",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS + MARVEL COMIC FLIP INTRO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;500;600;700;900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">

<style>
:root {
  --r:   #E50914;
  --rb:  #8B0000;
  --b:   #0050FF;
  --bg:  #040406;
  --si:  #C0C8D8;
  --di:  #445060;
  --bdr: rgba(229,9,20,0.22);
}

/* ── GLOBAL ── */
.stApp {
  background: var(--bg) !important;
  font-family: 'Rajdhani', sans-serif !important;
  color: #fff !important;
  overflow-x: hidden !important;
}
.stApp::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:
    radial-gradient(ellipse 90% 70% at 5% 0%,   rgba(229,9,20,0.16) 0%, transparent 60%),
    radial-gradient(ellipse 70% 55% at 95% 100%, rgba(0,50,200,0.14) 0%, transparent 60%);
  animation: bgShift 12s ease-in-out infinite alternate;
}
@keyframes bgShift { 0%{opacity:.7} 100%{opacity:1.1} }
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.35;
  background-image:
    linear-gradient(0deg,   rgba(229,9,20,.055) 1px, transparent 1px),
    linear-gradient(60deg,  rgba(229,9,20,.055) 1px, transparent 1px),
    linear-gradient(120deg, rgba(229,9,20,.055) 1px, transparent 1px);
  background-size: 58px 58px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: rgba(4,3,8,0.97) !important;
  border-right: 1px solid var(--bdr) !important;
  backdrop-filter: blur(26px) !important;
}
[data-testid="stSidebar"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg, transparent, var(--r), var(--b), transparent);
  animation: topScan 3s ease-in-out infinite;
}
@keyframes topScan { 0%,100%{opacity:.35} 50%{opacity:1} }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
  color:var(--si) !important; font-family:'Share Tech Mono',monospace !important;
  font-size:.67rem !important; letter-spacing:.1em !important; text-transform:uppercase !important;
}
[data-testid="stSidebar"] input {
  background:rgba(229,9,20,.04) !important; border:1px solid var(--bdr) !important;
  border-radius:4px !important; color:#fff !important; font-family:'Rajdhani',sans-serif !important;
}
[data-testid="stSidebar"] input:focus {
  border-color:var(--r) !important; box-shadow:0 0 14px rgba(229,9,20,.22) !important;
}

/* ── MAIN ── */
.main .block-container { padding-top:0 !important; max-width:1520px; position:relative; z-index:1; }
h1,h2,h3,h4 { font-family:'Bebas Neue',sans-serif !important; color:#fff !important; letter-spacing:.05em !important; }

/* ── CARDS ── */
.sp-card {
  background:rgba(14,6,10,0.75); backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px);
  border:1px solid var(--bdr); border-radius:12px; padding:26px;
  position:relative; overflow:hidden;
  transition:transform .35s ease, border-color .35s ease, box-shadow .35s ease;
  box-shadow:0 8px 40px rgba(0,0,0,.65), inset 0 1px 0 rgba(255,255,255,.03);
}
.sp-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg, transparent, rgba(229,9,20,.7), transparent);
}
.sp-card:hover {
  transform:translateY(-4px) scale(1.005); border-color:rgba(229,9,20,.5);
  box-shadow:0 18px 64px rgba(229,9,20,.15), 0 8px 40px rgba(0,0,0,.75);
}

/* ── STAT CARDS ── */
.stat-lbl { font-family:'Share Tech Mono',monospace; font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:var(--di); margin-bottom:12px; }
.stat-val { font-family:'Bebas Neue',sans-serif; font-size:2.9rem; line-height:1; letter-spacing:.06em; }
.stat-sub { font-family:'Barlow Condensed',sans-serif; font-size:.8rem; color:var(--di); margin-top:6px; }

/* ── BUTTON ── */
.stButton > button {
  background:linear-gradient(135deg,#E50914 0%,#8B0000 100%) !important;
  color:#fff !important; font-family:'Bebas Neue',sans-serif !important;
  font-size:1.15rem !important; letter-spacing:.13em !important;
  border:none !important; border-radius:6px !important; padding:14px 34px !important;
  box-shadow:0 4px 22px rgba(229,9,20,.45), inset 0 1px 0 rgba(255,255,255,.13) !important;
  transition:all .3s ease !important;
}
.stButton > button:hover { transform:translateY(-3px) !important; box-shadow:0 10px 38px rgba(229,9,20,.65) !important; }

/* ── MISC ── */
hr { border:none !important; height:1px !important; margin:3rem 0 !important; background:linear-gradient(90deg,transparent,rgba(229,9,20,.4),transparent) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:#3D1010; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--r); }
#MainMenu, footer, header { visibility:hidden; }
.js-plotly-plot .plotly .modebar { display:none !important; }
@keyframes glitch {
  0%,88%,100% { text-shadow:0 0 28px rgba(229,9,20,.7),0 0 55px rgba(0,80,255,.35); }
  90% { text-shadow:-3px 0 #E50914,3px 0 #0050FF; }
  93% { text-shadow:3px 0 #E50914,-3px 0 #0050FF; }
  96% { text-shadow:0 0 28px rgba(229,9,20,.7); }
}

/* ════════════════════════════════════════
   MARVEL COMIC-PAGE FLIP INTRO
   ════════════════════════════════════════ */
#intro-root {
  position:fixed; inset:0; z-index:9999; background:#000;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  overflow:hidden;
  animation:introExit .9s ease-in-out 7.2s forwards;
}
@keyframes introExit {
  0%  { opacity:1; transform:scale(1);    pointer-events:auto; }
  40% { opacity:1; transform:scale(1.04); }
  100%{ opacity:0; transform:scale(1.08); pointer-events:none; visibility:hidden; }
}
#intro-root::before {
  content:''; position:absolute; inset:0; z-index:10; background:var(--r);
  animation:redFlash .18s ease-out 4.05s forwards; opacity:0; pointer-events:none;
}
@keyframes redFlash { 0%{opacity:.85} 100%{opacity:0} }
#intro-root::after {
  content:''; position:absolute; inset:0; z-index:8; pointer-events:none;
  background:repeating-linear-gradient(0deg, rgba(0,0,0,0) 0, rgba(0,0,0,0) 2px, rgba(0,0,0,.12) 2px, rgba(0,0,0,.12) 4px);
}

/* PHASE 1 — COMIC STAGE */
#comic-stage {
  position:absolute; inset:0; perspective:1400px; background:#0a0002;
  animation:stageExit .25s ease-in 4.0s forwards; z-index:2;
}
@keyframes stageExit { 0%{opacity:1} 100%{opacity:0;pointer-events:none} }

.page { position:absolute; inset:0; transform-origin:left center; transform-style:preserve-3d; backface-visibility:hidden; display:flex; }
.page-left {
  width:50%; height:100%; background:#060106;
  border-right:2px solid rgba(229,9,20,.6);
  display:flex; align-items:center; justify-content:center;
  padding:40px; overflow:hidden; position:relative;
}
.page-left::after { content:''; position:absolute; inset:0; background:linear-gradient(135deg,rgba(229,9,20,.04) 0%,transparent 60%); }
.page-right {
  width:50%; height:100%; background:var(--r);
  display:flex; align-items:center; justify-content:center;
  position:relative; overflow:hidden;
}
.page-right::before { content:''; position:absolute; inset:0; background:radial-gradient(circle at 60% 40%,rgba(255,255,255,.18) 0%,transparent 65%); }

.code-block {
  font-family:'Share Tech Mono',monospace; font-size:clamp(.55rem,1.1vw,.78rem);
  line-height:1.65; color:#e8e8e8; white-space:pre; text-align:left;
  position:relative; z-index:1; max-width:100%; overflow:hidden;
}
.code-block .kw { color:#FF6B6B } .code-block .fn { color:#FFD93D }
.code-block .st { color:#6BCB77 } .code-block .cm { color:#556070 }
.code-block .er { color:#FF4444; font-weight:bold } .code-block .nm { color:#4ECDC4 }

.page-big-txt {
  font-family:'Bebas Neue',sans-serif; font-size:clamp(3rem,7vw,7rem);
  color:#fff; letter-spacing:.08em; line-height:.95; text-align:center;
  text-shadow:0 4px 20px rgba(0,0,0,.5); position:relative; z-index:1;
}
.page-big-txt .page-num { display:block; font-size:clamp(1rem,2vw,1.6rem); letter-spacing:.25em; opacity:.55; margin-bottom:12px; }

.page:nth-child(1){ z-index:18; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) .10s both }
.page:nth-child(2){ z-index:17; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) .42s both }
.page:nth-child(3){ z-index:16; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) .74s both }
.page:nth-child(4){ z-index:15; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) 1.06s both }
.page:nth-child(5){ z-index:14; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) 1.38s both }
.page:nth-child(6){ z-index:13; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) 1.70s both }
.page:nth-child(7){ z-index:12; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) 2.02s both }
.page:nth-child(8){ z-index:11; animation:flipPage .38s cubic-bezier(.4,1.4,.6,1) 2.34s both }

@keyframes flipPage {
  0%  { transform:rotateY(0deg)    translateZ(0px);  opacity:1 }
  40% { transform:rotateY(-110deg) translateZ(80px); opacity:1 }
  70% { transform:rotateY(-175deg) translateZ(30px); opacity:.8 }
  100%{ transform:rotateY(-180deg) translateZ(0px);  opacity:0 }
}
.page::after { content:''; position:absolute; top:0; left:0; width:15px; height:100%; background:linear-gradient(90deg,rgba(0,0,0,.7),transparent); z-index:5; }

/* PHASE 2 — LOGO */
#logo-phase {
  position:absolute; inset:0; z-index:4;
  display:flex; flex-direction:column; align-items:center; justify-content:center; background:#000;
  animation:logoPhaseIn .3s ease-out 4.1s both, logoPhaseOut .5s ease-in 6.3s forwards;
}
@keyframes logoPhaseIn  { from{opacity:0} to{opacity:1} }
@keyframes logoPhaseOut { from{opacity:1} to{opacity:0;pointer-events:none} }

.aegis-logo-box { background:var(--r); padding:18px 56px; position:relative; overflow:hidden; animation:logoScale 1.1s cubic-bezier(.22,1.4,.36,1) 4.2s both; }
@keyframes logoScale { 0%{transform:scaleX(0) skewX(-5deg)} 50%{transform:scaleX(1.04) skewX(-2deg)} 100%{transform:scaleX(1) skewX(0deg)} }
.aegis-logo-box::before { content:''; position:absolute; inset:0; background:linear-gradient(135deg,rgba(255,255,255,.2) 0%,transparent 60%); }
.aegis-logo-txt { font-family:'Bebas Neue',sans-serif; font-size:clamp(3.5rem,9vw,8.5rem); color:#fff; letter-spacing:.18em; position:relative; z-index:1; text-shadow:0 3px 0 rgba(0,0,0,.3),0 0 40px rgba(255,255,255,.2); }
.logo-sub-line { font-family:'Share Tech Mono',monospace; font-size:clamp(.65rem,1.3vw,.9rem); letter-spacing:.38em; color:rgba(255,255,255,.55); text-transform:uppercase; margin-top:18px; animation:fadeUp .7s ease 4.9s both; }
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:none} }

/* PHASE 3 — SPIDER DROP */
#spider-phase {
  position:absolute; inset:0; z-index:5; display:flex; align-items:center; justify-content:center;
  perspective:1200px; animation:spiderIn .2s ease 6.35s both; pointer-events:none;
}
@keyframes spiderIn { from{opacity:0} to{opacity:1} }

.sp-drop-card {
  width:min(540px,88vw);
  background: radial-gradient(ellipse at 15% 10%,rgba(229,9,20,.5) 0%,transparent 52%),
              radial-gradient(ellipse at 85% 90%,rgba(0,50,200,.4) 0%,transparent 52%),
              rgba(5,3,12,.97);
  border:1px solid rgba(229,9,20,.38); border-radius:18px; padding:48px 44px 42px;
  position:relative; overflow:hidden; transform-origin:center top;
  animation: sp3DDrop 1.1s cubic-bezier(.22,1.1,.36,1) 6.4s both,
             spSway   5s ease-in-out 7.6s infinite;
  box-shadow:0 0 90px rgba(229,9,20,.45),0 0 130px rgba(0,50,200,.3),0 50px 110px rgba(0,0,0,.95);
}
@keyframes sp3DDrop {
  0%  { transform:rotateX(85deg) translateY(-140vh) scale(.3); opacity:0 }
  50% { transform:rotateX(-6deg) translateY(3vh)    scale(1.02); opacity:1 }
  75% { transform:rotateX(3deg)  translateY(-2vh)   scale(.99) }
  100%{ transform:rotateX(0deg)  translateY(0)      scale(1); opacity:1 }
}
@keyframes spSway {
  0%,100%{ transform:rotate(0deg)    translateY(0)   }
  33%    { transform:rotate(-1.4deg) translateY(3px) }
  66%    { transform:rotate(1.1deg)  translateY(5px) }
}
.sp-thread {
  position:absolute; width:2px; height:80vh; top:-80vh; left:50%; transform:translateX(-50%);
  background:linear-gradient(to bottom,rgba(255,255,255,0) 0%,rgba(255,255,255,.75) 40%,rgba(255,255,255,.95) 65%,rgba(255,255,255,.25) 100%);
  filter:drop-shadow(0 0 8px rgba(255,255,255,.5)); animation:threadDraw .6s ease 6.3s both;
}
.sp-thread::before,.sp-thread::after { content:''; position:absolute; width:1px; height:100%; background:inherit; opacity:.35; }
.sp-thread::before{left:-6px} .sp-thread::after{left:6px}
@keyframes threadDraw { from{transform:translateX(-50%) scaleY(0);transform-origin:top} to{transform:translateX(-50%) scaleY(1)} }

.sp-web-deco {
  position:absolute; width:260px; height:260px; border-radius:50%; top:-70px; right:-70px; opacity:.06;
  background: repeating-radial-gradient(circle,transparent 0,transparent 18px,rgba(255,255,255,.9) 19px,transparent 20px),
              repeating-conic-gradient(rgba(255,255,255,.9) 0deg 2deg,transparent 2deg 30deg);
  animation:webSpin 22s linear infinite;
}
@keyframes webSpin { to{transform:rotate(360deg)} }

.c-tl,.c-tr,.c-bl,.c-br { position:absolute; width:20px; height:20px; border-color:rgba(229,9,20,.6); border-style:solid; }
.c-tl{top:16px;left:16px;border-width:2px 0 0 2px} .c-tr{top:16px;right:16px;border-width:2px 2px 0 0}
.c-bl{bottom:16px;left:16px;border-width:0 0 2px 2px} .c-br{bottom:16px;right:16px;border-width:0 2px 2px 0}

.sp-eyebrow { font-family:'Share Tech Mono',monospace; font-size:.66rem; letter-spacing:.22em; color:rgba(255,255,255,.5); text-transform:uppercase; margin-bottom:14px; display:flex; align-items:center; gap:12px; animation:fadeUp .6s ease 6.8s both; }
.sp-line{flex:1;height:1px;background:linear-gradient(90deg,var(--r),var(--b));max-width:45px}
.sp-title { font-family:'Bebas Neue',sans-serif; font-size:clamp(2.2rem,5vw,3.7rem); line-height:1; letter-spacing:.04em; color:#fff; margin-bottom:14px; text-shadow:0 0 30px rgba(229,9,20,.7),0 0 60px rgba(0,80,255,.35); animation:fadeUp .6s ease 7.0s both; }
.sp-body { font-family:'Barlow Condensed',sans-serif; font-size:1.05rem; color:rgba(192,200,216,.85); line-height:1.65; margin-bottom:26px; animation:fadeUp .6s ease 7.15s both; }
.sp-badge { display:inline-flex; align-items:center; gap:10px; padding:8px 16px; border-radius:4px; border:1px solid rgba(229,9,20,.4); background:rgba(229,9,20,.08); font-family:'Share Tech Mono',monospace; font-size:.61rem; letter-spacing:.18em; color:rgba(255,255,255,.65); text-transform:uppercase; animation:fadeUp .6s ease 7.3s both; }
.sp-dot { width:8px; height:8px; border-radius:50%; background:#22C55E; box-shadow:0 0 12px rgba(34,197,94,.9),0 0 24px rgba(34,197,94,.5); animation:dotPulse 1.5s ease-in-out infinite; }
@keyframes dotPulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.3);opacity:.7} }

/* ── DASHBOARD COMPONENTS ── */
.risk-verdict { border-radius:10px; padding:30px 34px; margin:1rem 0 3rem; position:relative; overflow:hidden; }
.sec-hdr { margin:4rem 0 1.8rem; padding-bottom:.9rem; border-bottom:1px solid rgba(229,9,20,.1); position:relative; }
.sec-hdr::before { content:''; position:absolute; bottom:-1px; left:0; width:80px; height:1px; background:var(--r); box-shadow:0 0 10px var(--r); }
.sec-title { font-family:'Bebas Neue',sans-serif; font-size:1.9rem; letter-spacing:.07em; color:#fff; margin:0 0 5px; }
.sec-sub { font-family:'Barlow Condensed',sans-serif; font-size:.97rem; color:var(--di); font-weight:400; }
.comp-row { display:flex; align-items:center; gap:18px; padding:13px 0; border-bottom:1px solid rgba(229,9,20,.07); }
.hero-wrap { padding:5rem 0 3rem; position:relative; z-index:1; }
.hero-eyebrow { font-family:'Share Tech Mono',monospace; font-size:.71rem; color:var(--r); letter-spacing:.22em; text-transform:uppercase; margin-bottom:1.1rem; display:flex; align-items:center; gap:13px; }
.hero-rule { display:inline-block; width:40px; height:2px; background:linear-gradient(90deg,var(--r),var(--b)); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMIC PAGE DATA
# ─────────────────────────────────────────────────────────────────────────────
pages_data = [
    {
        "code": '<span class="cm"># model_audit.py v3.1.0</span>\n<span class="kw">import</span> <span class="nm">numpy</span> <span class="kw">as</span> <span class="nm">np</span>\n<span class="kw">import</span> <span class="nm">pandas</span> <span class="kw">as</span> <span class="nm">pd</span>\n<span class="kw">from</span> <span class="nm">sklearn</span> <span class="kw">import</span> <span class="nm">metrics</span>\n\n<span class="fn">def</span> <span class="fn">compute_ece</span>(y_true, y_prob):\n    <span class="st">"""Expected Calibration Error"""</span>\n    bins = <span class="nm">np</span>.linspace(<span class="nm">0</span>, <span class="nm">1</span>, n_bins)\n    ece  = <span class="nm">0.0</span>\n    <span class="kw">for</span> b <span class="kw">in</span> <span class="fn">zip</span>(bins[:-<span class="nm">1</span>], bins[<span class="nm">1</span>:]):\n        mask = (y_prob &gt;= b[<span class="nm">0</span>]) &amp; (y_prob &lt; b[<span class="nm">1</span>])\n        <span class="kw">if</span> mask.<span class="fn">sum</span>() &gt; <span class="nm">0</span>:\n            acc  = y_true[mask].<span class="fn">mean</span>()\n            conf = y_prob[mask].<span class="fn">mean</span>()\n            ece += mask.<span class="fn">sum</span>() * <span class="fn">abs</span>(acc-conf)\n    <span class="kw">return</span> ece / <span class="fn">len</span>(y_true)',
        "big": "MODEL\nINIT", "num": "01/08",
    },
    {
        "code": '<span class="er">Traceback (most recent call last):</span>\n  File <span class="st">"pipeline.py"</span>, line <span class="nm">147</span>\n    <span class="er">raise DriftException(</span>\n        <span class="st">"PSI threshold exceeded: "</span>\n        <span class="st">f"score={psi:.4f} &gt; 0.25"</span>\n    <span class="er">)</span>\n\n<span class="er">DriftException: Distribution shift</span>\n<span class="er">detected in feature space.</span>\n<span class="er">PSI = 0.3847 — CRITICAL</span>\n\n<span class="cm">&gt;&gt;&gt; Triggering auto-rollback...</span>\n<span class="cm">&gt;&gt;&gt; Alerting SRE on-call...</span>\n<span class="er">DEPLOYMENT BLOCKED</span>',
        "big": "DRIFT\nALERT", "num": "02/08",
    },
    {
        "code": '<span class="cm"># calibration_surface.py</span>\n<span class="kw">from</span> <span class="nm">plotly</span> <span class="kw">import</span> <span class="nm">graph_objects</span> <span class="kw">as</span> <span class="nm">go</span>\n\nX, Y = <span class="nm">np</span>.meshgrid(\n    <span class="nm">np</span>.linspace(<span class="nm">0</span>, <span class="nm">1</span>, <span class="nm">50</span>),\n    <span class="nm">np</span>.linspace(<span class="nm">0</span>, <span class="nm">1</span>, <span class="nm">50</span>))\n\nZ = <span class="nm">np</span>.abs(X - Y)  <span class="cm"># gap</span>\n\nfig.<span class="fn">add_trace</span>(<span class="nm">go</span>.Surface(\n    x=X, y=Y, z=Z,\n    colorscale=<span class="st">"RdYlGn_r"</span>,\n    opacity=<span class="nm">0.92</span>,\n    contours_z=<span class="nm">dict</span>(\n        show=<span class="nm">True</span>,\n        usecolormap=<span class="nm">True</span>,\n        highlightcolor=<span class="st">"#E50914"</span>,\n        project_z=<span class="nm">True</span>)))',
        "big": "3D\nSURFACE", "num": "03/08",
    },
    {
        "code": '<span class="er">CRITICAL ERROR [0x4F3A]</span>\n<span class="er">MemoryError: OOM on GPU:0</span>\n<span class="cm">──────────────────────</span>\n<span class="cm">Allocated: 23.8 GiB</span>\n<span class="cm">Reserved:  24.0 GiB</span>\n<span class="er">──────────────────────</span>\n<span class="nm">RuntimeError</span>: CUDA device\nassert triggered at:\n  <span class="fn">torch/_C/_VariableFunctions</span>\n  line <span class="nm">5821</span>\n\n<span class="cm">&gt;&gt;&gt; Batch size: 512 -&gt; 64</span>\n<span class="cm">&gt;&gt;&gt; Mixed precision: fp16</span>\n<span class="er">RETRYING INFERENCE...</span>',
        "big": "CUDA\nERROR", "num": "04/08",
    },
    {
        "code": '<span class="cm"># risk_score.py</span>\n<span class="fn">def</span> <span class="fn">compute_risk</span>(ece,brier,acc,psi):\n    W = {\n        <span class="st">"calibration"</span>: <span class="nm">0.35</span>,\n        <span class="st">"accuracy"</span>:    <span class="nm">0.25</span>,\n        <span class="st">"brier"</span>:       <span class="nm">0.20</span>,\n        <span class="st">"drift"</span>:       <span class="nm">0.20</span>,\n    }\n    S = {\n        <span class="st">"calibration"</span>: <span class="fn">min</span>(ece/<span class="nm">.15</span>,<span class="nm">1.0</span>),\n        <span class="st">"accuracy"</span>:    <span class="nm">1</span>-<span class="fn">min</span>(acc,<span class="nm">1.0</span>),\n        <span class="st">"brier"</span>:       brier/<span class="nm">.25</span>,\n        <span class="st">"drift"</span>:       <span class="fn">min</span>(psi/<span class="nm">.25</span>,<span class="nm">1.0</span>),\n    }\n    <span class="kw">return</span> <span class="fn">sum</span>(W[k]*S[k] <span class="kw">for</span> k <span class="kw">in</span> W)',
        "big": "RISK\nENGINE", "num": "05/08",
    },
    {
        "code": '<span class="er">WARNING: Brier score degraded</span>\n<span class="er">Delta = +0.0842 over 72h</span>\n<span class="cm">──────────────────────</span>\n<span class="cm">Threshold: 0.1700</span>\n<span class="er">Current:   0.2542  HIGH</span>\n<span class="cm">──────────────────────</span>\n<span class="nm">ModelMonitor</span>.check(\n    window=<span class="st">"72h"</span>,\n    metric=<span class="st">"brier_score"</span>,\n    baseline=<span class="nm">0.1700</span>,\n    current=<span class="nm">0.2542</span>,\n    threshold=<span class="nm">0.05</span>\n)\n<span class="er">ALERT -&gt; PagerDuty</span>\n<span class="er">INCIDENT INC-4471 CREATED</span>',
        "big": "MODEL\nDEGRADE", "num": "06/08",
    },
    {
        "code": '<span class="cm"># feature_drift.py</span>\n<span class="fn">def</span> <span class="fn">compute_psi</span>(ref,prod,bins=<span class="nm">10</span>):\n    r = <span class="fn">_bin_counts</span>(ref, bins)\n    p = <span class="fn">_bin_counts</span>(prod, bins)\n    r = <span class="nm">np</span>.where(r==<span class="nm">0</span>,<span class="nm">1e-6</span>,r)\n    p = <span class="nm">np</span>.where(p==<span class="nm">0</span>,<span class="nm">1e-6</span>,p)\n    psi = <span class="nm">np</span>.sum(\n        (p-r)*<span class="nm">np</span>.log(p/r)\n    )\n    <span class="kw">return</span> <span class="fn">float</span>(psi)\n\n<span class="cm"># PSI &lt; 0.10 -&gt; stable</span>\n<span class="cm"># PSI 0.10-0.25 -&gt; monitor</span>\n<span class="er"># PSI &gt; 0.25 -&gt; CRITICAL</span>',
        "big": "PSI\nDETECT", "num": "07/08",
    },
    {
        "code": '<span class="cm"># sentinel_report.py</span>\n<span class="kw">from</span> <span class="nm">reportlab.platypus</span> <span class="kw">import</span> *\n\n<span class="fn">def</span> <span class="fn">generate_report</span>(model, metrics):\n    story = []\n    story.<span class="fn">append</span>(<span class="nm">Paragraph</span>(\n        <span class="st">f"SENTINEL AUDIT: {model}"</span>,\n        styles[<span class="st">"Title"</span>]))\n    story.<span class="fn">append</span>(<span class="nm">Spacer</span>(<span class="nm">1</span>,<span class="nm">24</span>))\n    <span class="kw">for</span> k,v <span class="kw">in</span> metrics.<span class="fn">items</span>():\n        story.<span class="fn">append</span>(<span class="nm">Paragraph</span>(\n            <span class="st">f"  {k}: {v:.4f}"</span>,\n            styles[<span class="st">"Body"</span>]))\n    doc.<span class="fn">build</span>(story)\n    <span class="kw">return</span> <span class="st">"REPORT_COMPILED"</span>',
        "big": "REPORT\nOUT", "num": "08/08",
    },
]

pages_html = ""
for p in pages_data:
    pages_html += f"""
    <div class="page">
      <div class="page-left"><div class="code-block">{p["code"]}</div></div>
      <div class="page-right">
        <div class="page-big-txt">
          <span class="page-num">{p["num"]}</span>{p["big"]}
        </div>
      </div>
    </div>"""

st.markdown(f"""
<div id="intro-root">
  <div id="comic-stage">{pages_html}</div>
  <div id="logo-phase">
    <div class="aegis-logo-box"><div class="aegis-logo-txt">AEGISML</div></div>
    <div class="logo-sub-line">STUDIOS · SENTINEL DIVISION · PRODUCTION INTELLIGENCE</div>
  </div>
  <div id="spider-phase">
    <div class="sp-drop-card">
      <div class="sp-thread"></div>
      <div class="sp-web-deco"></div>
      <div class="c-tl"></div><div class="c-tr"></div>
      <div class="c-bl"></div><div class="c-br"></div>
      <div class="sp-eyebrow">
        <span class="sp-line"></span>
        AEGISML // SPIDER-SENTINEL // PROD v3
        <span class="sp-line"></span>
      </div>
      <div class="sp-title">ML RELIABILITY<br/>&amp; WEB AUDITOR</div>
      <div class="sp-body">Swinging across your entire model skyline — tracing calibration error,
        covariate shift, and systemic risk across the production cityscape.</div>
      <div class="sp-badge">
        <span class="sp-dot"></span>
        SENTINEL ACTIVE · 3D DIAGNOSTICS ONLINE · AWAITING UPLOAD
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def hero_header():
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-eyebrow">
        <span class="hero-rule"></span>
        PRODUCTION SPIDER-SENTINEL · DEPLOYMENT INTELLIGENCE ENGINE
      </div>
      <h1 style="font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6.5vw,5.8rem);
                 line-height:.93;letter-spacing:.03em;margin:0 0 1.2rem;
                 background:linear-gradient(168deg,#FFFFFF 0%,#E0E0E0 30%,#E50914 72%,#8B0000 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 animation:glitch 7s ease-in-out infinite;">
        Model Reliability<br/>&amp; Distribution<br/>Auditor
      </h1>
      <p style="font-family:'Barlow Condensed',sans-serif;font-size:1.15rem;
                color:#A0AEC0;max-width:600px;line-height:1.6;">
        Real-time calibration fidelity, covariate shift detection, and
        institutional-grade risk assessment — powered by 3D intelligence.
      </p>
    </div>""", unsafe_allow_html=True)


def risk_badge(risk_level):
    pal = {
        "Low":      ("#10B981","rgba(16,185,129,.09)","rgba(16,185,129,.38)"),
        "Medium":   ("#F59E0B","rgba(245,158,11,.09)","rgba(245,158,11,.38)"),
        "High":     ("#EF4444","rgba(239,68,68,.09)", "rgba(239,68,68,.48)"),
        "Critical": ("#E11D48","rgba(225,29,72,.11)", "rgba(225,29,72,.58)"),
    }
    vd = {
        "Low":      "PRODUCTION DEPLOYMENT AUTHORIZED",
        "Medium":   "DEPLOYMENT APPROVED — MONITORING REQUIRED",
        "High":     "DEPLOYMENT BLOCKED // ELEVATED RISK",
        "Critical": "CRITICAL THREAT // IMMEDIATE ACTION REQUIRED",
    }
    c, bg, bgl = pal.get(risk_level, pal["High"])
    v          = vd.get(risk_level, "")
    pulse      = "animation:vpulse 2s ease-in-out infinite;" if risk_level in ("High","Critical") else ""
    st.markdown(f"""
    <style>@keyframes vpulse{{0%,100%{{box-shadow:0 0 22px {c}40}}50%{{box-shadow:0 0 55px {c}80}}}}</style>
    <div class="risk-verdict" style="background:linear-gradient(110deg,{bg} 0%,rgba(0,0,0,0) 68%);
         border:1px solid {bgl};border-left:4px solid {c};{pulse}">
      <div style="position:absolute;top:-60px;left:-60px;width:180px;height:180px;
                  background:{c};filter:blur(80px);opacity:.18;pointer-events:none;"></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;color:{c};
                  letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;">
        ▸ VERDICT · {risk_level.upper()} RISK PROFILE</div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:2.3rem;letter-spacing:.06em;
                  color:#fff;text-shadow:0 0 28px {c}80;line-height:1;">{v}</div>
    </div>""", unsafe_allow_html=True)


def section_header(title, subtitle=""):
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sec-hdr"><div class="sec-title">{title}</div>{sub}</div>',
                unsafe_allow_html=True)


def stat_card(label, value, sublabel="", color="#FFFFFF"):
    glow = f"text-shadow:0 0 30px {color}60;" if color != "#FFFFFF" else ""
    sub  = f'<div class="stat-sub">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
    <div class="sp-card" style="padding:22px 24px;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{color}90,transparent);"></div>
      <div class="stat-lbl">{label}</div>
      <div class="stat-val" style="color:{color};{glow}">{value}</div>{sub}
    </div>""", unsafe_allow_html=True)


def component_risk_row(name, score, level):
    cm = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
    c  = cm.get(level,"#A0AEC0")
    pct = (score / 3) * 100
    st.markdown(f"""
    <div class="comp-row">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:600;
                  color:#C0C8D8;min-width:190px;letter-spacing:.03em;">
        {name.replace('_',' ').upper()}</div>
      <div style="flex:1;height:3px;background:rgba(229,9,20,.08);border-radius:2px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{c},{c}99);
                    border-radius:2px;box-shadow:0 0 10px {c}90;"></div></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.77rem;color:{c};
                  min-width:72px;text-align:right;letter-spacing:.08em;">{level.upper()}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3D CALIBRATION SURFACE  (fixed — no duplicate keyword args)
# ─────────────────────────────────────────────────────────────────────────────
def build_3d_calibration_surface(y_true, y_prob, n_bins=10):
    edges       = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    frac_pos    = np.zeros(n_bins)
    counts      = np.zeros(n_bins)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            frac_pos[i] = y_true[mask].mean()
            counts[i]   = mask.sum()

    n_time  = 30
    time_ax = np.linspace(0, 1, n_time)
    X, T    = np.meshgrid(bin_centers, time_ax)
    Z_base  = np.abs(bin_centers - frac_pos)
    Z       = np.tile(Z_base, (n_time, 1))
    for t_i, t in enumerate(time_ax):
        wave   = 0.06 * np.sin(np.pi * t * 3 + bin_centers * np.pi * 2)
        Z[t_i] = np.clip(Z[t_i] + wave * (1 - Z_base), 0, 1)

    Z_ideal = np.zeros_like(Z) + 0.002
    fig     = go.Figure()

    # Main calibration surface
    fig.add_trace(go.Surface(
        x=X, y=T, z=Z,
        name="Calibration Gap",
        colorscale=[
            [0.00, "#0A3A1A"], [0.15, "#10B981"],
            [0.35, "#F59E0B"], [0.65, "#EF4444"], [1.00, "#E11D48"],
        ],
        opacity=0.92, showscale=True,
        colorbar=dict(
            title=dict(text="Gap", font=dict(color="#A0AEC0", family="Share Tech Mono", size=11)),
            tickfont=dict(color="#A0AEC0", family="Share Tech Mono", size=10),
            x=1.02, len=0.7, bgcolor="rgba(0,0,0,0)", bordercolor="rgba(229,9,20,0.3)",
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#E50914",
                   project=dict(z=True), width=1.5),
            x=dict(show=True, color="rgba(229,9,20,0.15)", width=1),
        ),
        hovertemplate="<b>Pred Prob:</b> %{x:.2f}<br><b>Time:</b> %{y:.2f}<br><b>Cal.Gap:</b> %{z:.3f}<extra></extra>",
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.3, fresnel=0.5),
        lightposition=dict(x=100, y=200, z=150),
    ))

    # Ideal-calibration plane
    fig.add_trace(go.Surface(
        x=X, y=T, z=Z_ideal, name="Ideal (Zero Gap)",
        colorscale=[[0, "rgba(0,200,255,0.25)"], [1, "rgba(0,200,255,0.25)"]],
        opacity=0.25, showscale=False, hoverinfo="skip",
    ))

    # Current-model scatter dots
    max_c = max(counts.max(), 1)
    fig.add_trace(go.Scatter3d(
        x=bin_centers,
        y=np.full(n_bins, 0.5),
        z=Z_base + 0.025,
        mode="markers+lines",
        name="Current Model",
        marker=dict(
            size=[max(6, c / max_c * 18) for c in counts],
            color=Z_base,
            colorscale=[[0, "#10B981"], [0.4, "#F59E0B"], [1, "#E50914"]],
            line=dict(color="rgba(255,255,255,0.5)", width=1),
            opacity=0.95,
        ),
        line=dict(color="rgba(229,9,20,0.6)", width=3),
        hovertemplate="<b>Bin:</b> %{x:.2f}<br><b>Gap:</b> %{z:.3f}<extra>Current Snapshot</extra>",
    ))

    # Vertical drop lines
    for i in range(n_bins):
        fig.add_trace(go.Scatter3d(
            x=[bin_centers[i], bin_centers[i]],
            y=[0.5, 0.5],
            z=[0.002, Z_base[i] + 0.025],
            mode="lines",
            line=dict(color="rgba(229,9,20,0.3)", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Axis dicts built individually — no shared base with overrides ──────
    xaxis_cfg = dict(
        title="Predicted Probability",
        titlefont=dict(color="#A0AEC0", family="Barlow Condensed", size=13),
        tickfont=dict(color="#A0AEC0", family="Share Tech Mono", size=10),
        gridcolor="rgba(229,9,20,0.12)",
        zerolinecolor="rgba(229,9,20,0.2)",
        showbackground=True,
        backgroundcolor="rgba(229,9,20,0.03)",
    )
    yaxis_cfg = dict(
        title="Time Window",
        titlefont=dict(color="#A0AEC0", family="Barlow Condensed", size=13),
        tickfont=dict(color="#A0AEC0", family="Share Tech Mono", size=10),
        gridcolor="rgba(229,9,20,0.08)",
        zerolinecolor="rgba(229,9,20,0.2)",
        showbackground=True,
        backgroundcolor="rgba(4,3,8,0.5)",
    )
    zaxis_cfg = dict(
        title="Calibration Gap",
        titlefont=dict(color="#A0AEC0", family="Barlow Condensed", size=13),
        tickfont=dict(color="#A0AEC0", family="Share Tech Mono", size=10),
        gridcolor="rgba(229,9,20,0.08)",
        zerolinecolor="rgba(229,9,20,0.2)",
        showbackground=True,
        backgroundcolor="rgba(4,3,8,0.5)",
        range=[0, 0.8],
    )

    fig.update_layout(
        scene=dict(
            xaxis=xaxis_cfg,
            yaxis=yaxis_cfg,
            zaxis=zaxis_cfg,
            bgcolor="rgba(4,3,8,0.0)",
            camera=dict(
    eye=dict(
        x=1.6*np.cos(st.session_state.get("cam_angle",0)),
        y=1.6*np.sin(st.session_state.get("cam_angle",0)),
        z=1.1
    )
)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0", family="Barlow Condensed"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        legend=dict(
            font=dict(color="#A0AEC0", family="Share Tech Mono", size=10),
            bgcolor="rgba(10,6,10,0.7)",
            bordercolor="rgba(229,9,20,0.25)",
            borderwidth=1, x=0.01, y=0.99,
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,6,10,0.95)",
            font=dict(family="Barlow Condensed", size=13, color="#fff"),
            bordercolor="rgba(229,9,20,0.5)",
        ),
    )
    return fig
if "cam_angle" not in st.session_state:
    st.session_state.cam_angle = 0

st.session_state.cam_angle += 0.02

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:1.8rem 0 .5rem;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.64rem;color:#E50914;
              letter-spacing:.18em;margin-bottom:6px;">▸ SENTINEL ONLINE</div>
  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
              letter-spacing:.1em;color:#fff;">AUDITOR ENGINE</div>
  <div style="width:100%;height:1px;margin-top:12px;opacity:.5;
              background:linear-gradient(90deg,#E50914,transparent);"></div>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("<br/>", unsafe_allow_html=True)
model_name     = st.sidebar.text_input("Model Identifier", value="Sentinel-v3.0-Omega")
threshold      = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
n_bins         = st.sidebar.slider("Calibration Bins",   5,   20,  10)

st.sidebar.markdown("<br/>", unsafe_allow_html=True)
prod_file      = st.sidebar.file_uploader("Production Predictions", type=["csv"])
ref_file       = st.sidebar.file_uploader("Reference Features",     type=["csv"])
prod_feat_file = st.sidebar.file_uploader("Production Features",    type=["csv"])

st.sidebar.markdown("""
<div style="margin-top:2rem;padding:12px 14px;border-radius:8px;
            border:1px solid rgba(229,9,20,.2);background:rgba(229,9,20,.04);">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;
              color:#E50914;letter-spacing:.12em;margin-bottom:8px;">FORMAT: y_true,y_prob</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.75rem;
              color:#A0AEC0;line-height:1.9;">
    1,0.92<br/>0,0.15<br/>1,0.87<br/><span style="color:#3D3D4D;">...</span>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
hero_header()

if prod_file is None:
    section_header("Awaiting Telemetry", "Upload model predictions to ignite the auditing sequence")
    st.markdown("""
    <div class="sp-card" style="max-width:520px;font-family:'Share Tech Mono',monospace;">
      <div style="color:#E50914;font-size:.7rem;margin-bottom:16px;letter-spacing:.15em;">
        ▸ EXPECTED FORMAT — PRODUCTION_DATA.CSV</div>
      <div style="color:#A0AEC0;font-size:.8rem;margin-bottom:8px;">y_true, y_prob</div>
      <div style="color:#fff;font-size:.85rem;line-height:2.1;">
        1, 0.92<br/>0, 0.15<br/>1, 0.87<br/><span style="color:#3D3D4D;">...</span></div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    prod_df = pd.read_csv(prod_file)
    assert "y_true" in prod_df.columns and "y_prob" in prod_df.columns
    y_true  = prod_df["y_true"].values.astype(int)
    y_prob  = prod_df["y_prob"].values.astype(float)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
except Exception as e:
    st.error(f"Error loading production data: {e}")
    st.stop()

ref_df = prod_feat_df = None
if ref_file:
    try:    ref_df = pd.read_csv(ref_file)
    except: st.warning("Could not load reference features.")
if prod_feat_file:
    try:    prod_feat_df = pd.read_csv(prod_feat_file)
    except: st.warning("Could not load production features.")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🕷 Executing Sentinel diagnostics..."):
    metrics     = compute_metrics(y_true, y_prob, threshold=threshold)
    calibration = compute_ece(y_true, y_prob, n_bins=n_bins)

    pred_drift = feature_drift = None
    if ref_df is not None and "y_prob" in ref_df.columns:
        pred_drift = compute_prediction_drift(ref_df["y_prob"].values.astype(float), y_prob)
    if ref_df is not None and prod_feat_df is not None:
        feature_drift = compute_feature_drift(ref_df, prod_feat_df)
    elif ref_df is not None and all(c not in ref_df.columns for c in ["y_true","y_prob"]):
        extra = prod_df.drop(columns=["y_true","y_prob"], errors="ignore")
        if len(extra.columns) > 0:
            feature_drift = compute_feature_drift(ref_df, extra)

    drift_score = (feature_drift["overall_drift_score"] if feature_drift
                   else pred_drift["psi"] if pred_drift else None)

    risk = compute_risk_score(
        ece=calibration["ece"],
        brier_score=metrics["brier_score"],
        accuracy=metrics["accuracy"],
        drift_score=drift_score,
    )

    rel_bytes   = reliability_diagram(y_true, y_prob, n_bins=n_bins, return_bytes=True)
    hist_bytes  = confidence_histogram(y_prob, return_bytes=True)
    drift_bytes = None
    if feature_drift and feature_drift["n_features"] > 0:
        drift_bytes = drift_heatmap(feature_drift, return_bytes=True)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
risk_level = risk["overall_risk_level"]
risk_badge(risk_level)

_AXIS = dict(
    gridcolor="rgba(229,9,20,0.07)",
    zerolinecolor="rgba(229,9,20,0.15)",
    linecolor="rgba(229,9,20,0.18)",
)
_BASE = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#A0AEC0", family="Barlow Condensed", size=13),
    margin=dict(l=40, r=18, t=48, b=42),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(10,6,10,0.96)",
        font=dict(family="Barlow Condensed", size=13, color="#fff"),
        bordercolor="rgba(229,9,20,0.5)",
    ),
)

# ── CORE TELEMETRY ─────────────────────────────────────────────────────────
section_header("Core Telemetry", "Primary performance indicators and baseline health signals")

c1, c2, c3, c4, c5 = st.columns(5)
rc_map = {"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#E11D48"}
rc = rc_map.get(risk_level, "#FFFFFF")

with c1: stat_card("ACCURACY",    f"{metrics['accuracy']:.1%}",    "Classification",  "#FFFFFF")
with c2: stat_card("F1 SCORE",    f"{metrics['f1']:.3f}",          "Weighted",        "#FFFFFF")
with c3: stat_card("BRIER SCORE", f"{metrics['brier_score']:.4f}", "Probabilistic",   "#EF4444" if metrics["brier_score"] > 0.17 else "#FFFFFF")
with c4: stat_card("ECE",         f"{calibration['ece']:.4f}",     "Calibration Err", "#EF4444" if calibration["ece"] > 0.10 else "#FFFFFF")
with c5: stat_card("DRIFT PSI",   f"{drift_score:.4f}" if drift_score else "N/A", "Distribution", rc)


# ── 3D CALIBRATION SURFACE ─────────────────────────────────────────────────
section_header(
    "3D Calibration Landscape",
    "Interactive surface: Z = calibration gap · Drag to rotate · Scroll to zoom · Hover for values"
)

st.markdown("""
<div class="sp-card" style="padding:18px 18px 0;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.66rem;
              color:#E50914;letter-spacing:.13em;margin-bottom:4px;">
    ▸ CALIBRATION ERROR SURFACE &nbsp;·&nbsp;
    <span style="color:#A0AEC0;">RED = HIGH GAP &nbsp;·&nbsp; GREEN = WELL CALIBRATED &nbsp;·&nbsp; DOTS = CURRENT MODEL &nbsp;·&nbsp; DRAG TO EXPLORE</span>
  </div>""", unsafe_allow_html=True)

fig_3d = build_3d_calibration_surface(y_true, y_prob, n_bins=n_bins)
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ── CALIBRATION DETAILS + RISK PANEL ───────────────────────────────────────
section_header("Calibration Diagnostics", "Confidence density and risk decomposition")

col_left, col_right = st.columns([1.55, 1])

with col_left:
    st.markdown('<div class="sp-card" style="padding:24px 24px 0;">', unsafe_allow_html=True)

    # Confidence histogram
bc  = np.linspace(0, 1, n_bins + 1)
mid = 0.5 * (bc[:-1] + bc[1:])
cnt, _ = np.histogram(y_prob, bins=bc)

fig_hist = go.Figure()

fig_hist.add_trace(go.Bar(

    x=mid,
    y=cnt,
    width=0.85 / n_bins,

    marker=dict(
        color=cnt,

        colorscale=[
            [0,"#00FFAA"],
            [0.35,"#00C8FF"],
            [0.6,"#FFD400"],
            [0.8,"#FF6A00"],
            [1,"#E50914"]
        ],

        line=dict(color="#E50914", width=1.5)
    ),

    hovertemplate=
    "<b>Probability:</b> %{x:.2f}<br>"
    "<b>Sample Count:</b> %{y}<extra></extra>"
))


fig_hist.add_trace(go.Scatter(

    x=mid,
    y=cnt,

    mode="lines",

    line=dict(
        color="#E50914",
        width=3
    ),

    fill="tozeroy",
    fillcolor="rgba(229,9,20,0.08)",

    showlegend=False,
    hoverinfo="skip"
))


fig_hist.update_layout(

    **_BASE,

    title=dict(
        text="CONFIDENCE DENSITY",
        font=dict(color="#fff", size=15, family="Bebas Neue"),
        x=.02
    ),

    xaxis=dict(
        **_AXIS,
        title="Predicted Probability",
        range=[0,1]
    ),

    yaxis=dict(
        **_AXIS,
        title="Sample Count"
    ),

    height=265,
    bargap=.12,
    showlegend=False
)

st.plotly_chart(fig_hist, use_container_width=True)



# ───────────────── 2D RELIABILITY CURVE ─────────────────

prob_true_rc, prob_pred_rc = calibration_curve(
    y_true,
    y_prob,
    n_bins=n_bins,
    strategy="uniform"
)

fig_rel = go.Figure()


# Ideal calibration line
fig_rel.add_trace(go.Scatter(

    x=[0,1],
    y=[0,1],

    mode="lines",

    line=dict(
        dash="dash",
        color="rgba(0,200,255,0.6)",
        width=2
    ),

    name="Ideal"
))


# Model reliability curve
fig_rel.add_trace(go.Scatter(

    x=prob_pred_rc,
    y=prob_true_rc,

    mode="lines+markers",

    line=dict(
        color="#E50914",
        width=4
    ),

    marker=dict(

        size=10,

        color=prob_true_rc,

        colorscale="Turbo",

        line=dict(
            color="white",
            width=2
        )
    ),

    fill="tozeroy",
    fillcolor="rgba(229,9,20,0.08)",

    hovertemplate=
    "<b>Predicted:</b> %{x:.3f}<br>"
    "<b>Actual:</b> %{y:.3f}<extra></extra>"
))


fig_rel.update_layout(

    **_BASE,

    title=dict(
        text="RELIABILITY CURVE",
        font=dict(color="#fff", size=15, family="Bebas Neue"),
        x=.02
    ),

    xaxis=dict(
        **_AXIS,
        title="Mean Predicted Probability",
        range=[0,1]
    ),

    yaxis=dict(
        **_AXIS,
        title="Fraction of Positives",
        range=[0,1]
    ),

    height=295,
    showlegend=False
)


st.plotly_chart(fig_rel, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── DISTRIBUTION SHIFT ──────────────────────────────────────────────────────
if drift_bytes or pred_drift or feature_drift:

    section_header(
        "Distribution Shift",
        "Covariate shift · Population Stability Index · Feature drift"
    )

    d1, d2 = st.columns([1.55, 1])

    # ───────────────── LEFT PANEL ─────────────────
    with d1:

        # Existing drift heatmap
        if drift_bytes:
            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            st.image(drift_bytes, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

       # ───────────────── NEW 3D DRIFT VISUALIZATION ─────────────────
if ref_df is not None and prod_feat_df is not None:

    st.markdown(
        """
        <div class="sp-card" style="margin-top:18px;">
        <div style="font-family:'Share Tech Mono',monospace;
                    font-size:.66rem;color:#E50914;
                    letter-spacing:.13em;margin-bottom:6px;">
        ▸ 3D FEATURE DRIFT LANDSCAPE
        </div>
        """,
        unsafe_allow_html=True
    )

    common_cols = list(set(ref_df.columns).intersection(set(prod_feat_df.columns)))

    feature_names = []
    ref_means = []
    prod_means = []
    drift_vals = []

    for col in common_cols:

        if pd.api.types.is_numeric_dtype(ref_df[col]):

            r = ref_df[col].dropna().values
            p = prod_feat_df[col].dropna().values

            if len(r) == 0 or len(p) == 0:
                continue

            rm = np.mean(r)
            pm = np.mean(p)

            drift = abs(rm - pm)

            feature_names.append(col)
            ref_means.append(rm)
            prod_means.append(pm)
            drift_vals.append(drift)

    if len(feature_names) > 0:

        x = np.arange(len(feature_names))

        fig_drift3d = go.Figure()

        # ── MAIN DRIFT POINTS
        fig_drift3d.add_trace(go.Scatter3d(

            x=x,
            y=ref_means,
            z=prod_means,

            mode="markers+lines",

            marker=dict(

                size=11,

                color=drift_vals,

                colorscale=[
                    [0,"#00FFAA"],
                    [0.25,"#00C8FF"],
                    [0.5,"#FFD400"],
                    [0.75,"#FF6A00"],
                    [1,"#E50914"]
                ],

                opacity=0.95,

                line=dict(
                    color="white",
                    width=1
                )
            ),

            line=dict(
                color="rgba(229,9,20,0.5)",
                width=3
            ),

            text=feature_names,

            hovertemplate=
            "<b>Feature:</b> %{text}<br>"
            "<b>Reference Mean:</b> %{y:.4f}<br>"
            "<b>Production Mean:</b> %{z:.4f}<br>"
            "<b>Drift Magnitude:</b> %{marker.color:.4f}<extra></extra>"
        ))


        # ── DRIFT PILLARS
        for i in range(len(x)):

            fig_drift3d.add_trace(go.Scatter3d(

                x=[x[i], x[i]],
                y=[ref_means[i], ref_means[i]],
                z=[0, prod_means[i]],

                mode="lines",

                line=dict(
                    color="rgba(229,9,20,0.35)",
                    width=2
                ),

                showlegend=False,
                hoverinfo="skip"
            ))


        fig_drift3d.update_layout(

            scene=dict(

                xaxis=dict(
                    title="Feature",
                    tickmode="array",
                    tickvals=x,
                    ticktext=feature_names,
                    gridcolor="rgba(229,9,20,0.1)"
                ),

                yaxis=dict(
                    title="Reference Distribution",
                    gridcolor="rgba(229,9,20,0.08)"
                ),

                zaxis=dict(
                    title="Production Distribution",
                    gridcolor="rgba(229,9,20,0.08)"
                ),

                bgcolor="rgba(0,0,0,0)",

                camera=dict(
                    eye=dict(x=1.4, y=-1.6, z=1.2)
                )
            ),

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            margin=dict(l=0, r=0, t=10, b=0),

            height=440
        )

        st.plotly_chart(fig_drift3d, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    # ───────────────── RIGHT PANEL ─────────────────
    with d2:

        if pred_drift or feature_drift:

            st.markdown(
                """
                <div class="sp-card">
                <div style="font-family:'Share Tech Mono',monospace;
                            font-size:.67rem;color:#E50914;
                            letter-spacing:.15em;margin-bottom:20px;">
                ▸ SHIFT SIGNALS
                </div>
                """,
                unsafe_allow_html=True
            )

            if pred_drift:

                for lbl, val in [
                    ("KL Divergence", f"{pred_drift['kl_divergence']:.4f}"),
                    ("Prediction PSI", f"{pred_drift['psi']:.4f}")
                ]:

                    st.markdown(
                        f"""
                        <div style="display:flex;justify-content:space-between;
                                    padding:12px 0;
                                    border-bottom:1px solid rgba(229,9,20,.07);">
                        <span style="font-family:'Barlow Condensed',sans-serif;
                                     font-size:.97rem;font-weight:600;color:#C0C8D8;">
                        {lbl}</span>

                        <span style="font-family:'Share Tech Mono',monospace;
                                     font-size:.9rem;color:#fff;">
                        {val}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if feature_drift:

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:12px 0;
                                border-bottom:1px solid rgba(229,9,20,.07);">

                    <span style="font-family:'Barlow Condensed',sans-serif;
                                 font-size:.97rem;font-weight:600;color:#C0C8D8;">
                    Feature PSI
                    </span>

                    <span style="font-family:'Share Tech Mono',monospace;
                                 font-size:.9rem;color:#fff;">
                    {feature_drift['overall_drift_score']:.4f}
                    </span>

                    </div>

                    <div style="font-family:'Barlow Condensed',sans-serif;
                                font-size:.95rem;color:#A0AEC0;
                                padding-top:14px;line-height:1.6;">
                    {feature_drift['overall_interpretation']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;max-width:680px;margin:0 auto 2.5rem;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:.67rem;
              color:#E50914;letter-spacing:.2em;margin-bottom:12px;">▸ INTELLIGENCE PACKAGE</div>
  <h3 style="font-family:'Bebas Neue',sans-serif;font-size:2.7rem;
             letter-spacing:.07em;color:#fff;margin:0 0 12px;line-height:1;">EXPORT AUDIT REPORT</h3>
  <p style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
            color:#A0AEC0;line-height:1.65;">
    Compile an institutional-grade PDF audit package including 3D calibration plots,
    drift metrics, risk scores, and formal deployment recommendations.
  </p>
</div>""", unsafe_allow_html=True)

_, cb, _ = st.columns([1, 1, 1])
with cb:
    if st.button("🕷  GENERATE REPORT", type="primary", use_container_width=True):
        with st.spinner("Compiling intelligence document..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                pdf_path = f.name
            generate_report(
                output_path=pdf_path, model_name=model_name,
                metrics=metrics, calibration=calibration, risk=risk,
                drift=feature_drift, prediction_drift=pred_drift,
                reliability_diagram_bytes=rel_bytes,
                confidence_hist_bytes=hist_bytes,
                drift_chart_bytes=drift_bytes,
                n_samples=len(y_true),
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            os.unlink(pdf_path)

        st.success("🕷 Audit Package compiled successfully.")
        st.download_button(
            label="⬇  DOWNLOAD AUDIT PDF",
            data=pdf_bytes,
            file_name=f"{model_name.replace(' ', '_')}_sentinel_audit.pdf",
            mime="application/pdf",
            use_container_width=True,
        )