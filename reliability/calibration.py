"""
calibration.py — Elite dark-theme charts + ECE computation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings  # Added missing import to fix NameError
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from io import BytesIO

# UI Styling Constants
BG_VOID  = "#020408"
BG_PANEL = "#060d18"
BG_CARD  = "#0a1828"
CYAN     = "#00d4ff"
BLUE     = "#0066ff"
GREEN    = "#00ff88"
ORANGE   = "#ff6b35"
RED      = "#ff2d55"
YELLOW   = "#ffb800"
TEXT_PRI = "#e8f4ff"
TEXT_SEC = "#5a8aaa"
GRID_COL = "#0d2035"


def _dark(fig, axes):
    fig.patch.set_facecolor(BG_VOID)
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.title.set_color(TEXT_PRI)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.8)


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []; ece = mce = total_oc = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (y_prob>=lo)&(y_prob<=hi) if i==n_bins-1 else (y_prob>=lo)&(y_prob<hi)
        bs = int(mask.sum())
        if bs == 0:
            bins.append({"bin_lower":round(lo,2),"bin_upper":round(hi,2),"count":0,
                         "accuracy":None,"confidence":None,"calibration_error":None}); continue
        ba = float(y_true[mask].mean()); bc = float(y_prob[mask].mean())
        ce = abs(ba-bc); ece += (bs/n)*ce; mce = max(mce,ce)
        total_oc += (bs/n)*(bc-ba)
        bins.append({"bin_lower":round(lo,2),"bin_upper":round(hi,2),"count":bs,
                     "accuracy":round(ba,4),"confidence":round(bc,4),"calibration_error":round(ce,4)})
    return {"ece":round(ece,4),"mce":round(mce,4),"overconfidence_gap":round(total_oc,4),"n_bins":n_bins,"bins":bins}


def reliability_diagram(y_true, y_prob, n_bins=10, title="Reliability Diagram", return_bytes=False):
    cal  = compute_ece(y_true, y_prob, n_bins=n_bins)
    bins = [b for b in cal["bins"] if b["count"]>0]
    centers=[( b["bin_lower"]+b["bin_upper"])/2 for b in bins]
    accs   =[b["accuracy"]          for b in bins]
    confs  =[b["confidence"]        for b in bins]
    counts =[b["count"]             for b in bins]
    errs   =[b["calibration_error"] for b in bins]

    fig = plt.figure(figsize=(10,7), facecolor=BG_VOID)
    gs  = GridSpec(4,1,figure=fig,hspace=0.05)
    ax  = fig.add_subplot(gs[:3,0]); axb = fig.add_subplot(gs[3,0],sharex=ax)
    _dark(fig,[ax,axb])
    w = 1.0/n_bins-0.015

    for cx,acc,conf in zip(centers,accs,confs):
        c = ORANGE if conf>acc else GREEN
        top,bot = max(acc,conf),min(acc,conf)
        ax.bar(cx,top-bot,width=w*0.98,bottom=bot,color=c,alpha=0.12,zorder=2)
        ax.bar(cx,acc,width=w,color=c,alpha=0.8,edgecolor="none",zorder=3)
        ax.bar(cx,0.005,width=w,bottom=acc-0.0025,color=c,alpha=1.0,zorder=4)

    ax.plot([0,1],[0,1],color=CYAN,linewidth=1.5,linestyle="--",alpha=0.55,zorder=5)
    ax.scatter(centers,confs,color=CYAN,s=65,zorder=6,edgecolors="#fff",linewidths=0.5)
    ax.scatter(centers,confs,color=CYAN,s=200,zorder=5,alpha=0.12)

    for cx,acc,conf,ce,cnt in zip(centers,accs,confs,errs,counts):
        ax.annotate(f"n={cnt}\nΔ{ce:.3f}",xy=(cx,max(acc,conf)+0.025),
                    ha="center",va="bottom",fontsize=6,color=TEXT_SEC,fontfamily="monospace")

    ax.text(0.02,0.97,f"ECE  {cal['ece']:.4f}\nMCE  {cal['mce']:.4f}\nGap  {cal['overconfidence_gap']:+.4f}",
            transform=ax.transAxes,fontsize=8,color=CYAN,fontfamily="monospace",va="top",
            bbox=dict(boxstyle="round,pad=0.5",facecolor=BG_CARD,edgecolor=CYAN+"40",linewidth=0.8))

    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.12)
    ax.set_ylabel("Fraction of Positives",fontsize=10,labelpad=8)
    ax.set_title(f"  {title}",fontsize=12,fontweight="bold",loc="left",pad=12)
    ax.tick_params(labelbottom=False)

    handles=[mpatches.Patch(color=ORANGE,alpha=0.8,label="Overconfident"),
             mpatches.Patch(color=GREEN, alpha=0.8,label="Underconfident"),
             plt.Line2D([0],[0],color=CYAN,linestyle="--",label="Perfect"),
             plt.Line2D([0],[0],color=CYAN,marker="o",linestyle="",label="Mean conf")]
    ax.legend(handles=handles,loc="lower right",fontsize=8,framealpha=0.15,
              facecolor=BG_CARD,edgecolor=GRID_COL,labelcolor=TEXT_SEC)

    axb.bar(centers,counts,width=w,color=BLUE,alpha=0.6,edgecolor="none")
    axb.set_ylabel("n",fontsize=8,labelpad=8)
    axb.set_xlabel("Mean Predicted Probability",fontsize=10,labelpad=8)
    axb.set_ylim(0,max(counts)*1.4 if counts else 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    if return_bytes:
        buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=BG_VOID)
        plt.close(fig); buf.seek(0); return buf.read()
    plt.show(); plt.close(fig)


def confidence_histogram(y_prob, title="Confidence Distribution", return_bytes=False):
    y_prob = np.asarray(y_prob)
    fig,axes=plt.subplots(1,2,figsize=(12,5),facecolor=BG_VOID,gridspec_kw={"width_ratios":[2,1]})
    ax,axs=axes; _dark(fig,list(axes))

    n_bins_h=25
    counts_h,edges_h,patches_h=ax.hist(y_prob,bins=n_bins_h,edgecolor="none",alpha=0.9)
    for patch,left in zip(patches_h,edges_h[:-1]):
        if   left<0.3: patch.set_facecolor(RED);    patch.set_alpha(0.75)
        elif left<0.5: patch.set_facecolor(YELLOW);  patch.set_alpha(0.75)
        elif left<0.7: patch.set_facecolor(ORANGE);  patch.set_alpha(0.75)
        else:          patch.set_facecolor(GREEN);   patch.set_alpha(0.80)

    try:
        kde=gaussian_kde(y_prob,bw_method=0.15)
        xs=np.linspace(0,1,300); ys=kde(xs)*len(y_prob)*(1.0/n_bins_h)
        ax.plot(xs,ys,color=CYAN,linewidth=2,zorder=5)
        ax.fill_between(xs,ys,alpha=0.08,color=CYAN)
    except Exception:
        pass

    mp=float(y_prob.mean())
    ax.axvline(x=0.5,color="#fff",linewidth=1.2,linestyle="--",alpha=0.45,label="Threshold 0.5")
    ax.axvline(x=mp, color=YELLOW,linewidth=1.5,alpha=0.8,label=f"Mean={mp:.3f}")
    ax.set_xlabel("Predicted Probability",fontsize=10,labelpad=8)
    ax.set_ylabel("Count",fontsize=10,labelpad=8)
    ax.set_title(f"  {title}",fontsize=12,fontweight="bold",loc="left",pad=12)
    ax.legend(fontsize=8,framealpha=0.15,facecolor=BG_CARD,edgecolor=GRID_COL,labelcolor=TEXT_SEC)

    axs.set_xlim(0,1); axs.set_ylim(0,1); axs.axis("off")
    stats=[("SAMPLES",f"{len(y_prob):,}"),("MEAN",f"{mp:.4f}"),
            ("MEDIAN",f"{float(np.median(y_prob)):.4f}"),("STD",f"{float(y_prob.std()):.4f}"),
            ("MIN",f"{float(y_prob.min()):.4f}"),("MAX",f"{float(y_prob.max()):.4f}"),
            ("HIGH CONF",f"{(y_prob>=0.7).mean():.1%}"),("LOW CONF",f"{(y_prob<=0.3).mean():.1%}")]
    axs.text(0.5,0.97,"STATS",ha="center",va="top",fontsize=7,color=CYAN,
              fontfamily="monospace",transform=axs.transAxes)
    for i,(lb,vl) in enumerate(stats):
        yp=0.88-i*0.10
        axs.text(0.06,yp,lb,ha="left",va="center",fontsize=7,color=TEXT_SEC,
                  fontfamily="monospace",transform=axs.transAxes)
        axs.text(0.94,yp,vl,ha="right",va="center",fontsize=8,color=CYAN,
                  fontfamily="monospace",fontweight="bold",transform=axs.transAxes)
        axs.axhline(y=yp-0.04,color=GRID_COL,linewidth=0.4,alpha=0.6)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    if return_bytes:
        buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=BG_VOID)
        plt.close(fig); buf.seek(0); return buf.read()
    plt.show(); plt.close(fig)


def calibration_error_heatmap(y_true, y_prob, n_bins=10, return_bytes=False):
    """Diverging bar: overconfidence gap per bin + colour strip."""
    cal  = compute_ece(y_true, y_prob, n_bins=n_bins)
    bins = [b for b in cal["bins"] if b["count"]>0]
    centers=[( b["bin_lower"]+b["bin_upper"])/2 for b in bins]
    accs   =[b["accuracy"]          for b in bins]
    confs  =[b["confidence"]        for b in bins]
    errs   =[b["calibration_error"] for b in bins]
    counts =[b["count"]             for b in bins]

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,7),facecolor=BG_VOID,
                                gridspec_kw={"height_ratios":[3,1],"hspace":0.06})
    _dark(fig,[ax1,ax2])
    w=1.0/n_bins-0.01
    diffs=[c-a for c,a in zip(confs,accs)]
    bcolors=[ORANGE if d>0 else GREEN for d in diffs]

    ax1.bar(centers,diffs,width=w,color=bcolors,alpha=0.85,edgecolor="none",zorder=3)
    for cx,d,c in zip(centers,diffs,bcolors):
        if d!=0:
            ax1.bar(cx,0.003,width=w,bottom=d-(0.003 if d>0 else 0),color=c,alpha=1,zorder=4)
    ax1.axhline(y=0,color=CYAN,linewidth=1.0,alpha=0.35)

    for cx,d,cnt in zip(centers,diffs,counts):
        ax1.annotate(f"{d:+.3f}\nn={cnt}",
                     xy=(cx,d+(0.008 if d>=0 else -0.008)),
                     ha="center",va="bottom" if d>=0 else "top",
                     fontsize=6.5,color=TEXT_SEC,fontfamily="monospace")

    ax1.text(0.99,0.97,f"ECE={cal['ece']:.4f}  MCE={cal['mce']:.4f}",
             transform=ax1.transAxes,fontsize=8,color=CYAN,fontfamily="monospace",
             va="top",ha="right",
             bbox=dict(boxstyle="round,pad=0.4",facecolor=BG_CARD,edgecolor=CYAN+"40",linewidth=0.8))

    ax1.set_ylabel("Confidence − Accuracy",fontsize=9,labelpad=8)
    ax1.set_title("  Calibration Error Analysis — Per-Bin Gap",fontsize=12,
                  fontweight="bold",loc="left",pad=12)
    ax1.tick_params(labelbottom=False)
    ax1.legend(handles=[mpatches.Patch(color=ORANGE,alpha=0.85,label="Overconfident"),
                        mpatches.Patch(color=GREEN, alpha=0.85,label="Underconfident")],
               fontsize=8,framealpha=0.15,facecolor=BG_CARD,edgecolor=GRID_COL,
               labelcolor=TEXT_SEC,loc="lower right")

    errs_arr=np.array(errs)
    nv=(errs_arr-errs_arr.min())/(errs_arr.max()-errs_arr.min()+1e-8)
    for cx,n,e in zip(centers,nv,errs):
        col=plt.cm.YlOrRd(0.2+n*0.8)
        ax2.bar(cx,1,width=w,color=col,alpha=0.9,edgecolor="none")
        ax2.text(cx,0.5,f"{e:.3f}",ha="center",va="center",fontsize=7,
                 color="#fff",fontfamily="monospace",fontweight="bold")
    ax2.set_xlim(0,1); ax2.set_ylim(0,1)
    ax2.set_xlabel("Confidence Bin Centre",fontsize=9,labelpad=8)
    ax2.set_ylabel("|Error|",fontsize=8); ax2.set_yticks([])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    if return_bytes:
        buf=BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=BG_VOID)
        plt.close(fig); buf.seek(0); return buf.read()
    plt.show(); plt.close(fig)