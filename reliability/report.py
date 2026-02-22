"""
report.py
---------
Generates a structured, professional PDF risk report using ReportLab.

Sections:
  1. Cover Page
  2. Executive Summary
  3. Model Performance Metrics
  4. Calibration Analysis (+ reliability diagram)
  5. Distribution Drift Analysis (+ PSI chart)
  6. Risk Assessment
  7. Recommendations & Verdict
"""

from __future__ import annotations
import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus.flowables import Flowable


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BRAND_DARK   = colors.HexColor("#1a1a2e")
BRAND_MID    = colors.HexColor("#16213e")
BRAND_ACCENT = colors.HexColor("#0f3460")
BRAND_BLUE   = colors.HexColor("#2980b9")
RISK_LOW     = colors.HexColor("#27ae60")
RISK_MED     = colors.HexColor("#f39c12")
RISK_HIGH    = colors.HexColor("#e67e22")
RISK_CRIT    = colors.HexColor("#e74c3c")
LIGHT_GREY   = colors.HexColor("#f4f6f8")
MID_GREY     = colors.HexColor("#bdc3c7")
TEXT_DARK    = colors.HexColor("#2c3e50")

RISK_COLOR_MAP = {
    "Low": RISK_LOW,
    "Medium": RISK_MED,
    "High": RISK_HIGH,
    "Critical": RISK_CRIT,
}


# ---------------------------------------------------------------------------
# Page template with header/footer
# ---------------------------------------------------------------------------

class _ReportCanvas(rl_canvas.Canvas):
    """Adds running header/footer to every page."""

    def __init__(self, *args, model_name="ML Model", **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(n_pages)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_header_footer(self, page_count):
        page_num = self._pageNumber
        width, height = A4

        # Header bar
        self.setFillColor(BRAND_DARK)
        self.rect(0, height - 1.2 * cm, width, 1.2 * cm, fill=1, stroke=0)
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(1 * cm, height - 0.8 * cm, "ML RELIABILITY & DISTRIBUTION SHIFT AUDITOR")
        self.setFont("Helvetica", 8)
        self.drawRightString(width - 1 * cm, height - 0.8 * cm, self._model_name)

        # Footer
        self.setFillColor(MID_GREY)
        self.setFont("Helvetica", 7.5)
        self.drawString(1 * cm, 0.6 * cm,
                        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Confidential")
        self.drawRightString(width - 1 * cm, 0.6 * cm, f"Page {page_num} of {page_count}")
        self.setStrokeColor(MID_GREY)
        self.setLineWidth(0.5)
        self.line(1 * cm, 1.1 * cm, width - 1 * cm, 1.1 * cm)


# ---------------------------------------------------------------------------
# Style factory
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()

    custom = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=colors.white,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            fontName="Helvetica",
            fontSize=14,
            textColor=colors.HexColor("#aed6f1"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=BRAND_DARK,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "subsection_heading": ParagraphStyle(
            "subsection_heading",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=BRAND_ACCENT,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_DARK,
            spaceAfter=6,
            leading=15,
            alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            textColor=colors.HexColor("#7f8c8d"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "verdict": ParagraphStyle(
            "verdict",
            fontName="Helvetica",
            fontSize=10,
            textColor=TEXT_DARK,
            spaceAfter=6,
            leading=15,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            rightIndent=10,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=TEXT_DARK,
            spaceAfter=4,
            leading=14,
            leftIndent=14,
            bulletIndent=4,
        ),
    }
    return custom


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

class _CoverPageFlowable(Flowable):
    """Draws the entire cover page using direct canvas calls — pixel-perfect alignment."""

    def __init__(self, model_name, report_date):
        super().__init__()
        self.model_name  = model_name
        self.report_date = report_date
        self._width, self._height = A4

    def wrap(self, available_width, available_height):
        return 0, 0

    def draw(self):
        c   = self.canv
        # Save state and reset transform to draw at page coordinates
        c.saveState()
        c.translate(-self.canv._pagesize[0] * 0 , 0)  # no-op, already at origin
        W   = self._width
        H   = self._height

        # ── Full page background ─────────────────────────────────────────
        c.setFillColor(colors.HexColor("#f8f9fb"))
        c.rect(0, 0, W, H, fill=1, stroke=0)

        # ── Left accent stripe ───────────────────────────────────────────
        c.setFillColor(BRAND_DARK)
        c.rect(0, 0, 0.6 * cm, H, fill=1, stroke=0)

        # ── Top accent bar ───────────────────────────────────────────────
        c.setFillColor(BRAND_DARK)
        c.rect(0, H - 2.2 * cm, W, 2.2 * cm, fill=1, stroke=0)

        # Top bar text
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(1.4 * cm, H - 1.3 * cm,
                     "ML RELIABILITY & DISTRIBUTION SHIFT AUDITOR")
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#aed6f1"))
        c.drawRightString(W - 1.2 * cm, H - 1.3 * cm, "Deployment Risk Assessment Report")

        # ── Hero dark banner ─────────────────────────────────────────────
        banner_top    = H - 2.2 * cm
        banner_height = 8.5 * cm
        banner_bottom = banner_top - banner_height

        c.setFillColor(BRAND_DARK)
        c.rect(0, banner_bottom, W, banner_height, fill=1, stroke=0)

        # Subtle accent line at banner bottom
        c.setFillColor(BRAND_BLUE)
        c.rect(0, banner_bottom, W, 0.18 * cm, fill=1, stroke=0)

        # ── Main title inside banner ─────────────────────────────────────
        title_y = banner_bottom + banner_height - 2.2 * cm
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 36)
        c.drawString(1.4 * cm, title_y, "ML Reliability &")

        c.setFont("Helvetica-Bold", 36)
        c.drawString(1.4 * cm, title_y - 1.25 * cm, "Distribution Shift")

        c.setFont("Helvetica", 30)
        c.setFillColor(colors.HexColor("#aed6f1"))
        c.drawString(1.4 * cm, title_y - 2.5 * cm, "Auditor")

        # ── Tag line ─────────────────────────────────────────────────────
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.HexColor("#7fb3d3"))
        c.drawString(1.4 * cm, banner_bottom + 0.6 * cm,
                     "Calibration  ·  Drift Detection  ·  Deployment Risk Scoring")

        # ── Model info block ─────────────────────────────────────────────
        info_top = banner_bottom - 1.4 * cm
        c.setFillColor(colors.HexColor("#eaf4fb"))
        c.roundRect(1.4 * cm, info_top - 3.2 * cm,
                    W - 2.8 * cm, 3.2 * cm, 6, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#aed6f1"))
        c.setLineWidth(0.8)
        c.roundRect(1.4 * cm, info_top - 3.2 * cm,
                    W - 2.8 * cm, 3.2 * cm, 6, fill=0, stroke=1)

        # Labels
        c.setFillColor(BRAND_ACCENT)
        c.setFont("Helvetica-Bold", 9)
        row1_y = info_top - 1.1 * cm
        row2_y = info_top - 2.1 * cm
        row3_y = info_top - 3.0 * cm

        c.drawString(2.0 * cm, row1_y, "MODEL")
        c.drawString(2.0 * cm, row2_y, "REPORT DATE")
        c.drawString(2.0 * cm, row3_y, "TYPE")

        # Values
        c.setFillColor(TEXT_DARK)
        c.setFont("Helvetica", 10)
        c.drawString(6.5 * cm, row1_y, str(W))  # placeholder
        c.drawString(6.5 * cm, row1_y,  "")
        # We do this properly:
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(BRAND_DARK)
        # Truncate model name if too long
        mname = self.model_name[:50] + ("..." if len(self.model_name)>50 else "")
        c.drawString(5.5 * cm, row1_y, mname)

        c.setFont("Helvetica", 10)
        c.setFillColor(TEXT_DARK)
        c.drawString(5.5 * cm, row2_y, self.report_date)
        c.drawString(5.5 * cm, row3_y, "Deployment Risk Assessment  —  Confidential")

        # ── Vertical divider inside info block ───────────────────────────
        c.setStrokeColor(colors.HexColor("#aed6f1"))
        c.setLineWidth(0.5)
        c.line(5.0 * cm, info_top - 0.4 * cm, 5.0 * cm, info_top - 2.8 * cm)

        # ── Feature strip ────────────────────────────────────────────────
        feat_y = info_top - 4.2 * cm
        features = [
            ("PERFORMANCE",   "Accuracy · F1 · Brier Score"),
            ("CALIBRATION",   "ECE · MCE · Reliability Diagram"),
            ("DRIFT",         "PSI · KL Divergence · Feature Shift"),
            ("RISK SCORING",  "Low / Medium / High / Critical"),
        ]
        box_w = (W - 2.8 * cm) / 4
        for i, (label, desc) in enumerate(features):
            bx = 1.4 * cm + i * box_w
            c.setFillColor(BRAND_DARK if i % 2 == 0 else BRAND_ACCENT)
            c.roundRect(bx + 0.1*cm, feat_y - 2.0*cm, box_w - 0.2*cm, 2.0*cm, 4, fill=1, stroke=0)

            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(bx + box_w/2, feat_y - 0.8*cm, label)
            c.setFont("Helvetica", 6.5)
            c.setFillColor(colors.HexColor("#aed6f1"))
            # Wrap long desc manually
            words = desc.split(" · ")
            c.drawCentredString(bx + box_w/2, feat_y - 1.3*cm, " · ".join(words[:2]))
            if len(words) > 2:
                c.drawCentredString(bx + box_w/2, feat_y - 1.7*cm, " · ".join(words[2:]))

        # ── Disclaimer ───────────────────────────────────────────────────
        disc_y = 2.0 * cm
        c.setFillColor(MID_GREY)
        c.setFont("Helvetica-Oblique", 7.5)
        c.drawCentredString(
            W / 2, disc_y,
            "Generated automatically by ML Reliability Auditor. "
            "Interpret metrics alongside domain expertise."
        )
        c.drawCentredString(
            W / 2, disc_y - 0.4 * cm,
            "Not a substitute for comprehensive model governance."
        )

        # ── Bottom accent bar ────────────────────────────────────────────
        c.setFillColor(BRAND_DARK)
        c.rect(0, 0, W, 1.2 * cm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#7fb3d3"))
        c.setFont("Helvetica", 7)
        c.drawCentredString(W / 2, 0.45 * cm,
                            f"CONFIDENTIAL  ·  {self.report_date}  ·  ML RELIABILITY AUDITOR")
        c.restoreState()


def _cover_page(story, styles, model_name, report_date):
    story.append(_CoverPageFlowable(model_name, report_date))
    story.append(PageBreak())


def _section_header(story, styles, title: str, section_num: int = None):
    label = f"{section_num}. {title}" if section_num else title
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_ACCENT, spaceAfter=4))
    story.append(Paragraph(label, styles["section_heading"]))


def _kv_table(data: list[tuple], col_widths=None):
    """Render a 2-column key/value table."""
    if col_widths is None:
        col_widths = [7 * cm, 10 * cm]
    rows = [[k, v] for k, v in data]
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR",  (0, 0), (0, -1), BRAND_ACCENT),
        ("TEXTCOLOR",  (1, 0), (1, -1), TEXT_DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


def _risk_badge_table(risk_level: str):
    """Coloured risk badge."""
    color = RISK_COLOR_MAP.get(risk_level, RISK_HIGH)
    t = Table([[f"  RISK LEVEL: {risk_level.upper()}  "]], colWidths=[8 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 13),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return t


def _metrics_table(metrics: dict, calibration: dict):
    """Combined performance + calibration table."""
    cm_raw = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    if cm_raw and len(cm_raw) == 2:
        tn = cm_raw[0][0]; fp = cm_raw[0][1]
        fn = cm_raw[1][0]; tp = cm_raw[1][1]
        cm_str = f"TP={tp}, TN={tn}, FP={fp}, FN={fn}"
    else:
        cm_str = "N/A"

    data = [
        ["Metric", "Value", "Interpretation"],
        ["Accuracy", f"{metrics.get('accuracy', 0):.2%}",
         _interp_accuracy(metrics.get('accuracy', 0))],
        ["F1 Score (weighted)", f"{metrics.get('f1', 0):.4f}",
         _interp_f1(metrics.get('f1', 0))],
        ["Brier Score", f"{metrics.get('brier_score', 0):.4f}",
         _interp_brier(metrics.get('brier_score', 0))],
        ["ECE", f"{calibration.get('ece', 0):.4f}",
         _interp_ece(calibration.get('ece', 0))],
        ["MCE", f"{calibration.get('mce', 0):.4f}", "Maximum per-bin calibration error"],
        ["Overconfidence Gap", f"{calibration.get('overconfidence_gap', 0):.4f}",
         "Positive = model is overconfident on average"],
        ["Confusion Matrix", cm_str, ""],
    ]

    col_widths = [5.5 * cm, 3.5 * cm, 8.5 * cm]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), BRAND_ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("ALIGN",         (1, 1), (1, -1), "CENTER"),
    ]))
    return t


def _component_risk_table(component_scores: dict):
    data = [["Component", "Score", "Level"]]
    for comp, info in component_scores.items():
        color = RISK_COLOR_MAP.get(info["level"], RISK_HIGH)
        data.append([comp.replace("_", " ").title(), str(info["score"]), info["level"]])

    col_widths = [7 * cm, 3 * cm, 4.5 * cm]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")

    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ALIGN",         (1, 1), (2, -1), "CENTER"),
    ]
    # Colour code risk levels in last column
    for i, (comp, info) in enumerate(component_scores.items(), start=1):
        c = RISK_COLOR_MAP.get(info["level"], RISK_HIGH)
        style.append(("TEXTCOLOR", (2, i), (2, i), c))
        style.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(
    output_path: str,
    model_name: str,
    metrics: dict,
    calibration: dict,
    risk: dict,
    drift: Optional[dict] = None,
    prediction_drift: Optional[dict] = None,
    reliability_diagram_bytes: Optional[bytes] = None,
    confidence_hist_bytes: Optional[bytes] = None,
    drift_chart_bytes: Optional[bytes] = None,
    n_samples: int = 0,
) -> str:
    """
    Generate the full PDF report and write to output_path.

    Returns the output_path.
    """
    report_date = datetime.now().strftime("%B %d, %Y")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        title=f"ML Reliability Report — {model_name}",
        author="ML Reliability Auditor",
    )

    styles = _styles()
    story = []

    # ── Cover ─────────────────────────────────────────────────────────────
    _cover_page(story, styles, model_name, report_date)

    # ── 1. Executive Summary ───────────────────────────────────────────────
    _section_header(story, styles, "Executive Summary", 1)

    risk_level = risk.get("overall_risk_level", "Unknown")
    story.append(Spacer(1, 4))
    story.append(_risk_badge_table(risk_level))
    story.append(Spacer(1, 10))

    summary_text = (
        f"This report presents the reliability assessment of <b>{model_name}</b> "
        f"evaluated on <b>{n_samples:,}</b> production samples as of <b>{report_date}</b>. "
        "The audit covers four dimensions: predictive performance, probability calibration, "
        "distribution drift, and overall deployment risk."
    )
    story.append(Paragraph(summary_text, styles["body"]))
    story.append(Spacer(1, 6))

    summary_kv = [
        ("Model Name", model_name),
        ("Evaluation Samples", f"{n_samples:,}"),
        ("Overall Risk Level", risk_level),
        ("Weighted Risk Score", f"{risk.get('weighted_score', 0):.3f} / 3.0"),
        ("Accuracy", f"{metrics.get('accuracy', 0):.2%}"),
        ("ECE (Calibration Error)", f"{calibration.get('ece', 0):.4f}"),
        ("Brier Score", f"{metrics.get('brier_score', 0):.4f}"),
        ("Distribution Drift (PSI)",
         f"{drift.get('overall_drift_score', 0):.4f}" if drift else "N/A — No reference data"),
        ("Report Date", report_date),
    ]
    story.append(_kv_table(summary_kv))

    # ── 2. Performance Metrics ─────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Model Performance Metrics", 2)

    perf_intro = (
        "The following metrics evaluate the model's predictive accuracy on the production dataset. "
        "Note that accuracy alone is an insufficient measure of model reliability — "
        "calibration quality and probabilistic accuracy (Brier score) are equally critical "
        "for deployment trustworthiness."
    )
    story.append(Paragraph(perf_intro, styles["body"]))
    story.append(Spacer(1, 8))
    story.append(_metrics_table(metrics, calibration))

    # ── 3. Calibration Analysis ─────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Calibration Analysis", 3)

    cal_intro = (
        "Calibration measures whether a model's predicted probabilities reflect true empirical frequencies. "
        "A perfectly calibrated model predicts 70% probability means the event occurs 70% of the time. "
        "Poor calibration leads to systematically over- or under-confident decisions, "
        "which is especially dangerous in high-stakes applications."
    )
    story.append(Paragraph(cal_intro, styles["body"]))

    story.append(Paragraph("Key Calibration Metrics", styles["subsection_heading"]))
    cal_data = [
        ("ECE (Expected Calibration Error)",
         f"{calibration.get('ece', 0):.4f}  —  {_interp_ece(calibration.get('ece', 0))}"),
        ("MCE (Maximum Calibration Error)",
         f"{calibration.get('mce', 0):.4f}  —  Worst single-bin error"),
        ("Overconfidence Gap",
         f"{calibration.get('overconfidence_gap', 0):.4f}  —  "
         f"{'Model is overconfident' if calibration.get('overconfidence_gap', 0) > 0 else 'Model is underconfident'}"),
        ("Number of Calibration Bins", str(calibration.get('n_bins', 10))),
    ]
    story.append(_kv_table(cal_data))

    if reliability_diagram_bytes:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Reliability Diagram", styles["subsection_heading"]))
        rel_diag = Paragraph(
            "Each bar represents a bin of predicted probabilities. "
            "Red bars indicate overconfident bins (predicted confidence exceeds observed accuracy). "
            "Green bars indicate underconfident bins. The diagonal dashed line represents perfect calibration.",
            styles["body"],
        )
        story.append(rel_diag)
        img = Image(io.BytesIO(reliability_diagram_bytes), width=14 * cm, height=12 * cm)
        story.append(img)
        story.append(Paragraph("Figure 1: Reliability Diagram — Calibration Analysis", styles["caption"]))

    if confidence_hist_bytes:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Confidence Score Distribution", styles["subsection_heading"]))
        story.append(Paragraph(
            "Distribution of predicted confidence scores across all production samples. "
            "A bimodal distribution near 0 and 1 indicates a decisive model; "
            "a flat or centered distribution may indicate low-confidence predictions.",
            styles["body"],
        ))
        img2 = Image(io.BytesIO(confidence_hist_bytes), width=14 * cm, height=8 * cm)
        story.append(img2)
        story.append(Paragraph("Figure 2: Confidence Score Histogram", styles["caption"]))

    # ── 4. Distribution Drift ──────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Distribution Shift Analysis", 4)

    if drift is None and prediction_drift is None:
        story.append(Paragraph(
            "No reference dataset was provided. Distribution shift analysis could not be performed. "
            "To enable drift detection, upload a reference (training) dataset alongside the production data.",
            styles["body"],
        ))
    else:
        drift_intro = (
            "Distribution shift occurs when the statistical properties of production data differ "
            "from the training distribution. This is one of the leading causes of model performance "
            "degradation in production. Two metrics are used: "
            "PSI (Population Stability Index) quantifies overall shift magnitude, "
            "while KL Divergence measures information-theoretic distance between distributions."
        )
        story.append(Paragraph(drift_intro, styles["body"]))

        if prediction_drift:
            story.append(Paragraph("Prediction Probability Drift", styles["subsection_heading"]))
            pd_data = [
                ("KL Divergence (prod ∥ ref)",
                 f"{prediction_drift.get('kl_divergence', 0):.4f}  —  {prediction_drift.get('kl_interpretation', '')}"),
                ("PSI (Predicted Probabilities)",
                 f"{prediction_drift.get('psi', 0):.4f}  —  {prediction_drift.get('psi_interpretation', '')}"),
            ]
            story.append(_kv_table(pd_data))

        if drift and drift.get("n_features", 0) > 0:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Feature-Level Drift Summary", styles["subsection_heading"]))
            feat_data = [
                ("Features Analysed", str(drift.get("n_features", 0))),
                ("Overall Drift Score (PSI)",
                 f"{drift.get('overall_drift_score', 0):.4f}  —  {drift.get('overall_interpretation', '')}"),
            ]
            story.append(_kv_table(feat_data))

            # Per-feature table
            features = drift.get("features", {})
            if features:
                story.append(Spacer(1, 6))
                feat_rows = [["Feature", "PSI", "Mean Shift", "Std Shift", "Interpretation"]]
                for fname, finfo in features.items():
                    feat_rows.append([
                        fname,
                        f"{finfo['psi']:.4f}",
                        f"{finfo['mean_shift']:+.4f}",
                        f"{finfo['std_shift']:+.4f}",
                        finfo["psi_interpretation"][:35] + "..." if len(finfo["psi_interpretation"]) > 35
                        else finfo["psi_interpretation"],
                    ])
                col_w = [3.5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm, 6.5 * cm]
                ft = Table(feat_rows, colWidths=col_w, hAlign="LEFT")
                ft.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
                    ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
                    ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                    ("ALIGN",         (1, 1), (3, -1), "CENTER"),
                ]))
                story.append(ft)

        if drift_chart_bytes:
            story.append(Spacer(1, 10))
            img3 = Image(io.BytesIO(drift_chart_bytes), width=14 * cm, height=9 * cm)
            story.append(img3)
            story.append(Paragraph(
                "Figure 3: Feature PSI Heatmap — Green <0.1 (stable), Orange 0.1–0.2 (moderate), Red >0.2 (significant)",
                styles["caption"],
            ))

    # ── 5. Risk Assessment ─────────────────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Risk Assessment", 5)

    story.append(Spacer(1, 4))
    story.append(_risk_badge_table(risk_level))
    story.append(Spacer(1, 10))

    risk_intro = (
        "The overall risk score is computed as a weighted combination of calibration quality, "
        "predictive performance, and distribution shift magnitude. "
        f"The weighted risk score is <b>{risk.get('weighted_score', 0):.3f}</b> on a scale of 0 (no risk) to 3 (critical)."
    )
    story.append(Paragraph(risk_intro, styles["body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Component Risk Breakdown", styles["subsection_heading"]))
    story.append(_component_risk_table(risk.get("component_scores", {})))

    # ── 6. Recommendations & Verdict ──────────────────────────────────────
    story.append(PageBreak())
    _section_header(story, styles, "Recommendations & Deployment Verdict", 6)

    story.append(Paragraph("Recommendations", styles["subsection_heading"]))
    for rec in risk.get("recommendations", []):
        clean_rec = rec.replace("⚠", "WARNING:").replace("🚨", "CRITICAL:").replace(
            "ℹ", "INFO:").replace("✅", "OK:").replace("📋", "NOTE:")
        story.append(Paragraph(f"• {clean_rec}", styles["bullet"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Deployment Verdict", styles["subsection_heading"]))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=RISK_COLOR_MAP.get(risk_level, RISK_HIGH), spaceAfter=6))

    verdict_text = risk.get("deployment_verdict", "No verdict available.")
    for line in verdict_text.replace("🚨", "").split(". "):
        if line.strip():
            story.append(Paragraph(line.strip() + ".", styles["verdict"]))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_DARK, spaceAfter=8))
    story.append(Paragraph(
        f"Report generated by ML Reliability Auditor  |  {report_date}",
        styles["caption"],
    ))

    # ── Build ───────────────────────────────────────────────────────────────
    def make_canvas(filename, **kwargs):
        return _ReportCanvas(filename, pagesize=A4, model_name=model_name)

    doc.build(story, canvasmaker=make_canvas)
    return output_path


# ---------------------------------------------------------------------------
# Metric interpreters
# ---------------------------------------------------------------------------

def _interp_accuracy(v):
    if v >= 0.90: return "Excellent"
    if v >= 0.80: return "Good"
    if v >= 0.70: return "Acceptable"
    return "Poor — requires investigation"

def _interp_f1(v):
    if v >= 0.90: return "Excellent"
    if v >= 0.80: return "Good"
    if v >= 0.70: return "Moderate"
    return "Poor"

def _interp_brier(v):
    if v <= 0.10: return "Excellent probabilistic accuracy"
    if v <= 0.17: return "Good"
    if v <= 0.22: return "Moderate"
    return "Poor — near-random probabilistic predictions"

def _interp_ece(v):
    if v <= 0.05: return "Well calibrated"
    if v <= 0.10: return "Moderate miscalibration"
    if v <= 0.15: return "Poor calibration"
    return "Severely miscalibrated"