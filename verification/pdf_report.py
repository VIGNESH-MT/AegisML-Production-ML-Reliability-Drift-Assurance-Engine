from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.lib import fonts

import json


# =====================================================
# Language Dictionary (12 Languages Supported)
# =====================================================

LANGUAGE_CONTENT = {

    "English": {
        "title": "AI Risk & Governance Audit Report",
        "executive_summary": "Executive Summary",
        "technical_section": "Technical Risk Assessment",
        "integrity_section": "Report Integrity",
        "verified": "The AI system has passed drift, bias, and governance validation checks.",
        "failed": "The AI system exhibits significant risk indicators requiring immediate investigation."
    },

    "German": {
        "title": "KI Risiko- und Governance-Prüfbericht",
        "executive_summary": "Zusammenfassung für Führungsebene",
        "technical_section": "Technische Risikoanalyse",
        "integrity_section": "Berichtsintegrität",
        "verified": "Das KI-System hat die Drift-, Bias- und Governance-Prüfungen erfolgreich bestanden.",
        "failed": "Das KI-System weist erhebliche Risikoindikatoren auf."
    },

    "French": {
        "title": "Rapport d'Audit de Risque et Gouvernance IA",
        "executive_summary": "Résumé Exécutif",
        "technical_section": "Évaluation Technique des Risques",
        "integrity_section": "Intégrité du Rapport",
        "verified": "Le système d'IA a réussi les contrôles de dérive et de biais.",
        "failed": "Le système d'IA présente des indicateurs de risque significatifs."
    },

    "Spanish": {
        "title": "Informe de Auditoría de Riesgo y Gobernanza IA",
        "executive_summary": "Resumen Ejecutivo",
        "technical_section": "Evaluación Técnica de Riesgos",
        "integrity_section": "Integridad del Informe",
        "verified": "El sistema de IA ha superado las validaciones de riesgo.",
        "failed": "El sistema de IA presenta indicadores de riesgo significativos."
    },

    "Italian": {
        "title": "Rapporto di Audit su Rischio e Governance IA",
        "executive_summary": "Sintesi Esecutiva",
        "technical_section": "Valutazione Tecnica del Rischio",
        "integrity_section": "Integrità del Rapporto",
        "verified": "Il sistema IA ha superato le verifiche di rischio.",
        "failed": "Il sistema IA mostra indicatori di rischio significativi."
    },

    "Polish": {
        "title": "Raport Audytu Ryzyka i Nadzoru AI",
        "executive_summary": "Podsumowanie Wykonawcze",
        "technical_section": "Techniczna Ocena Ryzyka",
        "integrity_section": "Integralność Raportu",
        "verified": "System AI przeszedł walidację ryzyka.",
        "failed": "System AI wykazuje istotne wskaźniki ryzyka."
    },

    "Romanian": {
        "title": "Raport de Audit privind Riscul și Guvernanța AI",
        "executive_summary": "Rezumat Executiv",
        "technical_section": "Evaluare Tehnică a Riscului",
        "integrity_section": "Integritatea Raportului",
        "verified": "Sistemul AI a trecut validările de risc.",
        "failed": "Sistemul AI prezintă indicatori semnificativi de risc."
    },

    "Dutch": {
        "title": "AI Risico- en Governance Auditrapport",
        "executive_summary": "Managementsamenvatting",
        "technical_section": "Technische Risicobeoordeling",
        "integrity_section": "Rapportintegriteit",
        "verified": "Het AI-systeem heeft de risicovalidatie doorstaan.",
        "failed": "Het AI-systeem vertoont significante risicofactoren."
    },

    "Russian": {
        "title": "Отчет по аудиту рисков и управления ИИ",
        "executive_summary": "Исполнительное резюме",
        "technical_section": "Техническая оценка риска",
        "integrity_section": "Целостность отчета",
        "verified": "Система ИИ успешно прошла проверки риска.",
        "failed": "Система ИИ демонстрирует значительные риски."
    },

    "Ukrainian": {
        "title": "Звіт з аудиту ризиків та управління ШІ",
        "executive_summary": "Виконавче резюме",
        "technical_section": "Технічна оцінка ризику",
        "integrity_section": "Цілісність звіту",
        "verified": "Система ШІ пройшла перевірку ризиків.",
        "failed": "Система ШІ демонструє значні ризики."
    },

    "Hindi": {
        "title": "एआई जोखिम और शासन ऑडिट रिपोर्ट",
        "executive_summary": "कार्यकारी सारांश",
        "technical_section": "तकनीकी जोखिम मूल्यांकन",
        "integrity_section": "रिपोर्ट की सत्यता",
        "verified": "एआई प्रणाली ने जोखिम सत्यापन पास किया है।",
        "failed": "एआई प्रणाली में महत्वपूर्ण जोखिम संकेतक पाए गए हैं।"
    },

    "Urdu": {
        "title": "اے آئی رسک اور گورننس آڈٹ رپورٹ",
        "executive_summary": "ایگزیکٹو خلاصہ",
        "technical_section": "تکنیکی رسک جائزہ",
        "integrity_section": "رپورٹ کی سالمیت",
        "verified": "اے آئی نظام نے رسک کی توثیق مکمل کی ہے۔",
        "failed": "اے آئی نظام میں نمایاں رسک کے اشارے موجود ہیں۔"
    }
}


# =====================================================
# PDF Generator
# =====================================================

def generate_pdf(report_json_path,
                 output_path="verification/Audit_Report.pdf",
                 language="English"):

    if language not in LANGUAGE_CONTENT:
        language = "English"

    lang = LANGUAGE_CONTENT[language]

    with open(report_json_path) as f:
        report_data = json.load(f)

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph(lang["title"], styles["Heading1"]))
    elements.append(Spacer(1, 0.4 * inch))

    # Executive Summary
    elements.append(Paragraph(lang["executive_summary"], styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    summary_text = lang["verified"] if report_data["system_status"] == "VERIFIED" else lang["failed"]
    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(Spacer(1, 0.4 * inch))

    # Technical Risk Table
    elements.append(Paragraph(lang["technical_section"], styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [["Case", "Risk Score", "Tier", "Status"]]

    for r in report_data["results"]:
        table_data.append([
            r.get("case", ""),
            str(r.get("risk_score", "")),
            r.get("risk_tier", ""),
            r.get("status", "")
        ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))

    elements.append(table)
    elements.append(PageBreak())

    # Integrity Section
    elements.append(Paragraph(lang["integrity_section"], styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(
        f"Integrity Hash (SHA256): {report_data.get('integrity_hash_sha256', '')}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        f"Verification Timestamp: {report_data.get('verification_timestamp', '')}",
        styles["Normal"]
    ))

    doc.build(elements)

    print(f"Premium Executive PDF generated at: {output_path}")