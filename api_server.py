print("=== LOADED CORRECT api_server.py (REAL DATA VERSION v4) ===")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Annotated
import pandas as pd
import uuid
import os
import numpy as np

from verification.metrics import calculate_psi, calculate_ks
from verification.risk_scoring import calculate_risk_score
from verification.governance import map_risk_tier
from verification.verification_runner import generate_verification_report
from verification.pdf_report import generate_pdf


# -------------------------------------------------
# App Initialization
# -------------------------------------------------

app = FastAPI(
    title="AI Risk & Governance Audit API",
    version="4.0.0-enterprise"
)


# -------------------------------------------------
# Root Endpoint
# -------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "AI Risk & Governance Audit API running",
        "version": "4.0.0-enterprise"
    }


# -------------------------------------------------
# Real CSV Upload Audit Endpoint (Enterprise Safe)
# -------------------------------------------------

@app.post("/audit-upload")
async def audit_upload(
    train_file: Annotated[UploadFile, File(..., description="Training CSV file")],
    prod_file: Annotated[UploadFile, File(..., description="Production CSV file")],
    column_name: Annotated[str, Form(..., description="Column to analyze")],
    language: Annotated[str, Form()] = "English",
):
    try:
        # Read CSV safely
        train_df = pd.read_csv(train_file.file)
        prod_df = pd.read_csv(prod_file.file)

        # Column existence validation
        if column_name not in train_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' not found in training data"
            )

        if column_name not in prod_df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' not found in production data"
            )

        # Extract column
        train_series = train_df[column_name].dropna()
        prod_series = prod_df[column_name].dropna()

        # Ensure numeric
        if not np.issubdtype(train_series.dtype, np.number):
            raise HTTPException(
                status_code=400,
                detail="Selected column must be numeric for drift analysis"
            )

        if not np.issubdtype(prod_series.dtype, np.number):
            raise HTTPException(
                status_code=400,
                detail="Selected column must be numeric for drift analysis"
            )

        # Minimum sample size protection
        if len(train_series) < 30 or len(prod_series) < 30:
            raise HTTPException(
                status_code=400,
                detail="Minimum 30 samples required in both datasets"
            )

        train_data = train_series.values
        prod_data = prod_series.values

        # Compute Drift Metrics
        psi = calculate_psi(train_data, prod_data)
        ks_stat, p_value = calculate_ks(train_data, prod_data)

        risk_score = calculate_risk_score(psi, p_value, bias_score=0)
        tier, action = map_risk_tier(risk_score)

        status = "PASS" if risk_score < 70 else "FAIL"

        result = {
            "case": "real_data_audit",
            "psi": round(float(psi), 5),
            "ks_pvalue": round(float(p_value), 5),
            "risk_score": float(risk_score),
            "risk_tier": tier,
            "required_action": action,
            "status": status
        }

        overall_status = status == "PASS"

        # Generate JSON Report
        generate_verification_report([result], overall_status)

        # Unique PDF file
        unique_id = str(uuid.uuid4())[:8]
        pdf_path = f"verification/Real_Audit_{unique_id}.pdf"

        generate_pdf(
            "verification/verification_report.json",
            output_path=pdf_path,
            language=language
        )

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="AI_Real_Data_Audit_Report.pdf"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))