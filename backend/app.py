from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
from datetime import datetime
import shap

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML-powered fraud detection with rule engine and SHAP explainability.",
    version="5.0.0"
)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Load Model
# -------------------------
try:
    model = joblib.load("credit_fraud.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# -------------------------
# SHAP Explainer
# -------------------------
try:
    explainer = shap.TreeExplainer(model)
    logging.info("SHAP explainer initialized.")
except Exception as e:
    logging.warning(f"SHAP initialization failed: {e}")
    explainer = None

# -------------------------
# Root Endpoint
# -------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Fraud Detection API is running."

# -------------------------
# Health Endpoint
# -------------------------
@app.get("/health")
def health():
    return {"status": "API running"}

# -------------------------
# Request Schema
# -------------------------
class FraudDetection(BaseModel):
    step: int = Field(..., ge=0)
    types: int = Field(..., ge=0, le=4)
    amount: float = Field(..., ge=0)
    oldbalanceorig: float = Field(..., ge=0)
    newbalanceorig: float = Field(..., ge=0)
    oldbalancedest: float = Field(..., ge=0)
    newbalancedest: float = Field(..., ge=0)
    isflaggedfraud: int = Field(..., ge=0, le=1)

# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(data: FraudDetection):

    try:
        # -------------------------
        # Validate Negatives
        # -------------------------
        if any([
            data.amount < 0,
            data.oldbalanceorig < 0,
            data.newbalanceorig < 0,
            data.oldbalancedest < 0,
            data.newbalancedest < 0
        ]):
            raise HTTPException(status_code=400, detail="Negative values not allowed.")

        # -------------------------
        # Feature Array
        # -------------------------
        features = np.array([[ 
            data.step,
            data.types,
            data.amount,
            data.oldbalanceorig,
            data.newbalanceorig,
            data.oldbalancedest,
            data.newbalancedest,
            data.isflaggedfraud
        ]])

        if not hasattr(model, "predict_proba"):
            raise HTTPException(status_code=500, detail="Model does not support probability prediction.")

        # -------------------------
        # ML Probability
        # -------------------------
        ml_probability = float(model.predict_proba(features)[0][1])
        ml_probability = round(ml_probability, 4)

        threshold = 0.4

        # -------------------------
        # Risk Level
        # -------------------------
        if ml_probability > 0.75:
            risk_level = "High Risk"
        elif ml_probability > threshold:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # -------------------------
        # Rule Engine
        # -------------------------
        rule_flags = []
        tolerance = 0.01

        if data.amount == 0:
            rule_flags.append("Zero amount transaction")

        if data.amount > 110000:
            rule_flags.append("Unusually large transaction")

        if abs(data.oldbalanceorig - data.amount - data.newbalanceorig) > tolerance:
            rule_flags.append("Sender balance mismatch")

        if abs(data.newbalancedest - (data.oldbalancedest + data.amount)) > tolerance:
            rule_flags.append("Receiver balance mismatch")

        if data.isflaggedfraud == 1:
            rule_flags.append("Bank system high-risk flag")

        # -------------------------
        # Final Decision Logic
        # -------------------------
        if ml_probability > 0.75:
            final_decision = "Fraud"

        elif ml_probability > threshold:
            final_decision = "Manual Review Required"

        elif len(rule_flags) > 0:
            final_decision = "Manual Review Required"

        else:
            final_decision = "Genuine"

        # -------------------------
        # SHAP Explanation
        # -------------------------
        shap_summary = {}
        if explainer:
            shap_values = explainer.shap_values(features)
            feature_names = [
                "step", "types", "amount", "oldbalanceorig",
                "newbalanceorig", "oldbalancedest",
                "newbalancedest", "isflaggedfraud"
            ]

            shap_summary = {
                feature_names[i]: round(float(shap_values[1][0][i]), 4)
                for i in range(len(feature_names))
            }

        # -------------------------
        # Logging
        # -------------------------
        logging.info(
            f"Probability: {ml_probability} | Risk: {risk_level} | "
            f"Decision: {final_decision} | Flags: {rule_flags}"
        )

        return {
            "ml_probability": ml_probability,
            "risk_level": risk_level,
            "threshold": threshold,
            "rule_flags": rule_flags,
            "final_decision": final_decision,
            "shap_summary": shap_summary,
            "timestamp_utc": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))