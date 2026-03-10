import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Financial Fraud Detection System",
    layout="wide"
)

# ----------------------------
# Custom CSS (Unified Theme)
# ----------------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: #e2e8f0; }
.block-container { padding-top: 2rem; }
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white; border-radius: 12px;
    height: 3em; font-weight: bold;
    border: none; transition: 0.3s ease-in-out;
}
.stButton>button:hover { transform: scale(1.05); }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Risk Weights
# ----------------------------
RISK_WEIGHTS = {
    "interbank": 0.06,
    "high_amount": 0.08,
    "late_night": 0.05,
    "full_drain": 0.10,
    "backend_flag": 0.07,
    "rtgs_violation": 0.12,
    "upi_threshold": 0.09
}

# ----------------------------
# Bank Mapping
# ----------------------------
bank_map = {
    "SBIN": "State Bank of India",
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "KKBK": "Kotak Mahindra Bank"
}

def get_bank_name(ifsc):
    if not ifsc or len(ifsc) < 4:
        return "Unknown Bank"
    return bank_map.get(ifsc[:4].upper(), "Unknown Bank")

# ----------------------------
# Header
# ----------------------------
st.title("🏦 Financial Fraud Detection System")
st.caption("A fraud detection engine combining machine learning probability scoring with contextual transaction analysis, dynamic percentage weighting, and explainable risk modeling.")
st.markdown("---")

# ----------------------------
# Transaction Input
# ----------------------------
st.subheader("Transaction Input")
col1, col2 = st.columns(2)

with col1:
    sender_acc = st.text_input("Sender Account Number", help="9 to 18 digits")
    sender_ifsc = st.text_input("Sender IFSC Code", placeholder="e.g., HDFC0000123")
    
    mode = st.selectbox(
        "Transaction Mode", 
        ["IMPS (Instant Transfer)", "UPI / QR Code", "NEFT (Batch Transfer)", "RTGS (High Value)", "Merchant Payment"]
    )
    
    if mode == "UPI / QR Code":
        receiver_vpa = st.text_input("Receiver VPA (UPI ID)", placeholder="merchant@upi")
        receiver_acc = "000000000"
        receiver_ifsc = "UPI00000000"
    else:
        receiver_acc = st.text_input("Receiver Account Number", help="9 to 18 digits")
        receiver_ifsc = st.text_input("Receiver IFSC Code", placeholder="e.g., ICIC0000456")
    
    mode_to_ml_type = {
        "IMPS (Instant Transfer)": 2, 
        "UPI / QR Code": 2,           
        "NEFT (Batch Transfer)": 2,   
        "RTGS (High Value)": 2,       
        "Merchant Payment": 3         
    }
    types = mode_to_ml_type.get(mode, 2)
    
    step = st.slider("Transaction Time (Hour of Day)", 0, 23, 12)

with col2:
    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=1.0)
    oldbalanceorig = st.number_input("Sender Balance (Before)", min_value=0.0, step=1.0)
    newbalanceorig = st.number_input("Sender Balance (After)", min_value=0.0, step=1.0)
    
    oldbalancedest = 0.0  
    newbalancedest = oldbalancedest + amount

st.markdown("---")

# ----------------------------
# Analyze Transaction
# ----------------------------
if st.button("🚀 Analyze Transaction"):

    errors = []
    if not (sender_acc.isdigit() and 9 <= len(sender_acc) <= 18):
        errors.append("Sender Account Number must be between 9 and 18 digits.")
    if len(sender_ifsc) != 11:
        errors.append("Sender IFSC must be exactly 11 characters.")

    if mode == "UPI / QR Code":
        if "@" not in receiver_vpa:
            errors.append("Please enter a valid VPA (must contain '@').")
    else:
        if not (receiver_acc.isdigit() and 9 <= len(receiver_acc) <= 18):
            errors.append(f"Receiver Account Number (9-18 digits) is required for {mode}.")
        if len(receiver_ifsc) != 11:
            errors.append(f"Receiver IFSC (11 characters) is required for {mode}.")
        if sender_acc == receiver_acc and sender_acc != "":
            errors.append("Sender and Receiver accounts cannot be the same.")

    if amount <= 0: errors.append("Transaction amount must be greater than 0.")
    if oldbalanceorig < amount: errors.append("Insufficient balance in sender account.")
    if abs(oldbalanceorig - amount - newbalanceorig) > 0.01: 
        errors.append("Sender balance mismatch: Before/After values don't align with amount.")
    
    if errors:
        for e in errors: st.error(e)
        st.stop()

    sender_bank = get_bank_name(sender_ifsc)
    receiver_bank = "UPI / Merchant" if mode == "UPI / QR Code" else get_bank_name(receiver_ifsc)
    
    is_interbank = (sender_bank != receiver_bank) if mode != "UPI / QR Code" else True
    is_full_drain = newbalanceorig == 0
    is_high_amount = amount > 200000
    is_late_night = 0 <= step <= 4
    
    is_rtgs_violation = (mode == "RTGS (High Value)" and amount < 200000)
    is_upi_high_risk = (mode == "UPI / QR Code" and amount > 100000)

    input_data = {
        "step": step, "types": types, "amount": amount,
        "oldbalanceorig": oldbalanceorig, "newbalanceorig": newbalanceorig,
        "oldbalancedest": oldbalancedest, "newbalancedest": newbalancedest,
        "isflaggedfraud": 0
    }
    
    rail_risks = {
        "IMPS (Instant Transfer)": 0.14,
        "UPI / QR Code": 0.18,
        "NEFT (Batch Transfer)": 0.06,
        "RTGS (High Value)": 0.09,
        "Merchant Payment": 0.04
    }

    try:
        response = requests.post("https://financial-fraud-detection-system-production.up.railway.app/predict", json=input_data, timeout=5)
        result = response.json()

        shap_summary = result.get("shap_summary", {})
        if shap_summary:
            ml_feature_names = list(shap_summary.keys())
            ml_shap_values = list(shap_summary.values())
        else:
            ml_feature_names = []
            ml_shap_values = []

        base_score = (result["ml_probability"] * 0.7) + (rail_risks.get(mode, 0.10) * 0.3)

    except:
        mode_base = rail_risks.get(mode, 0.10)
        base_score = min(mode_base + (amount / 3000000) + (abs(step - 12) * 0.003), 0.45)
        result = {"ml_probability": base_score}
        ml_shap_values = []
        ml_feature_names = []

    risk_score = base_score
    risk_breakdown = [("ML Rail Baseline", round(base_score, 3))]
    risk_categories = []

    conditions = [
        (is_interbank, "interbank", "Interbank Exposure", "Cross-Bank Risk"),
        (is_high_amount, "high_amount", "High Amount", "Value Risk"),
        (is_late_night, "late_night", "Late Night Activity", "Temporal Risk"),
        (is_full_drain, "full_drain", "Full Balance Drain", "Balance Depletion Risk"),
        (is_rtgs_violation, "rtgs_violation", "RTGS Protocol Anomaly", "Protocol Violation"),
        (is_upi_high_risk, "upi_threshold", "UPI Limit Warning", "Velocity Risk")
    ]
    
    for cond, key, label, category in conditions:
        if cond:
            impact = RISK_WEIGHTS[key] * (1 - risk_score)
            risk_score += impact
            risk_breakdown.append((label, round(impact, 3)))
            risk_categories.append(category)

    final_score = min(round(risk_score * 100, 1), 100.0)
    
    if final_score < 30:
        decision = "Genuine"
    elif final_score < 75: 
        decision = "Manual Review Required"
    else:
        decision = "Fraud"

    st.subheader("Transaction Classification")
    st.info(f"🔁 {mode}: {sender_bank} ➜ {receiver_bank}")

    st.subheader("Fraud Risk Score")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        number={'suffix': "%"},
        gauge={'axis': {'range': [0, 100]},
               'steps': [{'range': [0, 30], 'color': "#00c896"},
                         {'range': [30, 75], 'color': "#f4b400"},
                         {'range': [75, 100], 'color': "#ff4b4b"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Machine Learning Explainability (SHAP)")
    st.caption("Red features increase fraud probability, Green features decrease it.")

    if ml_shap_values:

    # 🔹 Feature Label Mapping (Raw → User Friendly)
        FEATURE_NAME_MAP = {
            "amount": "Transaction Amount",
            "oldbalanceorig": "Sender Balance (Before)",
            "newbalanceorig": "Sender Balance (After)",
            "oldbalancedest": "Recipient Balance (Before)",
            "newbalancedest": "Recipient Balance (After)",
            "step": "Transaction Time (Hour of Day)",
            "types": "Transaction Type",
            "isflaggedfraud": "Historical Fraud Indicator"
        }

        shap_df = pd.DataFrame({
            "Feature": ml_feature_names,
            "Impact": ml_shap_values
        })

        # Apply user-friendly names
        shap_df["Feature"] = shap_df["Feature"].map(FEATURE_NAME_MAP).fillna(shap_df["Feature"])
        shap_df = shap_df[~shap_df["Feature"].isin([
            "Recipient Balance (Before)",
            "Recipient Balance (After)",
            "Historical Fraud Indicator"
            ])]

        # Sort by absolute impact (strongest drivers first)
        shap_df["AbsImpact"] = shap_df["Impact"].abs()
        shap_df = shap_df.sort_values(by="AbsImpact", ascending=True)

        # Color logic
        shap_df["Color"] = [
        "#ff4b4b" if x > 0 else "#00c896"
        for x in shap_df["Impact"]
        ]

        fig_shap = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation='h',
            color="Color",
            color_discrete_map="identity",
            title="Top Drivers Influencing Fraud Probability"
        )

        fig_shap.update_layout(showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)

    st.subheader("📊 Risk Contribution Breakdown")
    breakdown_df = pd.DataFrame(risk_breakdown, columns=["Factor", "Weight"])
    breakdown_df["Contribution %"] = (breakdown_df["Weight"] / breakdown_df["Weight"].sum() * 100).round(2)
    fig_bars = px.bar(breakdown_df, x="Contribution %", y="Factor", orientation='h',
                      text="Contribution %", color="Contribution %", color_continuous_scale="Reds")
    fig_bars.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_bars, use_container_width=True)

    if decision == "Genuine": st.success("✅ Legitimate Transaction")
    elif decision == "Manual Review Required": st.warning("⚠ Manual Review Required")
    else: st.error("🚨 High-Risk Fraudulent Activity")

    if risk_categories:
        st.subheader("🏷 Risk Categories Identified")
        for cat in set(risk_categories): st.write(f"• {cat}")

    st.subheader("⚠ Risk Triggers")

    def show_trigger(text, level):
        color = {"critical": "#ff4b4b", "moderate": "#f4b400", "info": "#00c8ff"}.get(level, "#e2e8f0")
        st.markdown(f"<span style='color:{color}'>• {text}</span>", unsafe_allow_html=True)

    trigger_count = 0

    if is_rtgs_violation:
        show_trigger("RTGS Protocol Violation: Amount below ₹2 Lakh limit.", "critical")
        trigger_count += 1

    if is_full_drain:
        show_trigger("Full account balance drained.", "critical")
        trigger_count += 1

    if is_high_amount: 
        show_trigger("High transaction amount detected.", "moderate")
        trigger_count += 1

    if is_interbank:
        show_trigger("Interbank exposure (Cross-institutional).", "moderate")
        trigger_count += 1

    if is_upi_high_risk:
        show_trigger("UPI transaction exceeds standard individual safety limit.", "moderate")
        trigger_count += 1

    if is_late_night:
        show_trigger("Late-night activity detected.", "info")
        trigger_count += 1

    # 👇 THIS IS THE IMPORTANT PART
    if trigger_count == 0:
        st.success("No risk triggers activated. Transaction behavior appears normal.")
    if decision == "Genuine":
        st.subheader("🔎 Why This Was Considered Safe")
        st.write("• No extreme behavioral anomalies detected.")
        st.write("• Transaction aligns with standard balance thresholds.")
        st.write("• Model probability remained within safe range.")

        st.caption(f"Evaluation Timestamp: {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y | %I:%M:%S %p')}")