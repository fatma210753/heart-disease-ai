import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MediAssist", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7fb 0%, #f7fbff 50%, #e6f2ff 100%);
}

.main-title {
    text-align: center;
    color: #1976d2;
    font-size: 3.4rem;
    font-weight: 800;
}

.sub-title {
    text-align: center;
    color: #546e7a;
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
}

.section-title {
    color: #1565c0;
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 1rem;
}

.result-card {
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 1rem;
}

.red-card { background:#ffebee; color:#c62828; }
.green-card { background:#e8f5e9; color:#2e7d32; }
.orange-card { background:#fff3e0; color:#ef6c00; }

.text-box {
    background:#ffffffcc;
    border-radius:12px;
    padding:12px;
    border:1px solid #dbe8f4;
    color:#37474f;
}

.footer {
    text-align:center;
    color:#78909c;
    margin-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("heart_disease_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='main-title'>💙 MediAssist - Heart Disease AI System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered clinical diagnosis support tool</div>", unsafe_allow_html=True)

# -----------------------------
# Helper functions
# -----------------------------
def rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca):
    reasons = []

    if age >= 55:
        reasons.append("Older age")
    if cp == 0:
        reasons.append("Typical angina chest pain pattern")
    if trestbps >= 140:
        reasons.append("High blood pressure")
    if chol >= 240:
        reasons.append("High cholesterol level")
    if exang == 1:
        reasons.append("Exercise-induced angina")
    if oldpeak >= 2:
        reasons.append("Significant ST depression (oldpeak)")
    if ca >= 1:
        reasons.append("Blockage in major blood vessels")

    return reasons

def risk_from_probability(p):
    if p >= 0.75:
        return "High Risk", "red-card"
    elif p >= 0.45:
        return "Moderate Risk", "orange-card"
    else:
        return "Low Risk", "green-card"

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-title'>🩺 Patient Clinical Details</div>", unsafe_allow_html=True)

    age = st.number_input("Age (years)", 1, 120, 50)

    sex = st.selectbox(
        "Biological Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ][x]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        50, 250, 120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        50, 600, 200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    restecg = st.selectbox(
        "Resting Electrocardiographic Results",
        [0, 1, 2],
        format_func=lambda x: [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ][x]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        50, 250, 150
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        0.0, 10.0, 1.0
    )

    slope = st.selectbox(
        "Slope of the ST Segment",
        [0, 1, 2],
        format_func=lambda x: [
            "Upsloping",
            "Flat",
            "Downsloping"
        ][x]
    )

    ca = st.selectbox(
        "Number of Major Blood Vessels (0–4)",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia Test Result",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Unknown",
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ][x]
    )

    predict = st.button("🔍 Predict Diagnosis")

# -----------------------------
# Output
# -----------------------------
with col2:
    st.markdown("<div class='section-title'>📊 Diagnosis Result</div>", unsafe_allow_html=True)

    if predict:

        df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        for col in model_columns:
            if col not in df:
                df[col] = 0

        df = df[model_columns]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        # Prediction
        if pred == 1:
            st.markdown(f"<div class='result-card red-card'>⚠️ Heart Disease Detected<br>Confidence: {prob*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card green-card'>✅ No Heart Disease Detected<br>Confidence: {(1-prob)*100:.2f}%</div>", unsafe_allow_html=True)

        # Risk
        risk_label, risk_class = risk_from_probability(prob)
        st.markdown(f"<div class='result-card {risk_class}'>Risk Level: {risk_label}</div>", unsafe_allow_html=True)

        # Clinical Indicators (NO LINK ICON)
        st.markdown("<div class='section-title'>💡 Clinical Indicators</div>", unsafe_allow_html=True)
        reasons = rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca)
        st.markdown(f"<div class='text-box'>{', '.join(reasons) if reasons else 'No major indicators detected.'}</div>", unsafe_allow_html=True)

        # Recommendation (NO LINK ICON)
        st.markdown("<div class='section-title'>🩺 Recommendation</div>", unsafe_allow_html=True)

        if prob >= 0.75:
            rec = "Immediate clinical consultation is strongly recommended."
        elif prob >= 0.45:
            rec = "Further medical examination is advised."
        else:
            rec = "Maintain a healthy lifestyle and schedule regular check-ups."

        st.markdown(f"<div class='text-box'>{rec}</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed for AI Medical Diagnosis Coursework</div>", unsafe_allow_html=True)
