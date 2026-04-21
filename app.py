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
    color: #1f2d3d;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    color: #1976d2;
    font-size: 3.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #546e7a;
    font-size: 1.35rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

.section-title {
    color: #1565c0;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.small-note {
    color: #607d8b;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.3rem;
    font-size: 1.05rem;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(30, 136, 229, 0.22);
}

div.stButton > button:hover {
    color: white;
    background: linear-gradient(90deg, #1976d2, #2196f3);
}

.result-card {
    border-radius: 18px;
    padding: 18px 20px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.6rem;
    margin-bottom: 1rem;
}

.red-card {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
}

.green-card {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}

.orange-card {
    background: #fff3e0;
    color: #ef6c00;
    border: 1px solid #ffcc80;
}

.blue-card {
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #90caf9;
}

.footer {
    text-align: center;
    color: #78909c;
    margin-top: 1.8rem;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load files
# -----------------------------
model = joblib.load("heart_disease_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='main-title'>💙 MediAssist - Heart Disease AI System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered medical diagnosis support tool with machine learning and clinical reasoning</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Helper functions
# -----------------------------
def rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca):
    reasons = []

    if age >= 55:
        reasons.append("older age")
    if cp == 0:
        reasons.append("high-risk chest pain pattern")
    elif cp in [2, 3]:
        reasons.append("abnormal chest pain type")
    if trestbps >= 140:
        reasons.append("high blood pressure")
    if chol >= 240:
        reasons.append("high cholesterol")
    if exang == 1:
        reasons.append("exercise-induced angina")
    if oldpeak >= 2:
        reasons.append("high oldpeak")
    elif oldpeak >= 1:
        reasons.append("mild oldpeak elevation")
    if ca >= 1:
        reasons.append("major vessels affected")

    return reasons

def overall_risk_from_probability(prob):
    if prob >= 0.75:
        return "High Risk", "red-card"
    elif prob >= 0.45:
        return "Moderate Risk", "orange-card"
    else:
        return "Low Risk", "green-card"

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown("<div class='section-title'>🩺 Patient Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Enter the patient’s clinical details below.</div>", unsafe_allow_html=True)

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
    chol = st.number_input("Cholesterol", 50, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("ECG Result", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    predict_clicked = st.button("🔍 Predict Diagnosis")

with right:
    st.markdown("<div class='section-title'>📊 Diagnosis Result</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>The final diagnosis is driven by the trained machine learning model.</div>", unsafe_allow_html=True)

    if predict_clicked:
        data = {
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
        }

        df_input = pd.DataFrame([data])

        for col in model_columns:
            if col not in df_input:
                df_input[col] = 0

        df_input = df_input[model_columns]

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        # ML diagnosis
        if prediction == 1:
            st.markdown(
                f"<div class='result-card red-card'>⚠️ Heart Disease Detected<br><br>Model Confidence: {probability*100:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-card green-card'>✅ No Heart Disease Detected<br><br>Model Confidence: {(1-probability)*100:.2f}%</div>",
                unsafe_allow_html=True
            )

        # Overall risk based on model probability
        risk_label, risk_class = overall_risk_from_probability(probability)
        st.markdown(
            f"<div class='result-card {risk_class}'>Risk Level: {risk_label}</div>",
            unsafe_allow_html=True
        )

        # Clinical indicators from rule engine
        reasons = rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca)

        st.markdown("### 💡 Clinical Indicators")
        if reasons:
            st.write(", ".join(reasons))
        else:
            st.write("No major clinical indicators were flagged by the rule-based check.")

        st.markdown("### 🩺 Recommendation")
        if probability >= 0.75:
            st.error("Immediate clinical consultation is strongly recommended.")
        elif probability >= 0.45:
            st.warning("Further medical examination is advised.")
        else:
            st.success("Maintain a healthy lifestyle and attend regular medical check-ups.")

    else:
        st.markdown(
            "<div class='result-card blue-card'>Enter the patient information and click Predict Diagnosis to view the AI result.</div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='footer'>Developed for AI Medical Diagnosis Coursework</div>", unsafe_allow_html=True)
