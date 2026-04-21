import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MediAssist", layout="wide")

# -----------------------------
# Custom medical theme styling
# -----------------------------
st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #f8fcff 45%, #e3f2fd 100%);
    color: #1f2d3d;
}

/* Remove extra top spacing a bit */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
.main-title {
    text-align: center;
    color: #1976d2;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #4f5b67;
    font-size: 1.35rem;
    font-weight: 500;
    margin-bottom: 1.8rem;
}

/* Section cards */
.card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #d9ebf7;
    border-radius: 22px;
    padding: 24px 22px;
    box-shadow: 0 8px 24px rgba(25, 118, 210, 0.10);
    backdrop-filter: blur(6px);
    margin-bottom: 1rem;
}

/* Section headings */
.section-title {
    color: #1565c0;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

/* Small info pill */
.badge {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #bbdefb;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(90deg, #1e88e5, #42a5f5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    font-size: 1.05rem;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(30, 136, 229, 0.25);
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1976d2, #2196f3);
    color: white;
}

/* Input boxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 12px !important;
}

/* Result cards */
.result-box {
    border-radius: 18px;
    padding: 18px 20px;
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

.result-positive {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
}

.result-negative {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}

.risk-high {
    background: #ffebee;
    color: #b71c1c;
    border: 1px solid #ef9a9a;
}

.risk-medium {
    background: #fff3e0;
    color: #ef6c00;
    border: 1px solid #ffcc80;
}

.risk-low {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
}

/* Footer */
.footer {
    text-align: center;
    color: #607d8b;
    margin-top: 1.5rem;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model files
# -----------------------------
model = joblib.load("heart_disease_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.markdown("<div class='main-title'>💙 MediAssist - Heart Disease AI System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered medical diagnosis support tool with machine learning and clinical reasoning</div>", unsafe_allow_html=True)

# -----------------------------
# Rule engine
# -----------------------------
def rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca):
    score = 0
    reasons = []

    if age >= 55:
        score += 1
        reasons.append("older age")
    if cp == 0:
        score += 2
        reasons.append("chest pain risk")
    if trestbps >= 140:
        score += 1
        reasons.append("high blood pressure")
    if chol >= 240:
        score += 1
        reasons.append("high cholesterol")
    if exang == 1:
        score += 2
        reasons.append("exercise-induced angina")
    if oldpeak >= 2:
        score += 2
        reasons.append("high oldpeak")
    if ca >= 1:
        score += 2
        reasons.append("major vessels affected")

    if score >= 7:
        return "High Risk", reasons, "risk-high"
    elif score >= 4:
        return "Moderate Risk", reasons, "risk-medium"
    else:
        return "Low Risk", reasons, "risk-low"

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.08, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='badge'>Patient Input Panel</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🩺 Patient Information</div>", unsafe_allow_html=True)

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
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='badge'>Decision Support Output</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Diagnosis Result</div>", unsafe_allow_html=True)

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

        risk, reasons, risk_class = rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca)

        if prediction == 1:
            st.markdown(
                f"<div class='result-box result-positive'>⚠️ Heart Disease Detected<br><br>Confidence: {probability*100:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box result-negative'>✅ No Heart Disease Detected<br><br>Confidence: {(1-probability)*100:.2f}%</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<div class='result-box {risk_class}'>Risk Level: {risk}</div>",
            unsafe_allow_html=True
        )

        st.markdown("### 💡 AI Explanation")
        st.write(", ".join(reasons) if reasons else "No major risk factors detected.")

        st.markdown("### 🩺 Recommendation")
        if risk == "High Risk":
            st.error("Immediate clinical consultation is strongly recommended.")
        elif risk == "Moderate Risk":
            st.warning("Further medical examination is advised.")
        else:
            st.success("Maintain a healthy lifestyle and attend regular medical check-ups.")
    else:
        st.info("Enter the patient information and click Predict Diagnosis to view the AI result.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Developed for AI Medical Diagnosis Coursework</div>", unsafe_allow_html=True)
