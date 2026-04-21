import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="MediAssist", layout="wide")

# Load model
model = joblib.load("heart_disease_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💙 MediAssist - Heart Disease AI System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered medical diagnosis support tool</h4>", unsafe_allow_html=True)

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# INPUT SECTION
with col1:
    st.subheader("🧾 Patient Information")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Blood Pressure", 50, 250, 120)
    chol = st.number_input("Cholesterol", 50, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    restecg = st.selectbox("ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# RULE ENGINE
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
        reasons.append("high BP")
    if chol >= 240:
        score += 1
        reasons.append("high cholesterol")
    if exang == 1:
        score += 2
        reasons.append("exercise angina")
    if oldpeak >= 2:
        score += 2
        reasons.append("high oldpeak")
    if ca >= 1:
        score += 2
        reasons.append("vessels affected")

    if score >= 7:
        return "High Risk", reasons, "red"
    elif score >= 4:
        return "Moderate Risk", reasons, "orange"
    else:
        return "Low Risk", reasons, "green"

# OUTPUT SECTION
with col2:
    st.subheader("📊 Diagnosis Result")

    if st.button("🔍 Predict"):
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

        risk, reasons, color = rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca)

        # Prediction box
        if prediction == 1:
            st.error(f"⚠️ Heart Disease Detected\n\nConfidence: {probability*100:.2f}%")
        else:
            st.success(f"✅ No Heart Disease\n\nConfidence: {(1-probability)*100:.2f}%")

        # Risk level
        if color == "red":
            st.markdown(f"<h3 style='color:red;'>🔴 {risk}</h3>", unsafe_allow_html=True)
        elif color == "orange":
            st.markdown(f"<h3 style='color:orange;'>🟠 {risk}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:green;'>🟢 {risk}</h3>", unsafe_allow_html=True)

        # Explanation
        st.markdown("### 💡 AI Explanation")
        st.write(", ".join(reasons) if reasons else "No major risk factors detected.")

        # Recommendation
        st.markdown("### 🩺 Recommendation")
        if risk == "High Risk":
            st.error("Immediate clinical consultation is strongly recommended.")
        elif risk == "Moderate Risk":
            st.warning("Further medical examination is advised.")
        else:
            st.success("Maintain a healthy lifestyle and regular check-ups.")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed for AI Medical Diagnosis Coursework</p>", unsafe_allow_html=True)
