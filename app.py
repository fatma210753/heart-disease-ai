import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_disease_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("MediAssist - Heart Disease Diagnosis Support System")

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
        return "High Risk", reasons
    elif score >= 4:
        return "Moderate Risk", reasons
    else:
        return "Low Risk", reasons

if st.button("Predict"):

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

    risk, reasons = rule_engine(age, cp, trestbps, chol, exang, oldpeak, ca)

    st.subheader("Result")
    st.write("Prediction:", "Heart Disease" if prediction == 1 else "No Disease")
    st.write("Confidence:", f"{probability*100:.2f}%")

    st.subheader("AI Reasoning")
    st.write("Risk Level:", risk)
    st.write("Reasons:", ", ".join(reasons))