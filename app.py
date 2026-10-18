
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
sex_encoder = joblib.load("heart_sex_encoder.pkl")
cp_encoder = joblib.load("heart_cp_encoder.pkl")

st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
thalach = st.number_input("Maximum heart rate", min_value=50, max_value=250, value=150)

if st.button("Predict"):
    sex_val = 1.0 if sex == "Male" else 0.0
    cp_val = float(["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp) + 1)
    row = pd.DataFrame([[age, sex_encoder.transform([sex_val])[0], cp_encoder.transform([cp_val])[0], thalach]],
                       columns=["age", "sex", "cp", "thalach"])
    if model.predict(row)[0] == 1:
        st.error("Heart disease is likely")
    else:
        st.success("Heart disease is not likely")
