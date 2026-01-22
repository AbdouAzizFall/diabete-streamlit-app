import streamlit as st
import joblib
import numpy as np

st.title("Application de prédiction du diabète")

# Charger le modèle
model = joblib.load("diabetes_model.pkl")

# Inputs utilisateur
Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
Glucose = st.number_input("Glucose", 0, 300, 0)
BloodPressure = st.number_input("BloodPressure", 0, 200, 0)
Insulin = st.number_input("Insulin", 0, 900, 0)
BMI = st.number_input("BMI", 0.0, 70.0, 0.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.0)
Age = st.number_input("Age", 0, 120, 0)

# Prédiction
if st.button("Prédire"):
    # ORDRE EXACT attendu par le modèle :
    # ['Pregnancies', 'Insulin', 'BMI', 'Age', 
    #  'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
    
    X = np.array([[ 
        Pregnancies,
        Insulin,
        BMI,
        Age,
        Glucose,
        BloodPressure,
        DiabetesPedigreeFunction
    ]])
    
    st.write("Input envoyé au modèle :", X)  # Debug (tu peux enlever plus tard)
    
    pred = model.predict(X)[0]

    if pred == 1:
        st.error("⚠️ Diabète détecté")
    else:
        st.success("✅ Pas de diabète")
