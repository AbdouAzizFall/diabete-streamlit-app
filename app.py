import streamlit as st
import joblib
import numpy as np
from PIL import Image

# -----------------------------
# Titre et image d'introduction
# -----------------------------
st.set_page_config(page_title="Prédiction du diabète", page_icon="💉", layout="centered")

st.title("💉 Application de prédiction du diabète")

# Afficher ton image
img = Image.open("img.png")
st.image(img, width=700)

# Texte d'introduction
st.markdown("""
Bienvenue sur cette application de prédiction du risque de diabète.  
Le diabète est une maladie qui touche de plus en plus de personnes dans le monde, et particulièrement au Sénégal.  
Découvert tôt, il peut être contrôlé et le risque de complications graves peut être réduit.  

Cette application utilise le **Pima Indians Diabetes Dataset** et un modèle de **régression logistique** pour estimer votre risque de diabète en fonction de vos données personnelles.  

Je suis **El Hadji Abdou Aziz Fall**, étudiant en Sciences des données et Applications à l’Université Iba Der Thiam de Thiès.  
Pour me contacter : fallaziz699@gmail.com
""")

st.markdown("---")

# -----------------------------
# Inputs utilisateur
# -----------------------------
st.header("Entrez vos informations")
Pregnancies = st.number_input("Nombre de grossesses (Pregnancies)", 0, 20, 0)
Glucose = st.number_input("Glucose (mg/dL)", 0, 300, 0)
BloodPressure = st.number_input("Tension artérielle (BloodPressure)", 0, 200, 0)
Insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 0)
BMI = st.number_input("IMC (BMI)", 0.0, 70.0, 0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.0)
Age = st.number_input("Âge", 0, 120, 0)

st.markdown("---")

# -----------------------------
# Charger le modèle
# -----------------------------
model = joblib.load("diabetes_model.pkl")

# -----------------------------
# Fonction pour détecter facteurs de risque
# -----------------------------
def get_facteurs_risque(Glucose, BMI, Age, Insulin, Pregnancies, DiabetesPedigreeFunction, BloodPressure):
    facteurs = []
    if Glucose > 140:
        facteurs.append("Glucose élevé")
    if BMI > 25:
        facteurs.append("IMC élevé")
    if Age > 45:
        facteurs.append("Âge supérieur à la moyenne")
    if Insulin < 50:
        facteurs.append("Insulinémie basse")
    if Pregnancies > 3:
        facteurs.append("Nombre de grossesses élevé")
    if DiabetesPedigreeFunction > 1:
        facteurs.append("Histoire familiale de diabète")
    if BloodPressure > 130:
        facteurs.append("Hypertension")
    return facteurs

# -----------------------------
# Prédiction et affichage
# -----------------------------
if st.button("Prédire"):
    # ORDRE EXACT attendu par le modèle
    X = np.array([[ 
        Pregnancies,
        Insulin,
        BMI,
        Age,
        Glucose,
        BloodPressure,
        DiabetesPedigreeFunction
    ]])

    # Probabilité pour la prédiction
    prob = model.predict_proba(X)[0][1]  # probabilité de diabète
    prob_percent = round(prob * 100, 2)

    # Déterminer le niveau de risque
    if prob < 0.2:
        niveau_risque = "Faible"
        st.success(f"✅ Risque faible de diabète ({prob_percent}%)")
        st.info("Continuez vos bonnes habitudes : alimentation équilibrée, activité physique régulière, suivi médical annuel.")
    elif prob < 0.5:
        niveau_risque = "Modéré"
        st.warning(f"⚠️ Risque modéré de diabète ({prob_percent}%)")
        st.info("Il est conseillé de consulter un médecin pour un bilan complet et d’adopter un mode de vie plus sain : réduction du sucre, activité physique régulière, suivi régulier.")
    else:
        niveau_risque = "Élevé"
        st.error(f"❌ Risque élevé de diabète ({prob_percent}%)")
        st.info("Prenez rendez-vous avec un professionnel de santé rapidement pour un diagnostic précis et envisagez des changements urgents dans votre mode de vie.")

    # Afficher les facteurs contribuant
    facteurs = get_facteurs_risque(Glucose, BMI, Age, Insulin, Pregnancies, DiabetesPedigreeFunction, BloodPressure)
    if facteurs:
        st.subheader("🔎 Facteurs contribuant à votre risque :")
        st.write(", ".join(facteurs))
    else:
        st.subheader("👍 Aucun facteur de risque majeur détecté pour vos données.")

# -----------------------------
# Bonus : Section sur le diabète
# -----------------------------
st.markdown("---")
st.header("ℹ️ À propos du diabète")
st.markdown("""
Le diabète est une maladie chronique qui survient lorsque le corps ne peut pas réguler correctement le taux de sucre dans le sang.  
Au Sénégal, il touche un nombre croissant de personnes et peut entraîner de graves complications cardiovasculaires, rénales et visuelles si non traité.  

Cependant, une **détection précoce** permet de réduire le risque de complications et d'améliorer significativement la qualité de vie grâce à :  
- Une alimentation équilibrée  
- Une activité physique régulière  
- Un suivi médical adapté  
""")

st.markdown("---")
st.header("💡 Notes sur le modèle")
st.markdown("""
- Dataset utilisé : **Pima Indians Diabetes Dataset**  
- Modèle : **Régression logistique**  
- Variables clés : Glucose, BMI, Age, Insulin, Pression artérielle, Antécédents familiaux  

Cette application est un outil éducatif et **ne remplace pas un diagnostic médical officiel**.
""")
