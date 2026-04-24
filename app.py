import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_musee.pkl")

st.title("🏛️ Prédiction de visiteurs — Musée en Écosse")
st.markdown("Renseigne les conditions du jour pour estimer le nombre de visiteurs.")

jour = st.selectbox("Jour de la semaine", 
    options=list(range(7)),
    format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x])

mois = st.slider("Mois", 1, 12, 7)

est_vacances = st.checkbox("Période de vacances scolaires")

temperature = st.slider("Température (°C)", 2, 22, 15)

est_pluvieux = st.checkbox("Temps pluvieux")

if st.button("Prédire"):
    input_df = pd.DataFrame([{
        "jour_semaine": jour,
        "mois": mois,
        "est_vacances": int(est_vacances),
        "temperature_c": temperature,
        "est_pluvieux": int(est_pluvieux)
    }])
    prediction = int(model.predict(input_df)[0])
    st.success(f"Visiteurs estimés : **{prediction}**")
