import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "jour_semaine": np.random.randint(0, 7, n),
        "mois": np.random.randint(1, 13, n),
        "est_vacances": np.random.randint(0, 2, n),
        "temperature_c": np.random.uniform(2, 22, n),
        "est_pluvieux": np.random.randint(0, 2, n),
    })
    df["visiteurs"] = (
        200
        + df["jour_semaine"].apply(lambda x: 150 if x >= 5 else 0)
        + df["est_vacances"] * 300
        + df["temperature_c"] * 5
        - df["est_pluvieux"] * 80
        + df["mois"].apply(lambda x: 100 if x in [6, 7, 8] else 0)
        + np.random.normal(0, 30, n)
    ).clip(50).astype(int)

    features = ["jour_semaine", "mois", "est_vacances", "temperature_c", "est_pluvieux"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df["visiteurs"])
    return model

model = train_model()

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
