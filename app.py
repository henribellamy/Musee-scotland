import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(
    page_title="Musée d'Écosse — Prédiction visiteurs",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #2c5f8a;
    }
    h1 { color: #1a1a2e; font-weight: 700; }
    h3 { color: #2c5f8a; }
</style>
""", unsafe_allow_html=True)

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

# --- Header ---
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:4rem;text-align:center'>🏛️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("# Musée National d'Écosse")
    st.markdown("<p style='color:#666;margin-top:-10px'>Outil de prédiction du nombre de visiteurs</p>", unsafe_allow_html=True)

st.divider()

# --- Métriques en haut ---
st.markdown("### Statistiques de référence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Visiteurs moyens / jour", "312")
m2.metric("Pic weekend", "520")
m3.metric("Pic été", "580")
m4.metric("Creux hiver", "140")

st.divider()

# --- Inputs + Carte ---
st.markdown("### Paramètres de la prédiction")
left, right = st.columns([2, 1])

with left:
    col1, col2 = st.columns(2)
    with col1:
        jour = st.selectbox("Jour de la semaine",
            options=list(range(7)),
            format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x])
        mois = st.slider("Mois", 1, 12, 7,
            format=lambda x: ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"][x-1])
    with col2:
        temperature = st.slider("Température (°C)", 2, 22, 15)
        est_vacances = st.toggle("Période de vacances scolaires")
        est_pluvieux = st.toggle("Temps pluvieux")

with right:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/National_Museum_of_Scotland_-_2.jpg/640px-National_Museum_of_Scotland_-_2.jpg",
             caption="Musée National d'Écosse, Édimbourg", use_container_width=True)

st.divider()

# --- Prédiction ---
input_df = pd.DataFrame([{
    "jour_semaine": jour,
    "mois": mois,
    "est_vacances": int(est_vacances),
    "temperature_c": temperature,
    "est_pluvieux": int(est_pluvieux)
}])
prediction = int(model.predict(input_df)[0])

st.markdown("### Résultat")
res_col, graph_col = st.columns([1, 2])

with res_col:
    couleur = "#2ecc71" if prediction > 350 else "#e67e22" if prediction > 200 else "#e74c3c"
    st.markdown(f"""
    <div class='metric-card'>
        <p style='color:#666;margin:0;font-size:0.9rem'>Visiteurs estimés</p>
        <p style='font-size:3rem;font-weight:700;color:{couleur};margin:0'>{prediction}</p>
        <p style='color:#666;font-size:0.85rem;margin:0'>{'🟢 Forte affluence' if prediction > 350 else '🟡 Affluence modérée' if prediction > 200 else '🔴 Faible affluence'}</p>
    </div>
    """, unsafe_allow_html=True)

with graph_col:
    # Graphique : prédiction par mois pour les paramètres actuels
    mois_labels = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
    preds_mois = []
    for m in range(1, 13):
        row = pd.DataFrame([{
            "jour_semaine": jour,
            "mois": m,
            "est_vacances": int(est_vacances),
            "temperature_c": temperature,
            "est_pluvieux": int(est_pluvieux)
        }])
        preds_mois.append(int(model.predict(row)[0]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mois_labels,
        y=preds_mois,
        marker_color=["#2c5f8a" if i+1 != mois else "#e67e22" for i in range(12)],
        text=preds_mois,
        textposition="outside"
    ))
    fig.update_layout(
        title="Prédiction par mois (conditions actuelles)",
        xaxis_title="Mois",
        yaxis_title="Visiteurs",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False
    )
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)
