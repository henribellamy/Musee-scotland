import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(
    page_title="National Museum of Scotland · Forecast",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #ffffff; }
.block-container { padding: 3rem 4rem; max-width: 1200px; }

.top-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.4rem;
}
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #111;
    line-height: 1.15;
    margin: 0 0 0.3rem 0;
}
.page-sub {
    font-size: 0.95rem;
    color: #999;
    font-weight: 300;
    margin: 0;
}

.divider { border: none; border-top: 1px solid #f0f0f0; margin: 2rem 0; }

.kpi-row { display: flex; gap: 0; margin-bottom: 2.5rem; border: 1px solid #f0f0f0; border-radius: 12px; overflow: hidden; }
.kpi-item { flex: 1; padding: 1.5rem 2rem; border-right: 1px solid #f0f0f0; }
.kpi-item:last-child { border-right: none; }
.kpi-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #bbb; margin-bottom: 0.5rem; }
.kpi-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #111; }
.kpi-note { font-size: 0.75rem; color: #bbb; margin-top: 0.25rem; }

.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 1.25rem;
}

.result-wrap {
    border: 1px solid #f0f0f0;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-num {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    line-height: 1;
    margin: 0;
}
.result-sub { font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; color: #bbb; margin-top: 0.5rem; }
.pill {
    display: inline-block;
    margin-top: 1rem;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.pill-green { background: #f0faf4; color: #2d8a55; }
.pill-amber { background: #fff8ef; color: #b06a00; }
.pill-red   { background: #fff2f2; color: #b03030; }

label[data-testid="stWidgetLabel"] p { color: #555 !important; font-size: 0.85rem !important; font-weight: 400 !important; }
div[data-testid="stSelectbox"] label p { color: #555 !important; font-size: 0.85rem !important; }
.stToggle label { color: #555 !important; font-size: 0.85rem !important; }
footer, #MainMenu { display: none; }
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
MOIS = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
JOURS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="top-label">National Museum of Scotland · Edinburgh</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Visitor Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Predict daily attendance based on operational conditions</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi-item">
        <div class="kpi-label">Daily average</div>
        <div class="kpi-value">312</div>
        <div class="kpi-note">All year</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-label">Weekend peak</div>
        <div class="kpi-value">520</div>
        <div class="kpi-note">Saturday</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-label">Summer peak</div>
        <div class="kpi-value">580</div>
        <div class="kpi-note">July – August</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-label">Winter low</div>
        <div class="kpi-value">140</div>
        <div class="kpi-note">January</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls + Result ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Conditions</div>', unsafe_allow_html=True)
    jour = st.selectbox("Day of the week", options=list(range(7)), format_func=lambda x: JOURS[x])
    mois = st.slider("Month", 1, 12, 7)
    st.caption(f"{MOIS[mois-1]}")
    temperature = st.slider("Temperature (°C)", 2, 22, 15)
    c1, c2 = st.columns(2)
    with c1:
        est_vacances = st.toggle("School holidays")
    with c2:
        est_pluvieux = st.toggle("Rainy weather")

    input_df = pd.DataFrame([{
        "jour_semaine": jour, "mois": mois,
        "est_vacances": int(est_vacances),
        "temperature_c": temperature,
        "est_pluvieux": int(est_pluvieux)
    }])
    prediction = int(model.predict(input_df)[0])

    if prediction > 400:
        color, pill_class, label = "#2d8a55", "pill-green", "High attendance"
    elif prediction > 220:
        color, pill_class, label = "#b06a00", "pill-amber", "Moderate attendance"
    else:
        color, pill_class, label = "#b03030", "pill-red", "Low attendance"

    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-num" style="color:{color}">{prediction}</div>
        <div class="result-sub">estimated visitors</div>
        <span class="pill {pill_class}">{label}</span>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Monthly forecast</div>', unsafe_allow_html=True)
    preds = []
    for m in range(1, 13):
        row = pd.DataFrame([{"jour_semaine": jour, "mois": m,
            "est_vacances": int(est_vacances), "temperature_c": temperature,
            "est_pluvieux": int(est_pluvieux)}])
        preds.append(int(model.predict(row)[0]))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MOIS, y=preds,
        marker_color=["#111" if i+1 == mois else "#e8e8e8" for i in range(12)],
        marker_line_width=0,
        text=preds, textposition="outside",
        textfont=dict(color="#aaa", size=10),
    ))
    fig.add_hline(y=312, line_dash="dot", line_color="#ddd",
                  annotation_text="avg", annotation_font_color="#ccc", annotation_font_size=10)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=10, b=0, l=0, r=0),
        xaxis=dict(tickfont=dict(color="#bbb", size=11), showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#ccc", size=10), gridcolor="#f5f5f5", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">Attendance heatmap</div>', unsafe_allow_html=True)
    heat = np.zeros((7, 12))
    for j in range(7):
        for m in range(1, 13):
            row = pd.DataFrame([{"jour_semaine": j, "mois": m,
                "est_vacances": int(est_vacances), "temperature_c": temperature,
                "est_pluvieux": int(est_pluvieux)}])
            heat[j, m-1] = int(model.predict(row)[0])

    fig2 = go.Figure(go.Heatmap(
        z=heat, x=MOIS, y=JOURS,
        colorscale=[[0, "#f7f7f7"], [1, "#111111"]],
        showscale=False,
        hovertemplate="%{y} · %{x}<br>%{z} visitors<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210, margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(tickfont=dict(color="#bbb", size=10), showgrid=False),
        yaxis=dict(tickfont=dict(color="#bbb", size=10), showgrid=False),
    )
    st.plotly_chart(fig2, use_container_width=True)
