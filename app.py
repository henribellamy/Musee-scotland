import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Scottish Museum · Visitor Forecast",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0f1117;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0f1117 60%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -100px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.45);
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #1a1f35;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    height: 3px; width: 100%;
}
.metric-card.blue::after { background: linear-gradient(90deg, #63b3ed, transparent); }
.metric-card.green::after { background: linear-gradient(90deg, #68d391, transparent); }
.metric-card.orange::after { background: linear-gradient(90deg, #f6ad55, transparent); }
.metric-card.red::after { background: linear-gradient(90deg, #fc8181, transparent); }
.metric-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.metric-delta {
    font-size: 0.8rem;
    margin-top: 0.4rem;
    font-weight: 500;
}
.delta-up { color: #68d391; }
.delta-down { color: #fc8181; }

/* Panel */
.panel {
    background: #1a1f35;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}
.panel-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* Result */
.result-box {
    background: linear-gradient(135deg, #1a2744, #1a1f35);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-number {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
}
.result-label {
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-top: 0.5rem;
}
.status-pill {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 1rem;
}
.status-high { background: rgba(104,211,145,0.15); color: #68d391; border: 1px solid rgba(104,211,145,0.3); }
.status-mid { background: rgba(246,173,85,0.15); color: #f6ad55; border: 1px solid rgba(246,173,85,0.3); }
.status-low { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }

/* Sliders & widgets dark override */
div[data-testid="stSlider"] > div { padding: 0; }
label[data-testid="stWidgetLabel"] p { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; }
div[data-testid="stSelectbox"] label p { color: rgba(255,255,255,0.6) !important; font-size: 0.85rem !important; }

.stToggle label { color: rgba(255,255,255,0.6) !important; }
footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
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

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🏛 Visitor Intelligence</div>
    <div class="hero-title">National Museum<br>of Scotland</div>
    <div class="hero-subtitle">Edinburgh · Real-time visitor forecast</div>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-grid">
    <div class="metric-card blue">
        <div class="metric-label">Moyenne quotidienne</div>
        <div class="metric-value">312</div>
        <div class="metric-delta delta-up">↑ +8% vs année passée</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Pic weekend</div>
        <div class="metric-value">520</div>
        <div class="metric-delta delta-up">↑ Samedi</div>
    </div>
    <div class="metric-card orange">
        <div class="metric-label">Pic estival</div>
        <div class="metric-value">580</div>
        <div class="metric-delta delta-up">↑ Juillet–Août</div>
    </div>
    <div class="metric-card red">
        <div class="metric-label">Creux hivernal</div>
        <div class="metric-value">140</div>
        <div class="metric-delta delta-down">↓ Janvier</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="panel"><div class="panel-title">Paramètres du jour</div>', unsafe_allow_html=True)
    jour = st.selectbox("Jour de la semaine", options=list(range(7)),
        format_func=lambda x: JOURS[x])
    mois = st.slider("Mois", 1, 12, 7)
    st.caption(f"**{MOIS[mois-1]}**")
    temperature = st.slider("Température (°C)", 2, 22, 15)
    c1, c2 = st.columns(2)
    with c1:
        est_vacances = st.toggle("🎒 Vacances scolaires")
    with c2:
        est_pluvieux = st.toggle("🌧 Temps pluvieux")
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction
    input_df = pd.DataFrame([{
        "jour_semaine": jour, "mois": mois,
        "est_vacances": int(est_vacances),
        "temperature_c": temperature,
        "est_pluvieux": int(est_pluvieux)
    }])
    prediction = int(model.predict(input_df)[0])

    if prediction > 400:
        status_class, status_text, result_color = "status-high", "🟢 Forte affluence", "#68d391"
    elif prediction > 220:
        status_class, status_text, result_color = "status-mid", "🟡 Affluence modérée", "#f6ad55"
    else:
        status_class, status_text, result_color = "status-low", "🔴 Faible affluence", "#fc8181"

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Visiteurs estimés</div>
        <div class="result-number" style="color:{result_color}">{prediction}</div>
        <div><span class="{status_class} status-pill">{status_text}</span></div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel"><div class="panel-title">Prévisions sur l\'année</div>', unsafe_allow_html=True)

    preds_mois = []
    for m in range(1, 13):
        row = pd.DataFrame([{
            "jour_semaine": jour, "mois": m,
            "est_vacances": int(est_vacances),
            "temperature_c": temperature,
            "est_pluvieux": int(est_pluvieux)
        }])
        preds_mois.append(int(model.predict(row)[0]))

    colors = ["rgba(99,179,237,0.35)" if i+1 != mois else "#63b3ed" for i in range(12)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MOIS, y=preds_mois,
        marker_color=colors,
        marker_line_width=0,
        text=preds_mois,
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.5)", size=11),
    ))
    fig.add_hline(y=312, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                  annotation_text="moyenne", annotation_font_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(tickfont=dict(color="rgba(255,255,255,0.4)", size=11), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(tickfont=dict(color="rgba(255,255,255,0.3)", size=10), gridcolor="rgba(255,255,255,0.04)", showgrid=True),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap jour x mois
    st.markdown('<div class="panel"><div class="panel-title">Heatmap — Affluence par jour & mois</div>', unsafe_allow_html=True)
    heat = np.zeros((7, 12))
    for j in range(7):
        for m in range(1, 13):
            row = pd.DataFrame([{
                "jour_semaine": j, "mois": m,
                "est_vacances": int(est_vacances),
                "temperature_c": temperature,
                "est_pluvieux": int(est_pluvieux)
            }])
            heat[j, m-1] = int(model.predict(row)[0])

    fig2 = go.Figure(go.Heatmap(
        z=heat, x=MOIS, y=JOURS,
        colorscale=[[0, "#1a1f35"], [0.5, "#2b4a8a"], [1, "#63b3ed"]],
        showscale=False,
        hovertemplate="%{y} · %{x}<br>%{z} visiteurs<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
        yaxis=dict(tickfont=dict(color="rgba(255,255,255,0.4)", size=10)),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
