import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Market Shock Intelligence System",
    layout="wide"
)

# ============================
# Header
# ============================
st.title("Market Shock Intelligence System (MSIS)")
st.caption(
    "Detecting market regimes using unsupervised learning on returns, volatility, and drawdowns."
)

# ============================
# Regime Labels & Colors
# ============================
REGIME_LABELS = {
    0: "Low Volatility / Range-Bound Regime",
    1: "Trend-Driven Growth Regime",
    2: "Crisis / Stress Regime"
}

REGIME_COLORS = {
    0: "#2E86C1",  # Blue
    1: "#28B463",  # Green
    2: "#CB4335"   # Red
}

# ============================
# Sidebar Controls (Clean)
# ============================
st.sidebar.header("Market Controls")

selected_regime = st.sidebar.selectbox(
    "Select Market Regime",
    options=list(REGIME_LABELS.keys()),
    format_func=lambda x: REGIME_LABELS[x]
)

st.sidebar.markdown(
    f"""
    <div style="
        padding: 10px;
        margin-top: 10px;
        border-left: 5px solid {REGIME_COLORS[selected_regime]};
        background-color: #f8f9fa;
    ">
        <strong>Selected Regime</strong><br>
        {REGIME_LABELS[selected_regime]}
    </div>
    """,
    unsafe_allow_html=True
)

# ============================
# Fetch Data from FastAPI
# ============================
API_URL = "http://127.0.0.1:8000/regimes"
data = requests.get(API_URL).json()
df = pd.DataFrame(data)

regime_df = df[df["regime"] == selected_regime]

# ============================
# Regime Title
# ============================
st.markdown(
    f"<h2 style='color:{REGIME_COLORS[selected_regime]}'>"
    f"{REGIME_LABELS[selected_regime]}</h2>",
    unsafe_allow_html=True
)

# ============================
# Regime Metrics
# ============================
col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Daily Return",
    f"{regime_df['return'].mean():.4f}"
)

col2.metric(
    "Average Volatility",
    f"{regime_df['volatility'].mean():.4f}"
)

col3.metric(
    "Average Drawdown",
    f"{regime_df['drawdown'].mean():.4f}"
)

st.caption(
    "Metrics represent averages across all days classified into this regime."
)

# ============================
# Visual Analysis
# ============================
st.subheader("Regime Feature Distributions")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Return Distribution
axes[0].hist(
    regime_df["return"],
    bins=40,
    color=REGIME_COLORS[selected_regime],
    alpha=0.85
)
axes[0].set_title("Return Distribution")
axes[0].set_xlabel("Daily Return")
axes[0].set_ylabel("Frequency")

# Volatility Distribution
axes[1].hist(
    regime_df["volatility"],
    bins=40,
    color=REGIME_COLORS[selected_regime],
    alpha=0.85
)
axes[1].set_title("Volatility Distribution")
axes[1].set_xlabel("Volatility")

# Drawdown Distribution
axes[2].hist(
    regime_df["drawdown"],
    bins=40,
    color=REGIME_COLORS[selected_regime],
    alpha=0.85
)
axes[2].set_title("Drawdown Distribution")
axes[2].set_xlabel("Drawdown")

st.pyplot(fig)

# ============================
# Interpretation (Minimal & Clear)
# ============================
st.subheader("Regime Interpretation")

if selected_regime == 0:
    st.write(
        "This regime reflects stable market conditions with low volatility and limited downside risk."
    )

elif selected_regime == 1:
    st.write(
        "This regime captures sustained upward or downward trends with moderate volatility."
    )

elif selected_regime == 2:
    st.write(
        "This regime represents periods of market stress characterized by sharp volatility spikes and deep drawdowns."
    )
