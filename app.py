import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ---------- Page Configuration ----------
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")

# ---------- Data Generation & Model Training ----------
@st.cache_resource
def initialize_ai_models():
    """Generate synthetic fraud data and train ML models"""
    data = []
    
    # Generate synthetic dataset (same logic, renamed)
    for _ in range(1000):
        amount = random.uniform(100, 20000)
        login_attempts = random.uniform(1, 10)
        location_change = random.uniform(0, 1)
        device_change = random.uniform(0, 1)

        # Logic mapping (same structure)
        risk_score = amount + (login_attempts * 1000) + (location_change * 2000) + (device_change * 2000)

        if random.random() < 0.1:
            label = random.randint(0, 1)
        else:
            label = 1 if risk_score > 8000 else 0

        data.append([amount, login_attempts, location_change, device_change, label])

    df = pd.DataFrame(data, columns=["amount", "login_attempts", "location_change", "device_change", "fraud"])

    X = df[["amount", "login_attempts", "location_change", "device_change"]]
    y = df["fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- Random Forest ----------
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_scaled, y)

    # ---------- ANN ----------
    ann_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    ann_model.fit(X_scaled, y)

    return scaler, rf_model, ann_model

# Load models
with st.spinner("Initializing Fraud Detection Models..."):
    scaler, rf_model, ann_model = initialize_ai_models()

# ---------- UI ----------
st.title("🛡️ Prevention of Scams using ML & DL Models")
st.markdown("""
This system detects fraudulent transactions in **E-Commerce platforms** using Machine Learning models.
""")

# Sidebar inputs (ONLY renamed)
st.sidebar.header("⚙️ Transaction Parameters")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=50000.0, value=500.0)
login_attempts = st.sidebar.number_input("Login Attempts", min_value=1.0, max_value=10.0, value=1.0)
location_change = st.sidebar.number_input("Location Change (0 or 1)", min_value=0.0, max_value=1.0, value=0.0)
device_change = st.sidebar.number_input("Device Change (0 or 1)", min_value=0.0, max_value=1.0, value=0.0)

model_choice = st.sidebar.selectbox(
    "🧠 Select AI Model",
    ["Random Forest", "Neural Network"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Higher transaction amount and unusual behavior increase fraud risk.")

# Layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("🤖 AI Fraud Analysis")

    risk_score = amount + (login_attempts * 1000) + (location_change * 2000) + (device_change * 2000)

    st.markdown("### 🔍 Risk Breakdown")
    st.write(f"- Transaction Amount: `{amount:.2f}`")
    st.write(f"- Login Attempts: `{login_attempts}`")
    st.write(f"- Location Change: `{location_change}`")
    st.write(f"- Device Change: `{device_change}`")
    st.write(f"- Calculated Risk Score: `{risk_score:.2f}`")

    st.markdown("---")

    if st.button("Check Transaction", use_container_width=True):

        input_data = np.array([[amount, login_attempts, location_change, device_change]])
        input_scaled = scaler.transform(input_data)

        if model_choice == "Random Forest":
            pred = rf_model.predict(input_scaled)
        else:
            pred = ann_model.predict(input_scaled)

        if pred[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate")

with col2:
    st.header("📊 Fraud Risk Visualization")

    t = np.linspace(0, 10, num=100)
    risk_trend = risk_score * (t / max(t))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=risk_trend, mode='lines', name='Risk Growth', line=dict(width=3)))

    fig.update_layout(
        title="Fraud Risk Trend",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 📚 Concept")
st.markdown("""
This project uses Machine Learning models like **Random Forest** and **Neural Networks** 
to detect fraudulent transactions based on user behavior and transaction patterns.
""")

 
