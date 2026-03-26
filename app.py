import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ---------- Page Configuration ----------
st.set_page_config(page_title="Antigravity Simulation System", page_icon="🚀", layout="wide")

# ---------- Data Generation & Model Training ----------
@st.cache_resource
def initialize_ai_models():
    """Generate synthetic physics data and train ML models"""
    data = []
    g = 9.81
    
    # Generate balanced synthetic dataset
    for _ in range(1000):
        mass = random.uniform(1.0, 100.0)         # Mass in kg
        ext_force = random.uniform(0.0, 2000.0)     # External upward force in N
        init_vel = random.uniform(-50.0, 50.0)      # Initial velocity in m/s (up is positive)
        time_dur = random.uniform(1.0, 20.0)        # Time duration in s
        
        # Calculate acceleration: a = (F_ext - mg) / m
        accel = (ext_force - mass * g) / mass
        
        # Determine label: 1 if Antigravity (overcoming gravity, a > 0), 0 if Normal Gravity
        if random.random() < 0.1: # Inject some noise
            label = random.randint(0, 1)
        else:
            label = 1 if accel > 0 else 0
            
        data.append([mass, ext_force, init_vel, time_dur, label])

    df = pd.DataFrame(data, columns=["mass", "ext_force", "init_vel", "time_dur", "antigravity"])

    # Features and Labels
    X = df[["mass", "ext_force", "init_vel", "time_dur"]]
    y = df["antigravity"]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------- Train Random Forest ----------
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_scaled, y)

    # ---------- Train ANN ----------
    ann_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    ann_model.fit(X_scaled, y)
    
    return scaler, rf_model, ann_model

# Load models with spinner
with st.spinner("Initializing AI Physics Models..."):
    scaler, rf_model, ann_model = initialize_ai_models()

# ---------- UI Layout ----------
st.title("🚀 Antigravity Simulation & Visualization System")
st.markdown("""
Welcome to the **Antigravity Simulator**. This tool combines classical physics and Machine Learning to model 
the kinematics of an object and classify its state as **Normal Gravity** or **Antigravity**.
""")

# Sidebar inputs
st.sidebar.header("⚙️ Simulation Parameters")
mass = st.sidebar.number_input("Mass (kg)", min_value=0.1, max_value=1000.0, value=10.0, step=0.5)
ext_force = st.sidebar.number_input("External Upward Force (N)", min_value=0.0, max_value=5000.0, value=50.0, step=10.0)
init_vel = st.sidebar.number_input("Initial Velocity (m/s) [Up is +ve]", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)
time_dur = st.sidebar.number_input("Simulation Time (s)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

model_choice = st.sidebar.selectbox(
    "🧠 Select AI Model for Classification",
    ["Random Forest", "Neural Network"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Physics Note:** Earth's gravitational acceleration is assumed to be $g = 9.81 m/s^2$.")

# Main columns
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("🤖 AI Analysis & Logic")
    
    # Calculate Ground Truth Physics
    g = 9.81
    weight = mass * g
    net_force = ext_force - weight
    accel = net_force / mass
    
    st.markdown("### 🔍 Force Breakdown")
    st.write(f"- **Gravitational Pull (Weight)**: `{weight:.2f} N` (Downwards)")
    st.write(f"- **External Upward Force**: `{ext_force:.2f} N`")
    st.write(f"- **Net Force ($F_{{net}}$)**: `{net_force:.2f} N`")
    st.write(f"- **Acceleration ($a$)**: `{accel:.2f} m/s²`")
    
    st.markdown("---")
    
    if st.button("Predict State using AI", use_container_width=True):
        # Prepare input data for ML model
        input_data = np.array([[mass, ext_force, init_vel, time_dur]])
        input_scaled = scaler.transform(input_data)
        
        # Predict using selected model
        if model_choice == "Random Forest":
            pred = rf_model.predict(input_scaled)
        elif model_choice == "Neural Network":
            pred = ann_model.predict(input_scaled)
            
        # Display Results
        if pred[0] == 1:
            st.success("✨ **AI Classification: ANTIGRAVITY DETECTED!** ✨\nThe system state shows forces overcoming Earth's gravity.")
        else:
            st.error("🌍 **AI Classification: NORMAL GRAVITY.**\nThe object remains bound by gravitational pull.")

with col2:
    st.header("📈 Real-time Physics Simulation")
    
    # Generate time array
    t = np.linspace(0, time_dur, num=500)
    
    # Kinematic equations
    v = init_vel + accel * t
    s = init_vel * t + 0.5 * accel * t**2
    accel_array = np.full_like(t, accel)
    
    # Create interactive Plotly figure
    fig = go.Figure()
    
    # Position trace
    fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name='Position (m)', line=dict(color='royalblue', width=3)))
    
    # Velocity trace
    fig.add_trace(go.Scatter(x=t, y=v, mode='lines', name='Velocity (m/s)', line=dict(color='firebrick', width=3)))
    
    # Acceleration trace
    fig.add_trace(go.Scatter(x=t, y=accel_array, mode='lines', name='Acceleration (m/s²)', line=dict(color='green', width=3, dash='dash')))
    
    fig.update_layout(
        title="Kinematics Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Magnitude",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 📚 Theoretical Concept")
st.markdown("""
The **Antigravity Simulation and Visualization System** aims to visualize the classical mechanics of levitation and directed upward propulsion. By utilizing Newton's Second Law:

$$F_{net} = m \\cdot a$$

where $F_{net} = F_{external} - m \\cdot g$. 

If $F_{external} > m \\cdot g$, then the acceleration is positive, meaning the object is ascending and overcoming standard gravity. The AI models (Random Forest and Neural Network) are trained on these parameters to classify the condition autonomously, acting as intelligent monitors for hypothetical flight or hover systems.
""")
