import streamlit as st
import json
import os
import time

DATA_FILE = "data.json"
CONTROL_FILE = "control.json"

# --- Helper functions ---
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"1": [], "2": []}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_control(cmd):
    with open(CONTROL_FILE, "w") as f:
        json.dump(cmd, f)

# --- Layout ---
st.set_page_config(page_title="Exhibit Tracker", layout="wide")
st.title("🎥 Exhibit Tracker Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("▶ Start/Resume Tracking"):
    save_control({"action": "start"})
if st.sidebar.button("⏸ Pause Tracking"):
    save_control({"action": "pause"})
if st.sidebar.button("🔁 Reset Data"):
    save_control({"action": "reset"})
if st.sidebar.button("📦 Exit YOLO Process"):
    save_control({"action": "exit"})

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["Overview", "Zone 1", "Zone 2"])

placeholder = st.empty()
refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 0.5, 5.0, 1.0)

# --- Live loop ---
while True:
    data = load_data()
    zone1 = data.get("1", [])
    zone2 = data.get("2", [])
    avg1 = round(sum(zone1) / len(zone1), 1) if zone1 else 0
    avg2 = round(sum(zone2) / len(zone2), 1) if zone2 else 0

    with tab1:
        st.metric("Zone 1 Avg Stay (s)", avg1)
        st.metric("Zone 2 Avg Stay (s)", avg2)
        st.line_chart({"Zone 1": zone1, "Zone 2": zone2})

    with tab2:
        st.subheader("Zone 1 History")
        st.line_chart(zone1)
        st.write(zone1)

    with tab3:
        st.subheader("Zone 2 History")
        st.line_chart(zone2)
        st.write(zone2)

    time.sleep(refresh_rate)
    st.rerun()


# Use python -m streamlit run dashboard.py