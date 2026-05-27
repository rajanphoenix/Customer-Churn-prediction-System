import streamlit as st

# PAGE CONFIGURATION (Must be the very first Streamlit command)
st.set_page_config(
    page_title="RavenStack Churn Engine",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM THEME (Matching the Analysis page)
st.markdown("""
    <style>
    /* You can still add custom CSS here later if needed, but avoid forcing background colors */
    </style>
""", unsafe_allow_html=True)

# HOMEPAGE CONTENT
st.title("🦅 RavenStack Customer Success Portal")
st.markdown("---")

st.markdown("""
### Welcome to the Churn Intelligence Engine.
This platform bridges the gap between raw machine learning and operational business strategy. It is designed to help Customer Success Managers identify and prevent churn, while providing MLOps engineers the telemetry needed to ensure long-term model reliability.

**👈 Please select a module from the sidebar to begin:**
""")

st.markdown("<br>", unsafe_allow_html=True)

# Use info boxes for a cleaner, block-level layout
st.info("**📊 Analysis Dashboard:** Run chronological batch predictions, track portfolio health, evaluate account turnover, and export the high-risk watchlist.")

st.info("**⚙️ What-If Simulator:** Dynamically adjust customer drivers to simulate retention strategies, powered by SHAP feature explainability, and log custom intervention plans.")

st.info("**📈 MLOps Monitoring:** Track systemic Data Drift, monitor Prediction Drift (concept drift), and evaluate historical model performance against confirmed billing ground truth.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.success("🟢 System Status: Database, Inference Engine, and Monitoring Telemetry Online. Ready for analysis.")