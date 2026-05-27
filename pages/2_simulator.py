import streamlit as st
import pandas as pd
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)

from src.inference.predictor import ChurnPredictor
import database.db_manager as db
from src.utils import explainer

st.set_page_config(page_title="What-If Simulator", page_icon="⚙️", layout="wide")
st.title("⚙️ What-If Simulator & Explainer")

# --- 1. GLOBAL STATE & DATA LOAD ---
data_folder = os.path.join(project_root, 'data', 'processed', 'inference')
available_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')] if os.path.exists(data_folder) else []

# Sync the sidebar with the Analysis page using session_state
if 'selected_month_file' not in st.session_state:
    st.session_state.selected_month_file = available_files[0] if available_files else None

selected_file = st.sidebar.selectbox("Global Batch Context", available_files, 
                                     index=available_files.index(st.session_state.selected_month_file) if st.session_state.selected_month_file in available_files else 0)
st.session_state.selected_month_file = selected_file
display_month = selected_file.replace('batch_', '').replace('.csv', '').replace('_', ' ').title()

@st.cache_resource
def load_predictor(): return ChurnPredictor()

@st.cache_data
def load_data(filename):
    df = pd.read_csv(os.path.join(data_folder, filename))
    df.columns = df.columns.str.strip().str.lower()
    return df
 
predictor = load_predictor()
df_raw = load_data(selected_file)

# --- 2. ACCOUNT SELECTION ---
display_options = df_raw.apply(lambda x: f"{x['account_id']} | MRR: ${x.get('total_mrr', 0)}", axis=1)
selected_display = st.selectbox("Search by Account ID:", display_options)
selected_account_id = selected_display.split(" | ")[0]
base_customer = df_raw[df_raw['account_id'] == selected_account_id].iloc[0].to_dict()

st.markdown("---")

# --- 3. SIMULATOR & EXPLAINER ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("### Adjust Drivers")
    with st.form("what_if_form"):
        # Top 8 Features
        sim_data = {
            'tenure_days': st.number_input("Tenure (Days)", value=int(base_customer.get('tenure_days', 0)), step=30),
            'total_seats': st.number_input("Total Seats", value=int(base_customer.get('total_seats', 1)), step=1),
            'total_mrr': st.number_input("Total MRR ($)", value=int(base_customer.get('total_mrr', 0)), step=100),
            'avg_duration_last_30d': st.number_input("Avg Duration (30d)", value=float(base_customer.get('avg_duration_last_30d', 0.0)), step=10.0),
            'total_usage_count_30d': st.number_input("Total Usage (30d)", value=int(base_customer.get('total_usage_count_30d', 0)), step=10),
            'days_since_last_action': st.slider("Days Since Last Action", 0, 90, int(base_customer.get('days_since_last_action', 0))),
            'total_active_subs': st.number_input("Total Active Subs", value=int(base_customer.get('total_active_subs', 1)), step=1),
            'avg_resolution_last_90d': st.number_input("Avg Resolution Time (90d)", value=float(base_customer.get('avg_resolution_last_90d', 0.0)), step=1.0)
        }
        submitted = st.form_submit_button("Run Simulation & Explain", use_container_width=True)

with col2:
    st.markdown("### Risk Analysis")
    if submitted:
        # Merge simulated data back into the full user row
        simulated_customer = base_customer.copy()
        simulated_customer.update(sim_data)
        
        # Predict using our clean, modular function!
        prob, risk_tier, X_encoded = predictor.predict_single(simulated_customer)
        
        st.markdown(f"**Simulated Risk:** {prob*100:.1f}% ({risk_tier})")
        
        with st.spinner("Generating Explainer..."):
            fig, pushing, preventing = explainer.generate_shap_explanation(predictor.model, X_encoded)
            st.pyplot(fig)
            
            st.markdown("**🔴 Pushing toward Churn:**")
            for _, r in pushing.iterrows(): st.write(f"- {r['Feature']}: {r['Value']} (+{r['Impact']*100:.1f}%)")
            st.markdown("**🟢 Keeping Account Safe:**")
            for _, r in preventing.iterrows(): st.write(f"- {r['Feature']}: {r['Value']} ({r['Impact']*100:.1f}%)")
        
        # Save state for the DB logger
        st.session_state.last_sim = {'prob': prob, 'risk': risk_tier, 'features': sim_data}

# --- 4. SAVE SCENARIO TO DB ---
if 'last_sim' in st.session_state:
    st.markdown("---")
    colA, colB = st.columns([3, 1])
    with colA:
        scenario_title = st.text_input("Scenario Title", value=f"Retention Plan for {selected_account_id}")
    with colB:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Scenario"):
            db.log_prediction(
                account_id=selected_account_id,
                snapshot_month=display_month,
                prediction_type="WHAT_IF",
                probability=st.session_state.last_sim['prob'],
                risk_level=st.session_state.last_sim['risk'],
                features_dict=st.session_state.last_sim['features'],
                title=scenario_title
            )
            st.success("Scenario saved successfully!")
            del st.session_state.last_sim # Clear state after saving

# --- 5. HISTORICAL SCENARIOS ---
st.markdown("---")
st.subheader("Saved Scenarios for this Month")
history_df = db.get_historical_logs(snapshot_month=display_month, prediction_type="WHAT_IF")

if not history_df.empty:
    st.dataframe(history_df[['timestamp', 'account_id', 'title', 'risk_level', 'churn_probability', 'tenure_days', 'total_mrr']], hide_index=True)
else:
    st.info("No saved scenarios found for this month.")