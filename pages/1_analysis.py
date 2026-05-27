import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import matplotlib as plt
import shap 
# Ensure Python can find our backend modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.inference.predictor import ChurnPredictor
import database.db_manager as db
from src.utils import charts

# --- 0. PAGE SETUP & THEME ---
st.set_page_config(page_title="Analysis Dashboard", page_icon="📊", layout="wide")

# Custom CSS for that premium SaaS look (Light gray background, crisp white containers)
st.markdown("""
    <style>
    /* You can still add custom CSS here later if needed, but avoid forcing background colors */
    </style>
""", unsafe_allow_html=True)

st.title("📊 Churn Risk Analysis")
st.markdown("Identify high-value accounts at risk and track portfolio health.")

# --- 1. LOAD PREDICTOR ---
@st.cache_resource
def load_predictor():
    return ChurnPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"⚠️ Model not found. Ensure models/ churn_model_rf.pkl exists. Error: {e}")
    st.stop()

# --- 2. GLOBAL CONTROL (Sidebar) ---
st.sidebar.header("Batch Configuration")
data_folder = os.path.join(project_root, 'data', 'processed', 'inference')
print(f"Looking for CSV files in: {data_folder}")  # Debugging line to verify path

from datetime import datetime

if os.path.exists(data_folder):
    # 1. Get all CSV files
    raw_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # 2. Define a helper function to turn the filename into a real date
    def get_date_from_filename(filename):
        # Turns "batch_september_2024.csv" into a datetime object
        clean_name = filename.replace('batch_', '').replace('.csv', '')
        try:
            return datetime.strptime(clean_name, "%B_%Y")
        except ValueError:
            return datetime.min # If naming is weird, throw it at the beginning
            
    # 3. Sort the list of files chronologically using the helper function
    available_files = sorted(raw_files, key=get_date_from_filename)
else:
    available_files = []

if not available_files:
    st.warning(f"⚠️ No CSV files found in {data_folder}.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Month to Evaluate", available_files)
# Clean up the filename for display/logging (e.g., "batch_month_21.csv" -> "Month 21")
display_month = selected_file.replace('batch_', '').replace('.csv', '').replace('_', ' ').title()

# --- 3. DATA PROCESSING & DB LOGGING ---
@st.cache_data
@st.cache_data
def process_and_log_batch(filename, month_label):
    file_path = os.path.join(data_folder, filename)
    df_raw = pd.read_csv(file_path)
    
    df_raw.columns = df_raw.columns.str.strip()
    
    df_raw.columns = df_raw.columns.str.lower()
    
    # Run predictions in memory for the UI
    results_df = predictor.predict_batch(df_raw)
    
    # 2. Prevent Duplicate Database Entries
    existing_logs = db.get_historical_logs(snapshot_month=month_label, prediction_type="BATCH")
    if existing_logs.empty:
        db.log_batch_predictions(results_df, month_label)
        
    return results_df

# Execute the pipeline
df_results = process_and_log_batch(selected_file, display_month)

# --- 4. HISTORICAL TREND (Broadcast Chart) ---
with st.expander("📈 View Historical Churn Trend", expanded=True):
    # Pull ALL batch logs from the database to see the long-term trend
    all_history = db.get_historical_logs(prediction_type="BATCH")
    
    if not all_history.empty:
        trend_df = all_history.groupby('snapshot_month').apply(
            lambda x: (len(x[x['risk_level'] == 'High Risk']) / len(x)) * 100,
            include_groups=False
        ).reset_index(name='Churn_Rate_Pct')
        
        # FIX 3: Force chronological sorting by converting the month string to a date
        try:
            trend_df['Sort_Date'] = pd.to_datetime(trend_df['snapshot_month'])
            trend_df = trend_df.sort_values('Sort_Date')
        except:
            trend_df = trend_df.sort_values('snapshot_month') # Fallback
        
        fig_trend = px.line(trend_df, x='snapshot_month', y='Churn_Rate_Pct', markers=True,
                            title="Portfolio Churn Risk Over Time (%)",
                            labels={'snapshot_month': 'Month', 'Churn_Rate_Pct': 'Churn Risk (%)'})
        fig_trend.update_traces(line_color='#e74c3c', marker=dict(size=8))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No historical data available yet.")

st.markdown("---")
st.subheader(f"Snapshot: {display_month}")

# --- 5. KPI SUMMARY ROW ---
total_customers = len(df_results)
high_risk_df = df_results[df_results['Risk_Level'] == 'High Risk']
high_risk_count = len(high_risk_df)
churn_rate = (high_risk_count / total_customers) * 100
avg_prob = df_results['Churn_Probability'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers Evaluated", f"{total_customers:,}")
col2.metric("High Risk Accounts", f"{high_risk_count:,}")
col3.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Avg Portfolio Risk", f"{avg_prob:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. DYNAMIC RISK SEGMENTATION ---
st.markdown("---")
st.subheader("🔍 Deep Dive Segmentation")

df_viz = df_results.copy()

segment_options = {
    "Overall Risk Tier": ("Risk_Level", "categorical"), 
    "Industry": ("industry", "categorical"),
    "Country / Region": ("country", "categorical"),
    "Plan Tier": ("primary_plan", "categorical"),
    "Total MRR ($)": ("total_mrr", "numerical"),          
    "Customer Tenure (Days)": ("tenure_days", "numerical"),
    "Total Seats": ("total_seats", "numerical"),
    "Total Usage (30 Days)": ("total_usage_count_30d", "numerical")
}

selected_view = st.selectbox("Select a dimension to analyze:", list(segment_options.keys()))
col_name, data_type = segment_options[selected_view]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if data_type == "categorical":
        if col_name == "Risk_Level":
            fig1 = charts.plot_categorical_distribution(df_viz, 'Risk_Level')
            fig1.update_traces(marker_color=['#2ecc71', '#f1c40f', '#e74c3c']) 
        else:
            fig1 = charts.plot_categorical_distribution(df_viz, col_name)
    else:
        fig1 = charts.plot_numerical_distribution(df_viz, col_name)
    
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    if data_type == "categorical":
        if col_name == "Risk_Level":
            # FIX 2: Revenue at Risk Chart (Instead of the MRR scatter plot)
            rev_risk = df_viz.groupby('Risk_Level')['total_mrr'].sum().reset_index()
            fig2 = px.bar(rev_risk, x='Risk_Level', y='total_mrr', color='Risk_Level',
                          color_discrete_map=charts.RISK_COLORS, text_auto='.2s',
                          category_orders={"Risk_Level": ["High Risk", "Medium Risk", "Low Risk"]},
                          title="Revenue at Risk by Tier ($)")
            fig2.update_layout(xaxis_title="", yaxis_title="Total MRR ($)", showlegend=False)
        else:
            fig2 = charts.plot_categorical_risk(df_viz, col_name)
    else:
        fig2 = charts.plot_numerical_risk(df_viz, col_name)
        
    st.plotly_chart(fig2, use_container_width=True)

# --- 7. HIGH-RISK ACTION LIST ---
st.markdown("---")
st.subheader("🚨 High-Risk Action List")
st.markdown("Accounts requiring immediate Customer Success intervention. Sorted by risk severity.")

# Display top high/medium risk accounts
action_list = df_results[df_results['Risk_Level'].isin(['High Risk', 'Medium Risk'])].copy()
action_list = action_list.sort_values(by='Churn_Probability', ascending=False)

# Select impactful columns for the business user
display_cols = ['account_id', 'Churn_Probability', 'Risk_Level', 'total_mrr', 'tenure_days', 'days_since_last_action', 'tickets_last_30d']
existing_cols = [col for col in display_cols if col in action_list.columns]

st.dataframe(
    action_list[existing_cols].style.format({
        'Churn_Probability': '{:.3f}', 
        'total_mrr': '${:,.2f}'
    }),
    width="stretch",
    hide_index=True
)