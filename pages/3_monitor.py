import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Ensure Python can find our backend modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import database.db_manager as db
from src.utils import drift_detection as drift

# --- 0. PAGE SETUP & THEME ---
st.set_page_config(page_title="System Monitoring", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    /* You can still add custom CSS here later if needed, but avoid forcing background colors */
    </style>
""", unsafe_allow_html=True)

st.title("System & Model Monitoring")
st.markdown("Track global data health, model stability, and analyst utilization across all processed months.")

# --- 1. GLOBAL STATE & DATA LOAD ---
st.sidebar.header("Global Context")
data_folder = os.path.join(project_root, 'data', 'processed', 'inference')
ref_data_path = os.path.join(project_root, 'data', 'processed', 'not-inference','train_df.csv')

available_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')] if os.path.exists(data_folder) else []

if not available_files or not os.path.exists(ref_data_path):
    st.warning("Missing inference or training data.")
    st.stop()

# Sync the sidebar with the Analysis page using session_state
if 'selected_month_file' not in st.session_state:
    st.session_state.selected_month_file = available_files[0] if available_files else None

selected_file = st.sidebar.selectbox(
    "Deep Dive Month", 
    available_files, 
    index=available_files.index(st.session_state.selected_month_file) if st.session_state.selected_month_file in available_files else 0
)
st.session_state.selected_month_file = selected_file
display_month = selected_file.replace('batch_', '').replace('.csv', '').replace('_', ' ').title()

# --- 2. ANALYST TELEMETRY (System Health) ---
st.markdown(f"### Analyst Telemetry ({display_month})")

# Pull logs for the selected month
month_logs = db.get_historical_logs(snapshot_month=display_month)

batch_logs = month_logs[month_logs['prediction_type'] == 'BATCH']
what_if_logs = month_logs[month_logs['prediction_type'] == 'WHAT_IF']

total_analyzed = len(batch_logs)
scenarios_saved = len(what_if_logs)
high_risk_found = len(batch_logs[batch_logs['risk_level'] == 'High Risk'])

col1, col2, col3 = st.columns(3)
col1.metric("Accounts Analyzed (Batch)", f"{total_analyzed:,}")
col2.metric("High-Risk Accounts Flagged", f"{high_risk_found:,}")
col3.metric("Retention Scenarios Saved", f"{scenarios_saved:,}")

st.markdown("---")

# --- 3. GLOBAL DATA DRIFT TRACKING ---
st.markdown("### Data Drift Detection")

@st.cache_data
def get_cached_drift_results(ref_path, curr_path):
    """Runs the evidently report and parses it. Caches the heavy math!"""
    raw_dict = drift.generate_drift_report(ref_path, curr_path)
    return drift.parse_drift_results(raw_dict)

# Find out which months the analyst has ACTUALLY processed in the DB
all_db_logs = db.get_historical_logs(prediction_type="BATCH")
if all_db_logs.empty:
    st.info("No batch predictions have been run yet. Go to the Analysis tab to process a month!")
    st.stop()

processed_months = all_db_logs['snapshot_month'].unique()

# Calculate drift for ALL processed months to build the Global Trend
global_drift_history = []
selected_month_drift_df = pd.DataFrame()
selected_month_drift_count = 0
selected_month_total_count = 0

# Progress bar for the first time it loads
progress_text = "Analyzing global data health..."
my_bar = st.progress(0, text=progress_text)

for idx, f in enumerate(available_files):
    month_label = f.replace('batch_', '').replace('.csv', '').replace('_', ' ').title()
    
    # Only calculate drift if the analyst has officially processed this month
    if month_label in processed_months:
        curr_path = os.path.join(data_folder, f)
        d_count, t_count, d_df = get_cached_drift_results(ref_data_path, curr_path)
        
        global_drift_history.append({
            "Month": month_label,
            "Drifted Features": d_count,
            "Total Features": t_count
        })
        
        # Save the specific dataframe for the selected sidebar month
        if f == selected_file:
            selected_month_drift_df = d_df
            selected_month_drift_count = d_count
            selected_month_total_count = t_count
            
    my_bar.progress((idx + 1) / len(available_files), text=progress_text)
my_bar.empty() # Clear the progress bar when done

# 3A. Draw the Global Trend Chart
if len(global_drift_history) > 0:
    trend_df = pd.DataFrame(global_drift_history)
    # Ensure chronological sorting (Hack: convert to datetime if possible, or leave as string)
    try:
        trend_df['Sort_Date'] = pd.to_datetime(trend_df['Month'])
        trend_df = trend_df.sort_values('Sort_Date')
    except:
        pass 
        
    fig = px.bar(trend_df, x="Month", y="Drifted Features", 
                 title="Global Drift Trend: Features Drifting per Month",
                 text="Drifted Features", color="Drifted Features",
                 color_continuous_scale="Reds")
    # 18 is 50% of our 36 total features
    fig.add_hline(y=18, line_dash="dash", line_color="red", annotation_text="Dataset Drift Threshold (50%)")
    st.plotly_chart(fig, use_container_width=True)

# 3B. Draw the Deep Dive Table for the Selected Month
st.markdown(f"#### Deep Dive: {display_month}")

if selected_month_drift_count > 0:
    st.error(f"**ALERT:** {selected_month_drift_count} out of {selected_month_total_count} features drifted in {display_month}.")
    st.dataframe(
        selected_month_drift_df[['Feature', 'Severity', 'Drift Intensity']],
        column_config={
            "Drift Intensity": st.column_config.ProgressColumn(
                "Drift Intensity (0-100)",
                help="How far the data has shifted from the training baseline.",
                format="%d",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.success(f"**NO DRIFT:** All features in {display_month} match the training baseline.")



st.markdown("---")
st.markdown("### Prediction Drift (Concept Drift)")

# 1. Figure out which months we can compare
# (processed_months was defined earlier in the Data Drift section)
available_reference_months = [m for m in processed_months if m != display_month]

if len(processed_months) < 2:
    st.info("Not enough historical data to measure Prediction Drift. Process at least one more month in the Analysis tab to unlock comparisons!")
else:
    # Logic to default to the chronological previous month
    # For now, we just pick the most recent one that isn't the current month
    default_ref_index = 0 
    
    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        reference_month = st.selectbox(
            f"Comparing {display_month} against:", 
            available_reference_months,
            index=default_ref_index
        )

    # 2. Fetch the data from SQLite
    # We already have `batch_logs` for the current month from the Telemetry section!
    ref_logs = db.get_historical_logs(snapshot_month=reference_month, prediction_type="BATCH")

    # 3. Metric 1: The Macro Shift (Average Probability)
    ref_mean, curr_mean, delta = drift.calculate_prediction_summary(ref_logs, batch_logs)
    
    st.markdown(f"**Average Portfolio Risk**")
    st.metric(
        label=f"Mean Churn Probability ({display_month})", 
        value=f"{curr_mean * 100:.1f}%", 
        delta=f"{delta * 100:.1f}% vs {reference_month}",
        delta_color="inverse" # Red is bad (risk went up), Green is good (risk went down)
    )

    # 4. Create charts side-by-side
    col_tier, col_shape = st.columns(2)

    with col_tier:
        st.markdown("**Business Impact: Risk Tier Shift**")
        tier_df = drift.prepare_risk_tier_data(ref_logs, batch_logs, reference_month, display_month)
        
        if not tier_df.empty:
            # Force the order of the stacked bars to be logical
            category_orders = {"Risk Tier": ["High Risk", "Medium Risk", "Low Risk"]}
            color_map = {"High Risk": "#ff4b4b", "Medium Risk": "#ffa500", "Low Risk": "#00cc96"}
            
            fig_tier = px.bar(
                tier_df, x="Month", y="Percentage", color="Risk Tier",
                color_discrete_map=color_map, category_orders=category_orders,
                text_auto='.1f', title="Risk Distribution (%)"
            )
            fig_tier.update_traces(textposition='inside', textfont_size=14)
            st.plotly_chart(fig_tier, use_container_width=True)

    with col_shape:
        st.markdown("**Statistical Impact: Probability Density**")
        density_df = drift.prepare_density_data(ref_logs, batch_logs, reference_month, display_month)
        
        if not density_df.empty:
            fig_shape = px.histogram(
                density_df, x="Churn Probability", color="Month",
                barmode="overlay", histnorm="probability density",
                nbins=40, opacity=0.7, title="Prediction Shape Comparison"
            )
            # Add vertical lines for the risk thresholds (0.30 and 0.70)
            fig_shape.add_vline(x=0.25, line_dash="dot", line_color="gray", annotation_text="Medium Risk")
            fig_shape.add_vline(x=0.35, line_dash="dot", line_color="red", annotation_text="High Risk")
            
            st.plotly_chart(fig_shape, use_container_width=True)

# --- Append to the bottom of pages/3_Monitoring.py ---
import plotly.figure_factory as ff

st.markdown("---")
st.markdown("### Model Performance (Ground Truth)")

# 1. The December Trap Protection
if "December" in display_month:
    st.info(f"**Evaluation Pending:** The ground truth for {display_month} is not yet available. Model performance can only be evaluated after the month has fully concluded.")
else:
    st.markdown(f"Evaluating predictions made in **{display_month}** against actual billing outcomes.")
    
    # 2. Dynamic Threshold Slider
    # This lets the business user decide how aggressive they want the model to be
    decision_threshold = st.slider(
        "Decision Threshold (What probability equals 'High Risk'?)", 
        min_value=0.10, max_value=0.90, value=0.35, step=0.01,
        help="Lowering this catches more churners, but creates more false alarms for the CS team."
    )
    
    # 3. Calculate Metrics
    current_csv_path = os.path.join(data_folder, selected_file)
    metrics, cm = drift.calculate_performance_metrics(batch_logs, current_csv_path, threshold=decision_threshold)
    
    if metrics is None:
        st.warning("Could not load ground truth targets for this month.")
    else:
        # 4. Display Top-Line Business Metrics
        col_rec, col_prec, col_catch = st.columns(3)
        
        col_rec.metric(
            "Recall (Capture Rate)", 
            f"{metrics['Recall'] * 100:.1f}%",
            help="Out of all the people who ACTUALLY churned, what percentage did we successfully flag?"
        )
        col_prec.metric(
            "Precision (Efficiency)", 
            f"{metrics['Precision'] * 100:.1f}%",
            help="When we flagged an account as High Risk, how often were we actually right?"
        )
        col_catch.metric(
            "Total Churners Caught", 
            f"{metrics['True Positives']} / {metrics['Actual Churners']}",
            help="The raw number of saved accounts."
        )
        
        # 5. Draw the Confusion Matrix Heatmap
        st.markdown("**Confusion Matrix**")
        
        # Format the matrix for Plotly
        z = [[cm[1, 1], cm[1, 0]],  # True Positives, False Negatives (Actual Churners)
             [cm[0, 1], cm[0, 0]]]  # False Positives, True Negatives (Actual Safe)
             
        x = ['Predicted Churn (Alert CS)', 'Predicted Safe (Ignore)']
        y = ['Actually Churned', 'Actually Safe']
        
        # Create annotated heatmap
        fig_cm = ff.create_annotated_heatmap(
            z, x=x, y=y, 
            colorscale='Blues', 
            showscale=False
        )
        
        # Make it look clean
        fig_cm.update_layout(
            margin=dict(t=10, l=10, b=10, r=10),
            height=300
        )
        # Flip Y axis so "Actually Churned" is on top
        fig_cm['layout']['yaxis']['autorange'] = "reversed"
        
        col_matrix, col_text = st.columns([2, 1])
        with col_matrix:
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_text:
            st.markdown("#### The Business Reality")
            st.markdown(f"By setting the threshold to **{decision_threshold}**:")
            st.markdown(f"- The CS team successfully intervened on **{metrics['True Positives']}** accounts that were going to cancel.")
            st.markdown(f"- The model completely missed **{metrics['False Negatives']}** accounts that canceled without warning.")
            st.markdown(f"- The CS team wasted time investigating **{metrics['False Positives']}** healthy accounts (False Alarms).")