import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def generate_drift_report(ref_data_path, current_data_path):
    """
    Runs the Evidently DataDriftPreset.
    Returns the raw dictionary of results.
    """
    ref_data = pd.read_csv(ref_data_path)
    curr_data = pd.read_csv(current_data_path)
    
    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=ref_data, current_data=curr_data)
    
    return my_eval.dict()

def parse_drift_results(raw_results):
    """
    Parses the v0.7 Evidently dictionary into a clean Pandas DataFrame.
    Returns: (drifted_count, total_count, drift_df)
    """
    summary_metric = raw_results["metrics"][0]
    drifted_count = int(summary_metric["value"]["count"])
    total_count = int(drifted_count / summary_metric["value"]["share"]) if summary_metric["value"]["share"] > 0 else 0

    drifted_details = []
    # Loop through the metrics, skipping the summary at index 0
    for m in raw_results["metrics"][1:]: 
        col_name = m["config"].get("column")
        if not col_name: continue
        
        score = float(m["value"])
        threshold = float(m["config"]["threshold"])
        
        if score > threshold:
            drifted_details.append({"Feature": col_name, "Drift Score": score, "Threshold": threshold})

    # If nothing drifted, return empty DataFrame
    if not drifted_details:
        return drifted_count, total_count, pd.DataFrame()

    drift_df = pd.DataFrame(drifted_details)
    
    # 1. Filter out known false-positives
    drift_df = drift_df[~drift_df['Feature'].isin(['snapshot_date', 'account_id'])]
    
   # 2. Multiply by 100, cap at 100, THEN convert to integer
    drift_df['Drift Intensity'] = (drift_df['Drift Score'] * 100).clip(upper=100).astype(int)
    
    # 3. Add a human-readable severity label
    def get_severity(score):
        if score >= 80: return "🚨 Critical"
        if score >= 50: return "⚠️ High"
        return "🟡 Moderate"
        
    drift_df['Severity'] = drift_df['Drift Intensity'].apply(get_severity)
    
    # Sort worst to best
    drift_df = drift_df.sort_values(by="Drift Intensity", ascending=False)
    
    return drifted_count, total_count, drift_df

# --- Append to src/utils/drift.py ---

def calculate_prediction_summary(ref_logs, curr_logs):
    """Calculates the average churn probability and the delta between months."""
    ref_mean = ref_logs['churn_probability'].mean() if not ref_logs.empty else 0
    curr_mean = curr_logs['churn_probability'].mean() if not curr_logs.empty else 0
    delta = curr_mean - ref_mean
    return ref_mean, curr_mean, delta

def prepare_risk_tier_data(ref_logs, curr_logs, ref_name, curr_name):
    """Formats the risk tier counts into a single DataFrame for a stacked bar chart."""
    if ref_logs.empty or curr_logs.empty:
        return pd.DataFrame()

    # Get percentages instead of raw counts so we can compare different sized months
    ref_counts = ref_logs['risk_level'].value_counts(normalize=True).reset_index()
    ref_counts.columns = ['Risk Tier', 'Percentage']
    ref_counts['Month'] = ref_name

    curr_counts = curr_logs['risk_level'].value_counts(normalize=True).reset_index()
    curr_counts.columns = ['Risk Tier', 'Percentage']
    curr_counts['Month'] = curr_name

    combined_df = pd.concat([ref_counts, curr_counts])
    combined_df['Percentage'] = combined_df['Percentage'] * 100 # Convert to whole numbers
    return combined_df

def prepare_density_data(ref_logs, curr_logs, ref_name, curr_name):
    """Combines the raw probabilities into a single DataFrame for the overlapping histogram."""
    if ref_logs.empty or curr_logs.empty:
        return pd.DataFrame()
        
    ref_df = pd.DataFrame({'Churn Probability': ref_logs['churn_probability'], 'Month': ref_name})
    curr_df = pd.DataFrame({'Churn Probability': curr_logs['churn_probability'], 'Month': curr_name})
    
    return pd.concat([ref_df, curr_df])

# --- Append to src/utils/drift.py ---
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def calculate_performance_metrics(db_logs, csv_path, threshold=0.354):
    """
    Merges database predictions with CSV ground truth to calculate model performance.
    """
    if db_logs.empty:
        return None, None
        
    # Load the ground truth from the CSV
    try:
        df_true = pd.read_csv(csv_path)
        if 'target' not in df_true.columns or 'account_id' not in df_true.columns:
            return None, None
    except Exception:
        return None, None

    # Merge predictions with ground truth
    merged = pd.merge(
        db_logs[['account_id', 'churn_probability']], 
        df_true[['account_id', 'target']], 
        on='account_id', 
        how='inner'
    )
    
    if merged.empty:
        return None, None

    # Apply the dynamic threshold
    merged['predicted_churn'] = (merged['churn_probability'] >= threshold).astype(int)
    
    y_true = merged['target']
    y_pred = merged['predicted_churn']
    
    # Calculate Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        "Total Accounts": len(merged),
        "Actual Churners": sum(y_true),
        "Predicted Churners": sum(y_pred),
        "Precision": precision,
        "Recall": recall,
        "True Positives": cm[1, 1],  # Caught churners
        "False Positives": cm[0, 1], # False alarms
        "True Negatives": cm[0, 0],  # Correctly ignored safe accounts
        "False Negatives": cm[1, 0]  # Missed churners
    }
    
    return metrics, cm