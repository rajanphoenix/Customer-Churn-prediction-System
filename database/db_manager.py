import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DB_PATH = os.path.join(BASE_DIR, 'churn_logs.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            account_id TEXT,
            snapshot_month TEXT,
            prediction_type TEXT, 
            title TEXT,
            churn_probability REAL,
            risk_level TEXT,
            
            -- THE TOP 8 FEATURES --
            tenure_days REAL,
            total_seats REAL,
            total_mrr REAL,
            avg_duration_last_30d REAL,
            total_usage_count_30d REAL,
            days_since_last_action REAL,
            total_active_subs REAL,
            avg_resolution_last_90d REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(account_id, snapshot_month, prediction_type, probability, risk_level, 
                   features_dict, title=None):
    """Logs a single What-If Scenario using a dictionary of the 8 features."""
    if title is None or title.strip() == "":
        title = f"Scenario: {account_id}"

    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO inference_logs (
            account_id, snapshot_month, prediction_type, title, churn_probability, risk_level,
            tenure_days, total_seats, total_mrr, avg_duration_last_30d, 
            total_usage_count_30d, days_since_last_action, total_active_subs, avg_resolution_last_90d
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (account_id, snapshot_month, prediction_type, title, probability, risk_level, 
          features_dict['tenure_days'], features_dict['total_seats'], features_dict['total_mrr'], 
          features_dict['avg_duration_last_30d'], features_dict['total_usage_count_30d'], 
          features_dict['days_since_last_action'], features_dict['total_active_subs'], 
          features_dict['avg_resolution_last_90d']))
    conn.commit()
    conn.close()

def log_batch_predictions(results_df, snapshot_month):
    logs_df = pd.DataFrame({
        'account_id': results_df['account_id'],
        'snapshot_month': snapshot_month,
        'prediction_type': 'BATCH',
        'title': 'Batch Run',
        'churn_probability': results_df['Churn_Probability'],
        'risk_level': results_df['Risk_Level'],
        'tenure_days': results_df['tenure_days'],
        'total_seats': results_df['total_seats'],
        'total_mrr': results_df['total_mrr'],
        'avg_duration_last_30d': results_df['avg_duration_last_30d'],
        'total_usage_count_30d': results_df['total_usage_count_30d'],
        'days_since_last_action': results_df['days_since_last_action'],
        'total_active_subs': results_df['total_active_subs'],
        'avg_resolution_last_90d': results_df['avg_resolution_last_90d']
    })
    
    conn = get_connection()
    logs_df.to_sql('inference_logs', conn, if_exists='append', index=False)
    conn.close()

def get_historical_logs(snapshot_month=None, prediction_type=None):
    """Retrieves the logs as a Pandas DataFrame."""
    conn = get_connection()
    query = "SELECT * FROM inference_logs WHERE 1=1"
    
    if snapshot_month:
        query += f" AND snapshot_month = '{snapshot_month}'"
    if prediction_type:
        query += f" AND prediction_type = '{prediction_type}'"
        
    query += " ORDER BY timestamp DESC"
        
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    initialize_db()
    print(f"SUCCESS: Database schema updated and created at:\n{DB_PATH}")
else:
    initialize_db()