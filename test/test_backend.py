import pandas as pd
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.inference.predictor import ChurnPredictor
from database.db_manager import log_batch_predictions, get_historical_logs

print("1. Initializing the Churn Predictor...")
predictor = ChurnPredictor()

print("2. Loading the 21st Month Data...")
df_batch = pd.read_csv(r'C:\Users\AMAN SINGH\Git\sem6miniProject\customer-churn-prediction\data\processed\test21_df.csv')

print("3. Running Batch Predictions...")
results_df = predictor.predict_batch(df_batch)
print(f"   -> Successfully predicted {len(results_df)} accounts.")

print("4. Logging results to the SQLite Database...")
log_batch_predictions(results_df, "Month_21")

print("5. Retrieving logs from the Database...")
db_logs = get_historical_logs(snapshot_month="Month_21")

print("\nSUCCESS! Here are the first 5 rows currently sitting in your database:")
# Print the important columns to verify our SHAP features saved correctly
print(db_logs[['account_id', 'prediction_type', 'churn_probability', 'risk_level', 'tenure_days', 'total_mrr']].head())

print(f"\nTotal rows verified in database: {len(db_logs)}")

