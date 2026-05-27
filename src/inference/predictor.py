import pandas as pd
import numpy as np
import joblib
import os

save_directory = 'C:/Users/AMAN SINGH/Git/sem6miniProject/customer-churn-prediction/models/'
model_path = os.path.join(save_directory, 'churn_model_rf1.pkl')
features_path = os.path.join(save_directory, 'model_features_rf1.pkl')

class ChurnPredictor:
    def __init__(self, model_path=model_path, features_path=features_path):
        """
        Initializes the predictor by loading the saved Random Forest model 
        and the exact feature columns it expects.
        """
        if not os.path.exists(model_path) or not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing model files in models/ directory.")
            
        self.model = joblib.load(model_path)
        self.expected_features = joblib.load(features_path)
        
        self.medium_threshold = 0.300
        self.high_threshold = 0.354

    def _assign_risk_tier(self, probability):
        """Helper method to categorize risk based on data-driven thresholds."""
        if probability >= self.high_threshold:
            return "High Risk"
        elif probability >= self.medium_threshold:
            return "Medium Risk"
        else:
            return "Low Risk"

    def predict_batch(self, df):
        """
        Evaluates a full month of data.
        """
        # 1. Create a SEPARATE encoded dataframe for the model so we don't destroy the original columns
        df_encoded = pd.get_dummies(df, drop_first=False, dtype=int)
        print(len(df), "rows in batch input")  # Debugging line to verify data is loaded
        
        # making sure all expected features are present in the encoded dataframe, filling missing ones with 0
        X_batch = df_encoded.reindex(columns=self.expected_features, fill_value=0)
        
        # Predict Probabilities
        probs = self.model.predict_proba(X_batch)[:, 1]
        
        # Append results to a copy of the ORIGINAL raw dataframe to keep account_ids intact
        results_df = df.copy() 
        results_df['Churn_Probability'] = np.round(probs, 4)
        results_df['Risk_Level'] = results_df['Churn_Probability'].apply(self._assign_risk_tier)
        
        return results_df

    def predict_single(self, full_user_row_dict):
        """
        Evaluates a single user for What-If analysis.
        Returns the probability, risk tier, and the perfectly encoded X row for SHAP.
        """
        df_single = pd.DataFrame([full_user_row_dict])
        
        # One-hot encode the single row
        df_encoded = pd.get_dummies(df_single, drop_first=False, dtype=int)
        
        # Reindex to map to the 60 expected features
        X_single = df_encoded.reindex(columns=self.expected_features, fill_value=0)
        
        # Predict
        prob = self.model.predict_proba(X_single)[0, 1]
        risk = self._assign_risk_tier(prob)
        
        return prob, risk, X_single