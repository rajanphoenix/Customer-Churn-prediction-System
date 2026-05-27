import shap
import matplotlib.pyplot as plt
import pandas as pd

def generate_shap_explanation(model, X_single):
    """
    Takes a model and a single pre-processed row.
    Returns the Waterfall figure and the pushing/preventing text data.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_single)
    
    if len(shap_values.shape) == 3:
        shap_obj = shap_values[0, :, 1]
    else:
        shap_obj = shap_values[0]
        
    # Generate the Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_obj, max_display=8, show=False)
    plt.tight_layout()
    
    # Generate the Text Receipt
    shap_df = pd.DataFrame({
        'Feature': X_single.columns,
        'Value': X_single.iloc[0].values,
        'Impact': shap_obj.values
    })
    
    pushing_churn = shap_df[shap_df['Impact'] > 0].sort_values(by='Impact', ascending=False).head(3)
    preventing_churn = shap_df[shap_df['Impact'] < 0].sort_values(by='Impact', ascending=True).head(3)
    
    return fig, pushing_churn, preventing_churn