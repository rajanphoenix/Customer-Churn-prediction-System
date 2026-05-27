# Customer-Churn-Prediction.
## A machine learning approach to retain customers likely to fall out

# 🚀 Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/amanAtGit09/customer-churn-prediction.git
cd customer-churn-prediction
```

---

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```
---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 4. Run the Dashboard

```bash
streamlit run app.py
```

---

# ⚠️ Important Notes

- Some notebooks/scripts may still contain local system paths from development (example: `C:/Users/...`).
- If you encounter `FileNotFoundError`, update the dataset/model paths according to your local machine directories.


- MLflow tracking is configured locally.
- Previous experiment runs (`mlruns/`) are intentionally excluded from the repository.

### MLflow Setup (Optional)

To launch the MLflow experiment tracking UI locally:

```bash
mlflow ui
```

Then open in browser:

```text
http://127.0.0.1:5000
```

---

# 📦 Tech Stack

- Python
- Scikit-learn
- XGBoost
- MLflow
- SHAP
- Streamlit
- Pandas / NumPy
- Matplotlib / Seaborn

---

# 📌 Project Features

- Temporal churn prediction pipeline
- Feature engineering from raw SaaS activity data
- Time-series validation
- Batch inference simulation
- SHAP explainability
- What-if analysis
- Monitoring dashboard
- Drift & retraining workflow (prototype)
