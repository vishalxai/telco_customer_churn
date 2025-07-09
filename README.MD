# Telco Customer Churn Prediction

This project aims to predict customer churn based on demographic and service usage data using machine learning models.

---
📁 Project Structure
├── data/                        # Raw dataset (CSV)
├── notebooks/                  # Jupyter notebooks (EDA + modeling)
├── outputs/                    # Saved model (.pkl), reports
├── README.md                   # Project overview and instructions
└── .gitignore                  # Exclude unnecessary files

## 🔍 Problem Statement

Predict whether a customer will **churn or not** based on their service behavior and demographics using ML models (Logistic Regression, Random Forest, XGBoost).

---

## 📦 Models Used

- Logistic Regression
- Random Forest
- XGBoost

---

## 📊 Performance Snapshot

| Model            | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| LogisticRegression | 0.79     | 0.84/0.63 | 0.89/0.54 | 0.86/0.58 |
| RandomForest       | 0.80     | 0.82/0.68 | 0.92/0.46 | 0.87/0.55 |
| XGBoost            | 0.79     | 0.84/0.61 | 0.88/0.54 | 0.86/0.57 |

---

## 🛠️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/telco-customer-churn.git
cd telco-customer-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/01_data_loading_and_eda.ipynb

---

### ✅ 3. `requirements.txt` (optional but ideal)

Create it **in the root** using this command:

```bash
pip freeze > requirements.txt


git init
git add .
git commit -m "Initial commit - Telco Churn Prediction"
