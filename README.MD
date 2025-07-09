# 📊 Telco Customer Churn Prediction 🔥

A fully deployed, end-to-end machine learning project that predicts customer churn for a telecom company. Built with a trained ML pipeline, integrated into a Streamlit app for interactive usage, and hosted on the cloud.

👉 **[🚀 Live App Here](https://telco-churn-vishal.streamlit.app)**  
👉 **[📁 GitHub Codebase](https://github.com/vishalxai/telco_customer_churn)**

---

## 🚀 Features

- 📦 Data preprocessing using `ColumnTransformer`
- 🌲 Trained `RandomForestClassifier` with hyperparameter tuning
- 💾 Exported & loaded pipeline with `joblib`
- 📊 Streamlit app with:
  - Upload CSV → See Predictions → Download results
  - Clean UI and robust error handling
- 🔗 Deployed on Streamlit Cloud

---

## 🧠 Problem Statement

Customer churn leads to millions in lost revenue. This app helps telecom companies **predict which customers are likely to leave**, enabling proactive retention strategies.

---

## 🧰 Tech Stack

- **Languages/Libraries**: Python, Pandas, scikit-learn, Streamlit
- **Deployment**: Streamlit Community Cloud
- **Tools**: Git, Jupyter Notebook

---

## 📸 Screenshots

### 1. 📍 App Homepage  
[![Homepage](assets/01_app_homepage.png)](assets/01_app_homepage.png)

---

### 2. 📤 CSV Upload Interface  
[![CSV Upload](assets/02_sample_csv_uploaded.png)](assets/02_sample_csv_uploaded.png)

---

### 3. ✅ Prediction Output  
[![Predictions](assets/03_prediction_results.png)](assets/03_prediction_results.png)

---

### 4. ⬇️ Download Prediction CSV  
[![Download CSV](assets/04_download_prediction_csv.png)](assets/04_download_prediction_csv.png)

---

## ⚙️ Run Locally

```bash
git clone https://github.com/vishalxai/telco_customer_churn.git
cd telco_customer_churn/streamlit_app
pip install -r ../requirements.txt
streamlit run app.py

telco_customer_churn/
├── assets/                  # UI screenshots
├── data/                    # Raw & processed data
├── models/                  # Trained model (joblib .pkl)
├── notebooks/               # Jupyter notebooks (EDA + model)
├── outputs/                 # Model pipeline output
├── streamlit_app/
│   ├── app.py               # Streamlit frontend
│   ├── sample_input.csv     # Input template
├── requirements.txt
├── README.md

