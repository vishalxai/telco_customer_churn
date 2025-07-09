import streamlit as st
import pandas as pd
import joblib
import os

# Load pipeline model
model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "random_forest_pipeline_model.pkl")
model = joblib.load(model_path)

def main():
    st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Upload your customer data to get churn predictions using a trained ML model.")

    # Sample CSV download
    with open("sample_input.csv", "rb") as file:
        st.download_button("📁 Download Sample Input CSV", file, file_name="sample_input.csv", mime="text/csv")

    uploaded_file = st.file_uploader("📤 Upload your customer CSV file", type=["csv"])

    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Input Data Preview")
            st.dataframe(input_df.head())

            # Model prediction
            predictions = model.predict(input_df)
            prediction_labels = ["Yes" if pred == "Yes" else "No" for pred in predictions]
            input_df["Predicted_Churn"] = prediction_labels

            st.subheader("✅ Prediction Results")
            st.dataframe(input_df)

            # CSV download of results
            result_csv = input_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Prediction Results CSV", result_csv, file_name="churn_predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()