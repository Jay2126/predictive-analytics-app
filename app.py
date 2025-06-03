import streamlit as st
import joblib
import numpy as np

# Load latest models and encoders
treatment_model = joblib.load("treatment_model.pkl")
recovery_model = joblib.load("recovery_days_model.pkl")
outcome_model = joblib.load("outcome_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🩺 Complete Patient Prediction App")
st.markdown("Predicts **Treatment**, **Recovery Days**, and **Outcome** based on patient inputs.")

# Input fields
patient_area = st.selectbox("Patient Area", encoders['patient_area'].classes_)
diagnosis = st.selectbox("Diagnosis", encoders['diagnosis'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
age = st.slider("Age", 0, 100, 30)
month = st.selectbox("Admission Month", encoders['month'].classes_)
severity = st.selectbox("Severity", encoders['severity'].classes_)

if st.button("Predict All"):
    # Encode inputs
    encoded_inputs = {
        'patient_area': encoders['patient_area'].transform([patient_area])[0],
        'diagnosis': encoders['diagnosis'].transform([diagnosis])[0],
        'gender': encoders['gender'].transform([gender])[0],
        'age': age,
        'month': encoders['month'].transform([month])[0],
        'severity': encoders['severity'].transform([severity])[0]
    }

    feature_list = [
        encoded_inputs['patient_area'],
        encoded_inputs['diagnosis'],
        encoded_inputs['gender'],
        encoded_inputs['age'],
        encoded_inputs['month'],
        encoded_inputs['severity']
    ]

    # Predict Treatment
    treatment_encoded = treatment_model.predict([feature_list])[0]
    treatment_label = encoders['treatment'].inverse_transform([treatment_encoded])[0]

    # Predict Recovery Days
    feature_list_with_treatment = feature_list + [treatment_encoded]
    recovery_days = recovery_model.predict([feature_list_with_treatment])[0]

    # Predict Outcome
    feature_list_with_all = feature_list_with_treatment + [recovery_days]
    outcome_encoded = outcome_model.predict([feature_list_with_all])[0]
    outcome_label = encoders['outcome'].inverse_transform([outcome_encoded])[0]

    # Display results
    st.success(f"💊 **Predicted Treatment:** {treatment_label}")
    st.info(f"📅 **Estimated Recovery Days:** {round(recovery_days)}")
    st.warning(f"⚕️ **Predicted Outcome:** {outcome_label}")
