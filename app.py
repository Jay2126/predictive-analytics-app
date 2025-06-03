import streamlit as st
import joblib
import numpy as np

# Load models and encoders
treatment_model = joblib.load("treatment_model.pkl")
recovery_model = joblib.load("recovery_days_model.pkl")
outcome_model = joblib.load("outcome_model.pkl")
encoders = joblib.load("encoders.pkl")

# Page configuration
st.set_page_config(page_title="Health Predictor", page_icon="🩺", layout="centered")

# Sidebar branding
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Medical_caduceus_symbol.svg/1200px-Medical_caduceus_symbol.svg.png", width=80)
    st.title("🔍 Health Insight")
    st.markdown("Predict patient **Treatment**, **Recovery Time**, and **Outcome**.")

# Main title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Patient Recovery Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Please enter patient details below:")

# Helper
def add_placeholder(options, label="-- Select --"):
    return [label] + list(options)

# Input layout
col1, col2 = st.columns(2)
with col1:
    patient_area = st.selectbox("🌍 Patient Area", add_placeholder(encoders['patient_area'].classes_))
    diagnosis = st.selectbox("🧪 Diagnosis", add_placeholder(encoders['diagnosis'].classes_))
    gender = st.selectbox("👤 Gender", add_placeholder(encoders['gender'].classes_))
with col2:
    age = st.slider("🎂 Age", 0, 100, 30)
    month = st.selectbox("📅 Admission Month", add_placeholder(encoders['month'].classes_))
    severity = st.selectbox("⚠️ Severity", add_placeholder(encoders['severity'].classes_))

# Prediction
if st.button("🔮 Predict Now"):
    if "-- Select --" in (patient_area, diagnosis, gender, month, severity):
        st.error("🚫 Please fill in all the fields before prediction.")
    else:
        encoded_inputs = {
            'patient_area': encoders['patient_area'].transform([patient_area])[0],
            'diagnosis': encoders['diagnosis'].transform([diagnosis])[0],
            'gender': encoders['gender'].transform([gender])[0],
            'age': age,
            'month': encoders['month'].transform([month])[0],
            'severity': encoders['severity'].transform([severity])[0]
        }

        feature_list = list(encoded_inputs.values())

        # Prediction steps
        treatment_encoded = treatment_model.predict([feature_list])[0]
        treatment_label = encoders['treatment'].inverse_transform([treatment_encoded])[0]

        feature_list_with_treatment = feature_list + [treatment_encoded]
        recovery_days = recovery_model.predict([feature_list_with_treatment])[0]

        feature_list_with_all = feature_list_with_treatment + [recovery_days]
        outcome_encoded = outcome_model.predict([feature_list_with_all])[0]
        outcome_label = encoders['outcome'].inverse_transform([outcome_encoded])[0]

        # Results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        st.success(f"💊 **Recommended Treatment**: `{treatment_label}`")
        st.info(f"📆 **Estimated Recovery Days**: `{round(recovery_days)} days`")
        st.warning(f"🧬 **Predicted Outcome**: `{outcome_label}`")
        st.markdown("---")
