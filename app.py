import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and selected features
model = joblib.load("alzheimers_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

st.set_page_config(page_title="Alzheimer's Disease Prediction", layout="wide")

st.title("🧠 Alzheimer's Disease Classification System")

st.write("Enter the patient's details below to predict whether the patient is likely to have Alzheimer's Disease.")

# Patient Details

# Create a dictionary to store feature inputs dynamically
input_features = {}

# Dynamically generate input widgets based on selected_features
for feature in selected_features:
    if feature == "Age":
        input_features[feature] = st.number_input("Age", 60, 90, 70)
    elif feature == "Gender":
        input_features[feature] = st.selectbox("Gender", [0, 1])
    elif feature == "Ethnicity":
        input_features[feature] = st.selectbox("Ethnicity", [0, 1, 2, 3])
    elif feature == "EducationLevel":
        input_features[feature] = st.selectbox("Education Level", [0, 1, 2, 3])
    elif feature == "BMI":
        input_features[feature] = st.number_input("BMI", 15.0, 40.0, 25.0)
    elif feature == "Smoking":
        input_features[feature] = st.selectbox("Smoking", [0, 1])
    elif feature == "AlcoholConsumption":
        input_features[feature] = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
    elif feature == "PhysicalActivity":
        input_features[feature] = st.slider("Physical Activity", 0.0, 10.0, 5.0)
    elif feature == "DietQuality":
        input_features[feature] = st.slider("Diet Quality", 0.0, 10.0, 5.0)
    elif feature == "SleepQuality":
        input_features[feature] = st.slider("Sleep Quality", 4.0, 10.0, 7.0)
    elif feature == "FamilyHistoryAlzheimers":
        input_features[feature] = st.selectbox("Family History of Alzheimer's", [0, 1])
    elif feature == "CardiovascularDisease":
        input_features[feature] = st.selectbox("Cardiovascular Disease", [0, 1])
    elif feature == "Diabetes":
        input_features[feature] = st.selectbox("Diabetes", [0, 1])
    elif feature == "Depression":
        input_features[feature] = st.selectbox("Depression", [0, 1])
    elif feature == "HeadInjury":
        input_features[feature] = st.selectbox("Head Injury", [0, 1])
    elif feature == "Hypertension":
        input_features[feature] = st.selectbox("Hypertension", [0, 1])
    elif feature == "SystolicBP":
        input_features[feature] = st.number_input("Systolic BP", 90, 180, 120)
    elif feature == "DiastolicBP":
        input_features[feature] = st.number_input("Diastolic BP", 60, 120, 80)
    elif feature == "CholesterolTotal":
        input_features[feature] = st.number_input("Total Cholesterol", 150, 300, 200)
    elif feature == "CholesterolLDL":
        input_features[feature] = st.number_input("LDL Cholesterol", 50, 200, 100)
    elif feature == "CholesterolHDL":
        input_features[feature] = st.number_input("HDL Cholesterol", 20, 100, 50)
    elif feature == "CholesterolTriglycerides":
        input_features[feature] = st.number_input("Triglycerides", 50, 400, 150)
    elif feature == "MMSE":
        input_features[feature] = st.slider("MMSE Score", 0, 30, 20)
    elif feature == "FunctionalAssessment":
        input_features[feature] = st.slider("Functional Assessment", 0.0, 10.0, 5.0)
    elif feature == "MemoryComplaints":
        input_features[feature] = st.selectbox("Memory Complaints", [0, 1])
    elif feature == "BehavioralProblems":
        input_features[feature] = st.selectbox("Behavioral Problems", [0, 1])
    elif feature == "ADL":
        input_features[feature] = st.slider("ADL Score", 0.0, 10.0, 5.0)
    elif feature == "Confusion":
        input_features[feature] = st.selectbox("Confusion", [0, 1])
    elif feature == "Disorientation":
        input_features[feature] = st.selectbox("Disorientation", [0, 1])
    elif feature == "PersonalityChanges":
        input_features[feature] = st.selectbox("Personality Changes", [0, 1])
    elif feature == "DifficultyCompletingTasks":
        input_features[feature] = st.selectbox("Difficulty Completing Tasks", [0, 1])
    elif feature == "Forgetfulness":
        input_features[feature] = st.selectbox("Forgetfulness", [0, 1])
    # Add more conditions for other feature types if needed

# Prediction

if st.button("Predict"):
    # Ensure the input data is in the correct order as per selected_features
    input_data_ordered = [input_features[f] for f in selected_features]
    input_df = pd.DataFrame([input_data_ordered], columns=selected_features)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Prediction: Alzheimer's Disease Detected")
    else:
        st.success("✅ Prediction: No Alzheimer's Disease")

    st.write("Prediction Probability")

    st.write(probability)
