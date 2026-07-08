
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------

model = joblib.load("alzheimers_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Alzheimer's Disease Classification")
st.write("Enter patient details to predict Alzheimer's disease.")

# -----------------------------
# User Inputs
# -----------------------------

Age = st.number_input("Age", 60, 90, 70)
Gender = st.selectbox("Gender", [0, 1])
Ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
EducationLevel = st.selectbox("Education Level", [0, 1, 2, 3])

BMI = st.number_input("BMI", 15.0, 40.0, 25.0)

Smoking = st.selectbox("Smoking", [0, 1])

AlcoholConsumption = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0)

PhysicalActivity = st.slider("Physical Activity", 0.0, 10.0, 5.0)

DietQuality = st.slider("Diet Quality", 0.0, 10.0, 5.0)

SleepQuality = st.slider("Sleep Quality", 4.0, 10.0, 7.0)

FamilyHistoryAlzheimers = st.selectbox("Family History of Alzheimer's", [0, 1])

CardiovascularDisease = st.selectbox("Cardiovascular Disease", [0, 1])

Diabetes = st.selectbox("Diabetes", [0, 1])

Depression = st.selectbox("Depression", [0, 1])

HeadInjury = st.selectbox("Head Injury", [0, 1])

Hypertension = st.selectbox("Hypertension", [0, 1])

SystolicBP = st.number_input("Systolic Blood Pressure", 90, 180, 120)

DiastolicBP = st.number_input("Diastolic Blood Pressure", 60, 120, 80)

CholesterolTotal = st.number_input("Total Cholesterol", 150, 300, 200)

CholesterolLDL = st.number_input("LDL Cholesterol", 50, 200, 100)

CholesterolHDL = st.number_input("HDL Cholesterol", 20, 100, 50)

CholesterolTriglycerides = st.number_input("Triglycerides", 50, 400, 150)

MMSE = st.slider("MMSE Score", 0, 30, 20)

FunctionalAssessment = st.slider("Functional Assessment", 0.0, 10.0, 5.0)

MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])

BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])

ADL = st.slider("ADL Score", 0.0, 10.0, 5.0)

Confusion = st.selectbox("Confusion", [0, 1])

Disorientation = st.selectbox("Disorientation", [0, 1])

PersonalityChanges = st.selectbox("Personality Changes", [0, 1])

DifficultyCompletingTasks = st.selectbox("Difficulty Completing Tasks", [0, 1])

Forgetfulness = st.selectbox("Forgetfulness", [0, 1])

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "FunctionalAssessment": FunctionalAssessment,
        "ADL": ADL,
        "MemoryComplaints": MemoryComplaints,
        "MMSE": MMSE,
        "BehavioralProblems": BehavioralProblems,
        "SleepQuality": SleepQuality,
        "EducationLevel": EducationLevel,
        "CholesterolHDL": CholesterolHDL,
        "Hypertension": Hypertension,
        "FamilyHistoryAlzheimers": FamilyHistoryAlzheimers,
        "CholesterolLDL": CholesterolLDL,
        "Diabetes": Diabetes,
        "CardiovascularDisease": CardiovascularDisease,
        "BMI": BMI,
        "Disorientation": Disorientation,
        "CholesterolTriglycerides": CholesterolTriglycerides,
        "HeadInjury": HeadInjury,
        "Gender": Gender,
        "PersonalityChanges": PersonalityChanges,
        "Confusion": Confusion,
        "SystolicBP": SystolicBP,
        "Ethnicity": Ethnicity
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Alzheimer's Disease Detected")
    else:
        st.success("✅ No Alzheimer's Disease Detected")

    st.subheader("Prediction Probability")

    st.write(f"No Alzheimer's Disease : {probability[0]*100:.2f}%")

    st.write(f"Alzheimer's Disease : {probability[1]*100:.2f}%")
