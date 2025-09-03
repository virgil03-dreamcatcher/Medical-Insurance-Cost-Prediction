import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('xgboost_medical_insurance_model.pkl')

st.title("Medical Insurance Cost Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Convert inputs to model features as done in training
def preprocess_inputs(age, sex, bmi, children, smoker, region):
    # Map categorical inputs to encoded columns matching training
    sex_male = 1 if sex == 'Male' else 0
    smoker_yes = 1 if smoker == 'Yes' else 0
    region_northwest = 1 if region == 'Northwest' else 0
    region_southeast = 1 if region == 'Southeast' else 0
    region_southwest = 1 if region == 'Southwest' else 0
    
    # Construct a DataFrame with one row as input features
    input_dict = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [sex_male],
        'smoker_yes': [smoker_yes],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest],
    }
    return pd.DataFrame(input_dict)

if st.button("Predict Insurance Cost"):
    input_data = preprocess_inputs(age, sex, bmi, children, smoker, region)
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Medical Insurance Cost: ${prediction:,.2f}")

# Load data once for EDA
@st.cache_data
def load_data():
    return pd.read_csv('medical_insurance.csv')
df = load_data()

st.sidebar.title("Explore EDA Visualizations")
eda_option = st.sidebar.selectbox(
    "Select a Visualization",
    (
        "Distribution of Charges",
        "Charges vs Age",
        "Charges by Smoker Status",
        "Charges vs BMI by Smoker Status"
    )
)

st.header("Exploratory Data Analysis (EDA)")

if eda_option == "Distribution of Charges":
    st.subheader("Distribution of Medical Insurance Charges")
    fig, ax = plt.subplots()
    sns.histplot(df['charges'], kde=True, ax=ax)
    st.pyplot(fig)

elif eda_option == "Charges vs Age":
    st.subheader("Charges vs Age")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='charges', data=df, ax=ax)
    st.pyplot(fig)

elif eda_option == "Charges by Smoker Status":
    st.subheader("Charges by Smoker Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
    st.pyplot(fig)

elif eda_option == "Charges vs BMI by Smoker Status":
    st.subheader("Charges vs BMI by Smoker Status")
    fig, ax = plt.subplots()
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, ax=ax)
    st.pyplot(fig)
