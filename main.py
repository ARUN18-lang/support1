import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model.h5')

# Define the feature names
features1 = ['Disease_Group', 'Disease_Classification', 'Income_Level', 'Severity_of_Coma_Score', 
             'Medical_Charges', 'Total_Cost', 'Blood_pH_Level', 'ADL_Physical', 
             'ADL_Social', 'ADL_Cognitive']

# Define the class descriptions
class_descriptions = {
    0: '<span style="color:green;">Less than 2 months follow-up</span>: Indicates that the follow-up period for the patient was less than 2 months. This is typically used in clinical studies or patient records to note the duration of time after a treatment or procedure during which the patient was monitored.',
    1: '<span style="color:blue;">No (M2 and SIP pres)</span>: This likely means that the conditions or scores referred to as M2 and SIP (Sickness Impact Profile) were not present. In other words, the patient did not meet the criteria for M2 and had no SIP score.',
    2: '<span style="color:orange;">SIP >= 30</span>: Refers to a score of 30 or higher on the Sickness Impact Profile (SIP), which is a measure used to assess the impact of sickness on a patient\'s daily life. A higher score suggests a greater impact on health-related quality of life.',
    3: '<span style="color:purple;">ADL >= 4 (>= 5 if sur)</span>: Refers to the Activities of Daily Living (ADL) score, which assesses a patient\'s ability to perform everyday tasks. ADL >= 4 indicates that the patient has a score of 4 or higher. >= 5 if sur might imply that if the patient had surgery, the ADL score is 5 or higher.',
    4: '<span style="color:red;">Coma or Intub</span>: Refers to whether the patient was in a coma or was intubated. Intubation involves inserting a tube into the patient\'s airway to assist with breathing, usually when they are unable to do so on their own due to severe illness or injury.'
}

# Function to make predictions
def predict(features):
    features_array = np.array([features])
    prediction = model.predict(features_array)
    predicted_class = prediction.argmax(axis=1)[0]
    return predicted_class

# Streamlit UI
st.set_page_config(page_title="Disease Classification", page_icon=":syringe:", layout="wide")

# Add a header
st.title('Disease Classification Prediction')
st.write("This application predicts the disease classification based on input features.")

# Create a form for user input
with st.form(key='prediction_form'):
    st.header('Input Features')
    
    # Collect user input
    inputs = []
    for feature in features1:
        value = st.number_input(f'{feature}', value=0.0, format="%.2f")
        inputs.append(value)
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict')

# Predict and display results
if submit_button:
    prediction = predict(inputs)
    st.header('Prediction Result')
    
    # Get the description for the predicted class
    description = class_descriptions.get(prediction, 'No description available.')
    
    # Display the prediction with color and description
    st.markdown(f'<div style="font-size: 20px; font-weight: bold; color: #333;">The predicted class is: {prediction}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 16px; color: #555;">{description}</div>', unsafe_allow_html=True)

# Add some styling
st.markdown("""
    <style>
        .css-1d391kg {padding-top: 0rem;}
        .css-1l02c5r {padding-top: 1rem;}
    </style>
    """, unsafe_allow_html=True)
