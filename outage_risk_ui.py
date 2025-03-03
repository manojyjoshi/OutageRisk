import os
import shutil
import streamlit as st
import pandas as pd
import joblib  # For loading your trained model
import spacy
from spacy.cli import download

# Load your trained model and vectorizer
model = joblib.load('outage_risk_stacking_model.joblib')  # Load your trained model
vectorizer = joblib.load('vectorizer.joblib')  # Load the saved vectorizer
scaler = joblib.load('scaler.joblib')  # Load the saved scaler if you saved it

# Load your data
data = pd.read_csv('synthetic_defect_data.csv')

st.title('Outage Risk Prediction POC')

# Display the data
st.subheader('Defect Data')
st.dataframe(data)

# User input for a new defect
st.subheader('Predict Outage Risk for a New Defect')

severity = st.selectbox('Severity', [1, 2, 3, 4])
component = st.selectbox('Component', [1, 2, 3, 4])
defect_age = st.number_input('Defect Age (Days)', min_value=0, max_value=30, value=5)
description = st.text_area('Description', 'Enter defect description here')

# Preprocess the description
risk_keywords = [
    "data loss", "security", "authentication", "critical", "backup", "rollback",
    "connection", "timeout", "data corruption", "performance degradation",
    "system failure", "service disruption", "memory leak", "resource exhaustion",
    "access denied", "configuration error", "crash", "exception", "deadlock",
    "unresponsive"
]

def keyword_presence(description):
    for keyword in risk_keywords:
        if keyword in description.lower():
            return 1
    return 0

keyword_present = keyword_presence(description)

# **Fix: Ensure the same number of features as during training**
# Create a DataFrame from the input data
new_defect = pd.DataFrame([[severity, component, defect_age, keyword_present]],
                          columns=['Severity', 'Component', 'Defect Age (Days)', 'Keyword Present'])

# Apply TF-IDF vectorizer to the description
description_tfidf = vectorizer.transform([description]).toarray()

# Create the same feature columns that were used for training
new_defect_tfidf = pd.DataFrame(description_tfidf, columns=[f"word_{i}" for i in range(description_tfidf.shape[1])])

# Add the interaction feature "Severity_Component" to match the training feature set
new_defect['Severity_Component'] = new_defect['Severity'] * new_defect['Component']

# Concatenate the new features with the tfidf features to match the expected feature set
new_defect = pd.concat([new_defect, new_defect_tfidf], axis=1)

# Ensure that the new defect data has the same number of features as the training data
# Standardize the numeric features (e.g., Severity, Component, Defect Age, Severity_Component)
new_defect_scaled = scaler.transform(new_defect)

# Predict the outage risk
if st.button('Predict Outage Risk'):
    risk = model.predict(new_defect_scaled)[0]

    if risk == 1:
        st.error('High Outage Risk Detected!')
    else:
        st.success('Low Outage Risk.')
