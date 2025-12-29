import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Explainable Credit Risk ML", layout="centered")

st.title("💳 Explainable Credit Risk Prediction")
st.write("Enter applicant details to predict credit approval")

# Input fields
age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
employment_years = st.slider("Employment Years", 0, 40, 5)
credit_history = st.slider("Credit History Score", 0, 10, 5)

# Load training data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]

# Train model (lightweight)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
if st.button("Predict Credit Risk"):
    input_data = np.array([[age, income, loan_amount, employment_years, credit_history]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Credit Approved")
    else:
        st.error("❌ Credit Rejected")

    st.subheader("Explanation")
    st.write("""
    - Higher income and strong credit history increase approval chances  
    - Larger loan amounts increase risk  
    - Stable employment improves trust
    """)

st.markdown("---")
st.caption("Explainable Credit Risk ML Project")
