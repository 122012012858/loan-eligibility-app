import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample dataset
data = {
    'Age': [25, 35, 45, 20, 30, 50, 40],
    'Income': [50000, 60000, 80000, 20000, 40000, 100000, 70000],
    'LoanAmount': [10000, 20000, 15000, 5000, 12000, 25000, 18000],
    'CreditScore': [700, 650, 800, 600, 620, 750, 680],
    'Eligible': [1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)
X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
y = df['Eligible']

model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🏦 Bank Loan Eligibility Predictor")

st.sidebar.header("Applicant Information")
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, max_value=50000, value=10000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

if st.sidebar.button("Check Eligibility"):
    input_data = pd.DataFrame([[age, income, loan_amount, credit_score]],
                              columns=['Age', 'Income', 'LoanAmount', 'CreditScore'])
    prediction = model.predict(input_data)
    result = "✅ Eligible for Loan" if prediction[0] == 1 else "❌ Not Eligible for Loan"
    st.subheader("Prediction Result")
    st.success(result)