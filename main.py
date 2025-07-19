import streamlit as st
import pandas as pd
import joblib
import shap
import ast
import sys
import os
from agents.loan_amount_prediction import recommend_loan_amount





st.set_page_config(layout="wide")

st.title("AI-Powered Loan Amount Recommendation System")

# Add GitHub buttons below the title
st.markdown(
    """
    <div style='display: flex; gap: 1em;'>
        <a href="https://github.com/basanta11/streamlit_loan_amount_pred.git target="_blank">
            <button style='background-color:#24292F;color:white;padding:0.5em 1.2em;border:none;border-radius:5px;cursor:pointer;'>GitHub Main</button>
        </a>
        <a href="https://github.com/basanta11/loan_amount_pred.git target="_blank">
            <button style='background-color:#6f42c1;color:white;padding:0.5em 1.2em;border:none;border-radius:5px;cursor:pointer;'>GitHub Streamlit</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Layout: Description left, form right (no tab)
desc_col, form_col = st.columns([1.2, 1.8])
with desc_col:
    st.markdown(
        """
        **Dataset:** This app uses the Lending Club dataset of accepted loans from 2007 to 2018.  The model was trained from light gbm to predict suitable loan amounts for applicants based on these features.

        **Features:**
        - Annual Income, Employment Length, Debt-to-Income Ratio, FICO Score, Credit Inquiries, Open Credit Accounts, Revolving Balance
        - Loan Purpose, Home Ownership, Loan Term, Verification Status
        
        **Output:**
        - The app recommends a personalized loan amount for the user.
        - It also provides an explanation of the top factors that increase or decrease the recommended amount using SHAP (SHapley Additive exPlanations) values.
        """
    )

# Load model and explainer


# Load data

# Tab 1 – SHAP Chatbot
with form_col:
    st.header("💬 Rejection Explanation Bot")
    col1, col2 = st.columns(2)

with col1:
    annual_inc = st.number_input("Annual Income ($)", min_value=0, value=70000)
    emp_length = st.slider("Employment Length (Years)", 0, 20, 5)
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, value=15.0)
    fico_range_low = st.number_input("FICO Score (Low)", min_value=300, max_value=850, value=720)
    inq_last_6mths = st.number_input("Inquiries (Last 6 Months)", min_value=0, value=2)
with col2:
    open_acc = st.number_input("Open Credit Accounts", min_value=0, value=2)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, value=8000)

    term = st.selectbox("Loan Term", [" 36 months", " 60 months"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    purpose = st.selectbox("Loan Purpose", [
        "credit_card", "debt_consolidation", "home_improvement",
        "major_purchase", "small_business", "vacation", "car", "wedding", "medical", "other"
    ])
    verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

  
    
# Construct input
user_input = {
        'annual_inc': annual_inc,
        'emp_length': emp_length,
        'dti': dti,
        'fico_range_low': fico_range_low,
        'inq_last_6mths': inq_last_6mths,
        'open_acc': open_acc,
        'revol_bal': revol_bal,
        'purpose': purpose,
        'home_ownership': home_ownership,
        'term': term,
        'verification_status': verification_status
    }
   

if st.button("Recommend Loan Amount"):
    with st.spinner("🔄 Loading Data..."):
        try:
            result = recommend_loan_amount(user_input)
            st.subheader("💰 Recommended Loan Amount")
            st.write(f"**${result['recommended_loan_amount']:,.2f}**")

            # Feature explanations mapping
            feature_explanations = {
                'cat__verification_status_Verified': 'Income was verified',
                'cat__verification_status_Source Verified': 'Income was source verified',
                'cat__verification_status_Not Verified': 'Income was not verified',
                'num__fico_range_low': 'Higher FICO (credit) score',
                'num__revol_util_per_account': 'Higher revolving credit utilization per account',
                'num__revol_bal': 'Higher total revolving balance',
                'num__term_numeric': 'Longer loan term',
                'cat__purpose_grouped_debt_consolidation': 'Purpose: Debt consolidation',
                'cat__purpose_grouped_credit_card': 'Purpose: Credit card',
                'cat__purpose_grouped_home_improvement': 'Purpose: Home improvement',
                'cat__purpose_grouped_other': 'Other loan purpose',
                'num__annual_inc': 'Higher annual income',
                'num__emp_length': 'Longer employment length',
                'num__dti': 'Lower debt-to-income ratio',
                'num__inq_last_6mths': 'Fewer credit inquiries (last 6 months)',
                'num__open_acc': 'More open credit accounts',
                # Add more mappings as needed
            }

            st.subheader("🔍 Top Factors Increasing Amount")
            for item in result["top_positive_factors"]:
                explanation = feature_explanations.get(item['feature'], item['feature'])
                st.write(f"🟢 {explanation} contributed **+{item['shap_value']:.2f}**")

            st.subheader("🔻 Top Factors Decreasing Amount")
            for item in result["top_negative_factors"]:
                explanation = feature_explanations.get(item['feature'], item['feature'])
                st.write(f"🔴 {explanation} decreased **{item['shap_value']:.2f}**")
        except Exception as e:
            st.error(f"Error: {e}")






