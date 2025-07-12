import streamlit as st
import pandas as pd
import joblib
import shap
import ast
import sys
import os
st.write("Working dir:", os.getcwd())
st.write("Files:", os.listdir("models"))



from agents.loan_amount_prediction import recommend_loan_amount
  
st.set_page_config(layout="wide")
st.title("Loan Amount Recomendation")

tab1,= st.tabs([" Form"])

# Load model and explainer


# Load data

# Tab 1 – SHAP Chatbot
with tab1:
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
    # if st.button("Check Status"):
    #     try:
    #         st.markdown(explain_rejection( user_input, credit_type))
    #     except Exception as e:
    #         st.error(f"Error: {e}")

    if st.button("Recommend Loan Amount"):
        with st.spinner("🔄 Loading Data..."):
            try:
                result = recommend_loan_amount(user_input)
                st.subheader("💰 Recommended Loan Amount")
                st.write(f"**${result['recommended_loan_amount']:,.2f}**")

                st.subheader("🔍 Top Factors Increasing Amount")
                for item in result["top_positive_factors"]:
                    st.write(f"🟢 `{item['feature']}` contributed **+{item['shap_value']:.2f}**")

                st.subheader("🔻 Top Factors Decreasing Amount")
                for item in result["top_negative_factors"]:
                    st.write(f"🔴 `{item['feature']}` contributed **{item['shap_value']:.2f}**")
            except Exception as e:
                st.error(f"Error: {e}") 
# Tab 2 – Natural Language SQL

# Tab 3 – Multi-Agent


