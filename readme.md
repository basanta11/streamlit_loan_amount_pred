# Credit Risk Loan Approval & Amount Recommendation App 🎯

A Streamlit-based machine learning application for recommending loan amounts, explaining rejections using SHAP, and enabling natural language queries on your data.

## Features

- **Loan Amount Recommendation:**  
  Enter applicant details to get a recommended loan amount based on a trained ML model.

- **SHAP-based Explainability:**  
  See the top factors (features) that increase or decrease the recommended loan amount, with SHAP value explanations.

- **Rejection Explanation Bot:**  
  Understand why a loan application might be rejected, under review, or approved, with risk assessment.

- **Text-to-SQL Query (Prototype):**  
  Ask questions about your data in plain English and get answers using LLM-powered SQL generation.

## Demo

Try the app live: [Streamlit Cloud Link](https://apploanamountpred-5zocuidd3najnjzfmxj8hj.streamlit.app/)  

## Getting Started

```bash
git clone https://github.com/yourname/loan_amount_pred
cd loan_amount_pred
pip install -r requirements.txt
streamlit run app/main.py


## Usage

1. **Loan Recommendation:**  
   - Fill in applicant details (income, employment, DTI, FICO, etc.).
   - Click "Recommend Loan Amount" to get a suggested amount and SHAP explanations.

2. **Rejection Explanation:**  
   - Use the chatbot to see why an application might be rejected or approved.

3. **Text-to-SQL:**  
   - Ask questions about your data in natural language (prototype).

## Input Fields

- Annual Income
- Employment Length
- Debt-to-Income Ratio
- FICO Score
- Inquiries (last 6 months)
- Open Credit Accounts
- Revolving Balance
- Loan Term
- Home Ownership
- Loan Purpose
- Verification Status

## Model & Data

- Trained on LendingClub data (2007–2018).
- Uses LightGBM for prediction and SHAP for explainability.
- Feature engineering includes income bands, FICO buckets, employment length groups, and more.

## Project Structure

- `app/main.py` – Streamlit app entry point
- `models/` – Trained models and preprocessors
- `data/` – Raw and processed datasets
- `src/` – Source code (agents, components, notebooks)

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## License

MIT License
