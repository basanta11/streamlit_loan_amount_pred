import json
import joblib
import pandas as pd
import shap
import os
import streamlit as st



model_path = os.path.join("models", "loan_prediction_copy_model.joblib")
scaler_path = os.path.join("models", "preprocessor.joblib")



@st.cache_resource
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {type(e).__name__} — {e}")
        raise e

model = load_model()

def recommend_loan_amount(customer_input):
    sample_data=pd.DataFrame([customer_input]) 

    features = [
    'annual_inc', 'emp_length', 'dti', 'fico_range_low',
    'inq_last_6mths', 'open_acc', 'revol_bal',
    'purpose', 'home_ownership', 'term', 'verification_status',
    'term_numeric', 'revol_util_per_account',
    'emp_len_group', 'purpose_grouped','fifo_bucket','income_band'
    ]

    preprocessor = joblib.load(scaler_path)


    # Transform using preprocessor
    sample_data_fe = add_engineered_features_clean(sample_data)


    prediction_features=sample_data_fe[features]


    predicted_loan_amnt = model.predict(prediction_features)[0]


# Step 4: Predict the loan amount
    # Transform using preprocessor
    X_sample = preprocessor.transform(sample_data_fe)

    # SHAP explanation
    # Get the fitted tree model and preprocessor from your pipeline
    regressor = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']

# Transform your data using the preprocessor
    X_sample_transformed = preprocessor.transform(sample_data_fe)

# Now use TreeExplainer on the regressor and transformed data
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_sample_transformed)

    # Get feature names after encoding
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(X_sample.shape[1])]

    # Extract top positive and negative contributors
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    top_positive = shap_df[shap_df["shap_value"] > 0].head(3)
    top_negative = shap_df[shap_df["shap_value"] < 0].head(3)

    # Return readable result
    return {
        "recommended_loan_amount": round(predicted_loan_amnt, 2),
        "top_positive_factors": top_positive.to_dict(orient="records"),
        "top_negative_factors": top_negative.to_dict(orient="records")
    }
    


# Step 2: Apply your feature engineering (cleaned version)
def add_engineered_features_clean(df):
    df = df.copy()
    df['term_numeric'] = df['term'].str.extract('(\d+)').astype(float)
    df['revol_util_per_account'] = df['revol_bal'] / (df['open_acc'] + 1e-5)
    df['emp_len_group'] = pd.cut(
        df['emp_length'],
        bins=[-1, 1, 3, 6, 10, float('inf')],
        labels=['<1yr', '1-3yr', '3-6yr', '6-10yr', '10+'],
        include_lowest=True
    )
    top_purposes = df['purpose'].value_counts().nlargest(5).index
    df['purpose_grouped'] = df['purpose'].where(df['purpose'].isin(top_purposes), 'Other')
    df['fifo_bucket'] = pd.cut(df['fico_range_low'], [600, 650, 700, 750, 800, 850], labels=['bad', 'fair', 'good', 'very_good', 'excellent'])
    df['income_band'] = pd.cut(df['annual_inc'], [0, 40000, 80000, 120000, float('inf')], labels=['low', 'mid', 'high', 'very_high'])

    return df
