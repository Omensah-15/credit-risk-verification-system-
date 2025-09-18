# utils/preprocessing.py
import pandas as pd

EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame for training. Maps categorical values and engineers features.
    """
    df = df.copy()
    # Map categorical columns with safe fallbacks
    df['employment_status'] = df.get('employment_status').map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level').map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose').map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present').map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Engineered features (avoid division by zero)
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)

    return df

def preprocess_inference_data(input_data):
    """
    Accepts a dict (single row) or DataFrame (multiple rows). Returns preprocessed DataFrame.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    return preprocess_data(df)
