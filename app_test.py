import os
import json
import hashlib
import sqlite3
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, Tuple, Optional, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Credit Risk Assessment System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Configuration & Paths --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assessment_results.db")
FEATURE_COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR, "calibration_model.pkl")
BASE_MODEL_FILE = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")

# -------------------- Database Initialization --------------------
def init_db():
    """Initialize SQLite database for assessment results."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assessment_results (
            applicant_id TEXT PRIMARY KEY,
            applicant_name TEXT,
            applicant_email TEXT,
            age INTEGER,
            data_hash TEXT,
            risk_score INTEGER,
            probability_of_default REAL,
            risk_category TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Utility: Deterministic Hashing --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Generate deterministic SHA-256 hash of applicant data.
    Excludes transient fields like submission_timestamp/timestamp.
    """
    data_copy = dict(data)
    for transient in ("submission_timestamp", "timestamp"):
        data_copy.pop(transient, None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """Verify data integrity by comparing hashes."""
    return generate_data_hash(data) == original_hash

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    """
    Preprocess input data for model inference.
    Accepts dict (single row) or DataFrame (batch).
    Returns DataFrame aligned with training features.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Categorical mappings
    df['employment_status'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level', pd.Series()).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series()).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Ensure required columns exist
    required_cols = ['annual_income', 'loan_amount', 'num_previous_loans', 'credit_history_length', 'num_defaults']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Engineered features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)

    # Ensure numeric columns are numeric and fill missing values
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median = df[c].median() if not df[c].isnull().all() else 0
        df[c] = df[c].fillna(median)

    # Load expected feature columns
    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_FILE}. "
            "Please run train.py first to generate the required model files."
        )

    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    
    # Add missing columns with zero
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# -------------------- Model Loading --------------------
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
    """Load calibrated model and feature columns. Returns (model, features) or (None, None)."""
    try:
        model = joblib.load(CALIBRATED_MODEL_FILE)
        feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
        return model, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# -------------------- Prediction Functions --------------------
def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    """
    Predict for a single applicant.
    Returns: (probability, risk_score, category, processed_features)
    """
    model, feature_columns = load_models()
    if model is None:
        raise RuntimeError("Model not loaded. Please run train.py first.")
    
    processed = preprocess_inference_data(input_dict)
    
    try:
        proba = float(model.predict_proba(processed)[:, 1][0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    
    # Calculate risk score (higher = safer)
    risk_score = int(round((1 - proba) * 1000))
    
    # Determine risk category
    if proba < 0.1:
        category = "Very Low Risk"
    elif proba < 0.2:
        category = "Low Risk"
    elif proba < 0.4:
        category = "Medium Risk"
    elif proba < 0.6:
        category = "High Risk"
    else:
        category = "Very High Risk"
    
    return proba, risk_score, category, processed

# -------------------- Session State Initialization --------------------
if 'assessment_results' not in st.session_state:
    st.session_state.assessment_results = {}

# -------------------- Main UI --------------------
st.title("Credit Risk Assessment System")
st.markdown("---")

# Sidebar navigation (simplified)
menu = st.sidebar.selectbox(
    "Navigation",
    ["New Assessment", "Assessment History"]
)

# -------------------- New Assessment --------------------
if menu == "New Assessment":
    st.header("New Credit Risk Assessment")
    
    with st.form("assessment_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Applicant Information")
            applicant_id = st.text_input("Applicant ID*", help="Unique identifier for the applicant")
            applicant_name = st.text_input("Full Name*")
            applicant_email = st.text_input("Email*")
            age = st.slider("Age", 18, 100, 30)
            annual_income = st.number_input("Annual Income (USD)*", min_value=0, value=50000, step=1000)
            employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed", "student"])
            education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
            credit_history_length = st.slider("Credit History (years)", 0, 30, 5)
        
        with col2:
            st.subheader("Loan Details")
            num_previous_loans = st.slider("Number of Previous Loans", 0, 20, 2)
            num_defaults = st.slider("Number of Defaults", 0, 10, 0)
            avg_payment_delay_days = st.slider("Avg Payment Delay (days)", 0, 60, 5)
            current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
            loan_amount = st.number_input("Loan Amount (USD)*", min_value=0, value=25000, step=1000)
            loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"])
            collateral_present = st.radio("Collateral Present", ["Yes", "No"])

        st.subheader("Additional Information")
        col3, col4 = st.columns(2)
        
        with col3:
            identity_verified = st.radio("Identity Verified", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            fraud_alert = st.radio("Fraud Alert Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        with col4:
            transaction_consistency = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.8)
            digital_footprint = st.slider("Digital Footprint Score", 0, 10, 5)

        submitted = st.form_submit_button("Run Assessment", type="primary")

    if submitted:
        if not all([applicant_id, applicant_name, applicant_email]):
            st.error("Please provide Applicant ID, Name, and Email.")
        else:
            with st.spinner("Processing assessment..."):
                application = {
                    "applicant_id": applicant_id,
                    "applicant_name": applicant_name,
                    "applicant_email": applicant_email,
                    "age": int(age),
                    "annual_income": float(annual_income),
                    "employment_status": employment_status,
                    "education_level": education_level,
                    "credit_history_length": int(credit_history_length),
                    "num_previous_loans": int(num_previous_loans),
                    "num_defaults": int(num_defaults),
                    "avg_payment_delay_days": int(avg_payment_delay_days),
                    "current_credit_score": int(current_credit_score),
                    "loan_amount": float(loan_amount),
                    "loan_term_months": int(loan_term_months),
                    "loan_purpose": loan_purpose,
                    "collateral_present": collateral_present,
                    "identity_verified": int(identity_verified),
                    "transaction_consistency": float(transaction_consistency),
                    "fraud_alert": int(fraud_alert),
                    "digital_footprint": int(digital_footprint),
                    "submission_timestamp": datetime.utcnow().isoformat()
                }

                # Generate data hash
                data_hash = generate_data_hash(application)
                
                # Run prediction
                try:
                    proba, score, category, processed = predict_single(application)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                # Store in database
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO assessment_results
                        (applicant_id, applicant_name, applicant_email, age, data_hash, 
                         risk_score, probability_of_default, risk_category, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        application['applicant_id'], 
                        application['applicant_name'], 
                        application['applicant_email'], 
                        application['age'],
                        data_hash, 
                        score, 
                        proba, 
                        category,
                        application['submission_timestamp']
                    ))
                    conn.commit()
                finally:
                    conn.close()

                # Store in session state
                st.session_state.assessment_results[applicant_id] = {
                    "data": application,
                    "hash": data_hash,
                    "probability_of_default": proba,
                    "risk_score": score,
                    "risk_category": category,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Display results
                st.success("Assessment completed successfully!")
                st.markdown("### Assessment Results")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Score", f"{score}/1000")
                c2.metric("Default Probability", f"{proba:.2%}")
                c3.metric("Risk Category", category)

                st.markdown(f"**Data Hash:** `{data_hash}`")
                
                # Data integrity verification option
                if st.button("Verify Data Integrity"):
                    if verify_data_hash(application, data_hash):
                        st.success("✓ Data integrity verified - Hash matches")
                    else:
                        st.error("✗ Data integrity check failed - Hash mismatch")

# -------------------- Assessment History --------------------
elif menu == "Assessment History":
    st.header("Assessment History")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM assessment_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No assessment records yet. Create a new assessment to get started.")
    else:
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Assessments", len(df))
        col2.metric("Average Risk Score", f"{df['risk_score'].mean():.0f}")
        col3.metric("High Risk Count", len(df[df['risk_category'].str.contains('High', na=False)]))

        # Display records
        st.subheader("Assessment Records")
        
        # Format for display
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else "N/A"
        )
        display_df['data_hash_short'] = display_df['data_hash'].apply(lambda x: f"{x[:16]}..." if x else "N/A")

        st.dataframe(
            display_df[[
                'applicant_id', 'applicant_name', 'risk_score', 'risk_category',
                'probability_of_default', 'data_hash_short', 'timestamp'
            ]],
            use_container_width=True,
            column_config={
                "applicant_id": "Applicant ID",
                "applicant_name": "Name",
                "risk_score": "Risk Score",
                "risk_category": "Category",
                "probability_of_default": "Default Probability",
                "data_hash_short": "Data Hash",
                "timestamp": "Timestamp"
            }
        )

        # Export functionality
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Export",
            csv,
            "assessment_history.csv",
            "text/csv",
            key='download-csv'
        )

        # Detailed view with data integrity check
        st.subheader("Verify Record Integrity")
        selected_id = st.selectbox(
            "Select Applicant ID to verify",
            df['applicant_id'].tolist()
        )
        
        if selected_id:
            record = df[df['applicant_id'] == selected_id].iloc[0]
            
            st.markdown(f"**Stored Data Hash:** `{record['data_hash']}`")
            
            # Recreate data for verification (simplified example)
            if st.button("Verify Data Integrity"):
                # In a real scenario, you'd need to reconstruct the original data
                # This is a simplified example showing how to verify
                st.info("Data integrity verification would compare current data with stored hash")
                
                # Example verification (would need original data reconstruction)
                sample_data = {
                    "applicant_id": record['applicant_id'],
                    "applicant_name": record['applicant_name'],
                    "applicant_email": record['applicant_email'],
                    "age": record['age']
                }
                
                new_hash = generate_data_hash(sample_data)
                
                if new_hash == record['data_hash']:
                    st.success("✓ Data integrity verified - Hash matches")
                else:
                    st.warning("Note: Full verification requires complete original data")
