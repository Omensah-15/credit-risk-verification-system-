import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Credit Risk Assessment",
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
    """
    data_copy = dict(data)
    data_copy.pop("submission_timestamp", None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """Verify data integrity by comparing hashes."""
    return generate_data_hash(data) == original_hash

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Personal': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    """
    Preprocess input data for model inference.
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
            f"Feature columns file not found. Please run train.py first."
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
    """Load calibrated model and feature columns."""
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

# Sidebar navigation
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
            st.subheader("Personal Information")
            applicant_id = st.text_input("Applicant ID *", help="Unique identifier for the applicant")
            applicant_name = st.text_input("Full Name *")
            applicant_email = st.text_input("Email Address *")
            age = st.slider("Age", 18, 100, 30)
            employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed", "student"])
            education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
        
        with col2:
            st.subheader("Financial Information")
            annual_income = st.number_input("Annual Income (USD) *", min_value=0, value=50000, step=1000, format="%d")
            loan_amount = st.number_input("Loan Amount (USD) *", min_value=0, value=25000, step=1000, format="%d")
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Personal", "Car Loan", "Education", "Home Loan"])
            loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
            collateral_present = st.radio("Collateral Present", ["Yes", "No"], horizontal=True)
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Credit History")
            credit_history_length = st.slider("Credit History (years)", 0, 30, 5)
            num_previous_loans = st.slider("Number of Previous Loans", 0, 20, 2)
            num_defaults = st.slider("Number of Defaults", 0, 10, 0)
            current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
        
        with col4:
            st.subheader("Payment Behavior")
            avg_payment_delay_days = st.slider("Average Payment Delay (days)", 0, 60, 5)

        submitted = st.form_submit_button("Run Assessment", type="primary", use_container_width=True)

    if submitted:
        # Validate required fields
        missing_fields = []
        if not applicant_id:
            missing_fields.append("Applicant ID")
        if not applicant_name:
            missing_fields.append("Full Name")
        if not applicant_email:
            missing_fields.append("Email Address")
        
        if missing_fields:
            st.error(f"Please provide: {', '.join(missing_fields)}")
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

                # Display results in a clean format
                st.markdown("### Assessment Results")
                
                # Results in a nice container
                with st.container():
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    # Risk Score with color coding based on value
                    score_color = "green" if score > 700 else "orange" if score > 400 else "red"
                    col_result1.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Risk Score</p>
                            <p style="font-size: 36px; font-weight: bold; color: {score_color};">{score}</p>
                            <p style="color: #666;">/1000</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Default Probability
                    proba_color = "green" if proba < 0.2 else "orange" if proba < 0.4 else "red"
                    col_result2.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Default Probability</p>
                            <p style="font-size: 36px; font-weight: bold; color: {proba_color};">{proba:.1%}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Risk Category
                    category_colors = {
                        "Very Low Risk": "#28a745",
                        "Low Risk": "#5cb85c",
                        "Medium Risk": "#f0ad4e",
                        "High Risk": "#d9534f",
                        "Very High Risk": "#c9302c"
                    }
                    cat_color = category_colors.get(category, "#666")
                    col_result3.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                            <p style="color: #666; margin-bottom: 5px;">Risk Category</p>
                            <p style="font-size: 28px; font-weight: bold; color: {cat_color};">{category}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                # Data hash display
                st.markdown("### Data Integrity")
                st.code(data_hash, language="text")
                
                # Verification button
                col_verify, _ = st.columns([1, 3])
                with col_verify:
                    if st.button("Verify Data Integrity", use_container_width=True):
                        if verify_data_hash(application, data_hash):
                            st.success("Data integrity verified - Hash matches")
                        else:
                            st.error("Data integrity check failed - Hash mismatch")

# -------------------- Assessment History --------------------
elif menu == "Assessment History":
    st.header("Assessment History")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM assessment_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No assessment records found. Create a new assessment to get started.")
    else:
        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Assessments", len(df))
        
        with col2:
            avg_score = df['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_score:.0f}")
        
        with col3:
            high_risk_count = len(df[df['risk_category'].str.contains('High', na=False)])
            st.metric("High Risk Cases", high_risk_count)

        # Assessment records table
        st.subheader("Assessment Records")
        
        # Prepare display dataframe
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.1%}" if pd.notnull(x) else "N/A"
        )
        display_df['data_hash_short'] = display_df['data_hash'].apply(lambda x: f"{x[:16]}..." if x else "N/A")
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        # Display as a clean table
        st.dataframe(
            display_df[[
                'applicant_id', 'applicant_name', 'risk_score', 'risk_category',
                'probability_of_default', 'timestamp'
            ]],
            use_container_width=True,
            column_config={
                "applicant_id": "Applicant ID",
                "applicant_name": "Name",
                "risk_score": "Risk Score",
                "risk_category": "Risk Category",
                "probability_of_default": "Default Prob.",
                "timestamp": "Assessment Date"
            }
        )

        # Export and verify section
        col_export, col_verify = st.columns(2)
        
        with col_export:
            csv = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV Export",
                csv,
                "assessment_history.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_verify:
            # Record verification
            if not df.empty:
                selected_id = st.selectbox(
                    "Select applicant to verify",
                    df['applicant_id'].tolist(),
                    key="verify_select"
                )
                
                if selected_id and st.button("Verify Record Integrity", use_container_width=True):
                    record = df[df['applicant_id'] == selected_id].iloc[0]
                    
                    st.markdown("**Stored Data Hash:**")
                    st.code(record['data_hash'], language="text")
                    
                    # Simplified verification note
                    st.info(
                        "To fully verify data integrity, you would need to reconstruct "
                        "the original application data and compare its hash with the stored hash."
                    )
