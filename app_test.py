import os
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, Any, Tuple

import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="CreditIQ — Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  GLOBAL STYLES  — clean, theme-adaptive
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* CSS variables that adapt to both light/dark themes */
:root {
    /* Base colors - will be overridden by Streamlit's theme */
    --bg-primary: var(--background-color, #FFFFFF);
    --bg-secondary: var(--secondary-background-color, #F7F8FA);
    --text-primary: var(--text-color, #111827);
    --text-secondary: var(--text-color, #4B5563);
    --text-tertiary: #9CA3AF;
    --border-light: rgba(128, 128, 128, 0.2);
    --border-medium: rgba(128, 128, 128, 0.3);
    
    /* Accent colors */
    --accent: #1D4ED8;
    --accent-hover: #1E40AF;
    --accent-soft: rgba(29, 78, 216, 0.1);
    --green: #059669;
    --green-soft: rgba(5, 150, 105, 0.1);
    --amber: #D97706;
    --amber-soft: rgba(217, 119, 6, 0.1);
    --red: #DC2626;
    --red-soft: rgba(220, 38, 38, 0.1);
    
    /* Effects */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 25px rgba(0,0,0,0.1), 0 4px 10px rgba(0,0,0,0.04);
    
    /* Border radius */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

/* Base typography */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}

/* Headings */
h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
h2 { font-size: 1.15rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }

/* Form card */
[data-testid="stForm"] {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 2.2rem !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stRadio label {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.055em !important;
    color: var(--text-secondary) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-primary) !important;
    border: 1.5px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    transition: all 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-primary) !important;
    border: 1.5px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

/* Radio pills - modern toggle style */
.stRadio > div {
    gap: 4px !important;
    background: var(--bg-secondary) !important;
    padding: 4px !important;
    border-radius: 40px !important;
    display: inline-flex !important;
    border: 1px solid var(--border-light) !important;
}
.stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 5px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin: 0 !important;
}
.stRadio > div > label:hover {
    color: var(--accent) !important;
}
.stRadio > div > label[data-checked="true"] {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── COOL BUTTONS — theme adaptive, modern ── */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.025em !important;
    border-radius: var(--radius-md) !important;
    padding: 0.62rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Primary button — gradient, elevated */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-hover)) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3) !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(29, 78, 216, 0.4) !important;
}
.stFormSubmitButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.3) !important;
}

/* Secondary button — glass morphism */
.stButton > button {
    background: rgba(128, 128, 128, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: rgba(128, 128, 128, 0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Download button — subtle, clean */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text-primary) !important;
    border: 1.5px dashed var(--border-medium) !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
    transform: translateY(-2px) !important;
}

/* Button ripple effect on click */
.stButton > button::after,
.stFormSubmitButton > button::after,
.stDownloadButton > button::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(255, 255, 255, 0.5);
    opacity: 0;
    border-radius: 100%;
    transform: scale(1, 1) translate(-50%);
    transform-origin: 50% 50%;
}
.stButton > button:focus:not(:active)::after,
.stFormSubmitButton > button:focus:not(:active)::after,
.stDownloadButton > button:focus:not(:active)::after {
    animation: ripple 0.6s ease-out;
}

@keyframes ripple {
    0% {
        transform: scale(0, 0);
        opacity: 0.5;
    }
    100% {
        transform: scale(20, 20);
        opacity: 0;
    }
}

/* Metrics/KPI cards */
[data-testid="stMetric"],
.kpi-card {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.1rem 1.3rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s !important;
}
[data-testid="stMetric"]:hover,
.kpi-card:hover {
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.055em !important;
    color: var(--text-tertiary) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Alerts - theme adaptive */
.stSuccess { 
    background: var(--green-soft) !important; 
    border-left: 3px solid var(--green) !important; 
    border-radius: var(--radius-sm) !important; 
}
.stError { 
    background: var(--red-soft) !important; 
    border-left: 3px solid var(--red) !important; 
    border-radius: var(--radius-sm) !important; 
}
.stWarning { 
    background: var(--amber-soft) !important; 
    border-left: 3px solid var(--amber) !important; 
    border-radius: var(--radius-sm) !important; 
}

/* Dividers */
hr { 
    border-color: var(--border-light) !important; 
    margin: 1.8rem 0 !important; 
}

/* Custom utility classes */
.kpi-label {
    font-size: 0.7rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.07em; 
    color: var(--text-tertiary); 
    margin-bottom: 0.4rem;
}
.kpi-value { 
    font-size: 2rem; 
    font-weight: 700; 
    line-height: 1.1; 
    color: var(--text-primary); 
}
.kpi-sub { 
    font-size: 0.78rem; 
    color: var(--text-tertiary); 
    margin-top: 0.2rem; 
}

.section-eyebrow {
    font-size: 0.7rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.08em; 
    color: var(--accent);
    padding-bottom: 0.55rem; 
    border-bottom: 1.5px solid var(--border-light); 
    margin-bottom: 1.3rem;
}

/* Hash/fingerprint block */
.hash-block {
    background: var(--bg-secondary); 
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm); 
    padding: 14px 18px;
}
.hash-label {
    font-size: 0.68rem; 
    font-weight: 600; 
    text-transform: uppercase;
    letter-spacing: 0.07em; 
    color: var(--text-tertiary); 
    margin-bottom: 5px;
}
.hash-value {
    font-family: 'IBM Plex Mono', monospace; 
    font-size: 0.78rem;
    color: var(--green); 
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  PATHS & DB
# ──────────────────────────────────────────────
DATA_DIR    = "data";   MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "assessment_results.db")

# Model file paths
CALIBRATION_MODEL_PATH = os.path.join(MODELS_DIR, "calibration_model.pkl")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS assessment_results (
            applicant_id TEXT PRIMARY KEY, applicant_name TEXT, applicant_email TEXT,
            age INTEGER, data_hash TEXT, risk_score INTEGER,
            probability_of_default REAL, risk_category TEXT, timestamp TEXT,
            annual_income REAL, loan_amount REAL, employment_status TEXT, credit_score INTEGER
        )""")
    conn.commit(); conn.close()
init_db()

def generate_data_hash(data: Dict[str, Any]) -> str:
    d = {k: v for k, v in data.items() if k != "submission_timestamp"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()

def verify_data_hash(data, h): return generate_data_hash(data) == h

# ──────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load the LightGBM model, calibration model, and feature columns"""
    try:
        # Load feature columns first (needed for preprocessing)
        feature_columns = None
        if os.path.exists(FEATURE_COLUMNS_PATH):
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            st.sidebar.success(f"Loaded {len(feature_columns)} feature columns")
        
        # Load the LightGBM model
        lgbm_model = None
        if os.path.exists(LGBM_MODEL_PATH):
            lgbm_model = joblib.load(LGBM_MODEL_PATH)
            st.sidebar.success("Loaded LightGBM model")
        
        # Load the calibration model
        calibration_model = None
        if os.path.exists(CALIBRATION_MODEL_PATH):
            calibration_model = joblib.load(CALIBRATION_MODEL_PATH)
            st.sidebar.success("Loaded calibration model")
        
        # For prediction, we'll use the LGBM model (calibration model might be applied separately)
        model = lgbm_model
        
        if model is None:
            st.sidebar.error("No model loaded!")
            
        return model, calibration_model, feature_columns
        
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

# ──────────────────────────────────────────────
#  ENCODING MAPS
# ──────────────────────────────────────────────
EMPLOYMENT_MAP = {
    'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2, 'Retired': 3, 'Student': 4,
    'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3
}
EDUCATION_MAP = {
    'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
}
LOAN_PURPOSE_MAP = {
    'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 
    'Home Loan': 4, 'Personal': 5, 'Debt Consolidation': 6, 
    'Major Purchase': 7, 'Medical': 8, 'Vacation': 9
}

def preprocess_inference_data(input_data, feature_columns=None):
    """Preprocess input data for model prediction"""
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    
    # Basic encoding
    df['employment_status_encoded'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAP).fillna(0).astype(int)
    df['education_level_encoded'] = df.get('education_level', pd.Series()).map(EDUCATION_MAP).fillna(0).astype(int)
    df['loan_purpose_encoded'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAP).fillna(0).astype(int)
    df['collateral_present_encoded'] = df.get('collateral_present', pd.Series()).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Ensure all required columns exist
    required_cols = [
        'annual_income', 'loan_amount', 'num_previous_loans', 'credit_history_length', 
        'num_defaults', 'age', 'current_credit_score', 'avg_payment_delay_days', 'loan_term_months'
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0
    
    # Feature engineering
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['payment_per_month'] = df['loan_amount'] / (df['loan_term_months'] + 1)
    df['payment_to_income'] = df['payment_per_month'] / (df['annual_income'] / 12 + 0.001)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    df['credit_utilization'] = (df['num_previous_loans'] * df['loan_amount']) / (df['annual_income'] + 1)
    df['payment_reliability'] = 1 / (df['avg_payment_delay_days'] + 1)
    df['credit_history_x_score'] = df['credit_history_length'] * df['current_credit_score']
    df['default_x_delay'] = df['num_defaults'] * df['avg_payment_delay_days']
    df['age_x_income'] = df['age'] * df['annual_income'] / 100000
    
    # Create squared and log features
    for col in ['current_credit_score', 'annual_income', 'credit_history_length', 'age']:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Create dummy variables if needed by the model
    if feature_columns:
        # Check if the model expects dummy variables
        if any('employment_status_' in col for col in feature_columns):
            for status in ['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student']:
                df[f'employment_status_{status}'] = (df['employment_status'] == status).astype(int)
        
        if any('education_level_' in col for col in feature_columns):
            for level in ['High School', 'Diploma', 'Bachelor', 'Master', 'PhD']:
                df[f'education_level_{level}'] = (df['education_level'] == level).astype(int)
        
        if any('loan_purpose_' in col for col in feature_columns):
            for purpose in ['Business', 'Crypto-Backed', 'Car Loan', 'Education', 'Home Loan']:
                df[f'loan_purpose_{purpose}'] = (df['loan_purpose'] == purpose).astype(int)
        
        if 'collateral_present_Yes' in feature_columns:
            df['collateral_present_Yes'] = (df['collateral_present'] == 'Yes').astype(int)
            df['collateral_present_No'] = (df['collateral_present'] == 'No').astype(int)
        
        # Credit score categories
        if any('credit_score_category_' in col for col in feature_columns):
            df['credit_score_category_Poor'] = (df['current_credit_score'] < 580).astype(int)
            df['credit_score_category_Fair'] = ((df['current_credit_score'] >= 580) & (df['current_credit_score'] < 670)).astype(int)
            df['credit_score_category_Good'] = ((df['current_credit_score'] >= 670) & (df['current_credit_score'] < 740)).astype(int)
            df['credit_score_category_Very Good'] = ((df['current_credit_score'] >= 740) & (df['current_credit_score'] < 800)).astype(int)
            df['credit_score_category_Excellent'] = (df['current_credit_score'] >= 800).astype(int)
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the feature columns the model expects
        X = df[feature_columns].copy()
    else:
        # If no feature columns provided, use all numeric columns
        X = df.select_dtypes(include=[np.number]).copy()
    
    X = X.fillna(0)
    return X

def predict_single(input_dict):
    """Make a prediction using the loaded models"""
    model, calibration_model, feature_columns = load_model_artifacts()
    
    if model is None:
        raise RuntimeError("No model loaded. Please check the models directory.")
    
    # Preprocess the input data
    processed = preprocess_inference_data(input_dict, feature_columns)
    
    try:
        # Get probability from the model
        if hasattr(model, 'predict_proba'):
            proba = float(model.predict_proba(processed)[:, 1][0])
        else:
            # Fallback for models without predict_proba
            proba = float(model.predict(processed)[0])
        
        # Apply calibration if available (this might be handled differently based on your calibration model)
        if calibration_model is not None:
            try:
                # If calibration_model is a CalibratedClassifierCV or similar
                if hasattr(calibration_model, 'predict_proba'):
                    proba = float(calibration_model.predict_proba(processed)[:, 1][0])
                else:
                    # Assume it's a calibration function/transformer
                    proba = float(calibration_model.predict_proba(processed)[0][1])
            except:
                # If calibration fails, keep the original probability
                pass
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")
    
    # Convert probability to score (0-1000 scale, higher = better)
    score = max(0, min(1000, int(round((1 - proba) * 1000))))
    
    # Assign risk category
    if proba < 0.1:
        cat = "Very Low Risk"
    elif proba < 0.2:
        cat = "Low Risk"
    elif proba < 0.4:
        cat = "Medium Risk"
    elif proba < 0.6:
        cat = "High Risk"
    else:
        cat = "Very High Risk"
    
    return proba, score, cat, processed

# ──────────────────────────────────────────────
#  RISK PALETTE
# ──────────────────────────────────────────────
RISK = {
    "Very Low Risk":  {"hex": "#059669", "bg": "rgba(5, 150, 105, 0.1)", "border": "#059669"},
    "Low Risk":       {"hex": "#10B981", "bg": "rgba(16, 185, 129, 0.1)", "border": "#10B981"},
    "Medium Risk":    {"hex": "#D97706", "bg": "rgba(217, 119, 6, 0.1)", "border": "#D97706"},
    "High Risk":      {"hex": "#EA580C", "bg": "rgba(234, 88, 12, 0.1)", "border": "#EA580C"},
    "Very High Risk": {"hex": "#DC2626", "bg": "rgba(220, 38, 38, 0.1)", "border": "#DC2626"},
}

def risk_color(s):
    if s >= 800: return "#059669"
    if s >= 600: return "#10B981"
    if s >= 400: return "#D97706"
    if s >= 200: return "#EA580C"
    return "#DC2626"

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:0.2rem;">
        <span style="font-family:'Playfair Display',Georgia,serif;font-size:1.55rem;
                     font-weight:600;">CreditIQ</span>
    </div>
    <div style="font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;
                color:var(--text-tertiary);margin-bottom:2rem;">Risk Assessment Platform</div>
    """, unsafe_allow_html=True)

    menu = st.selectbox("Navigate", ["New Assessment", "Assessment History"],
                        label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    # Load models and show status
    model, calibration_model, feature_columns = load_model_artifacts()
    
    # Model status display
    if model is not None:
        fc = len(feature_columns) if feature_columns else "N/A"
        model_status = "Active"
        status_color = "#059669"
        
        # Check if calibration is loaded
        cal_status = "Loaded" if calibration_model is not None else "Not loaded"
    else:
        fc = "—"
        model_status = "Not loaded"
        status_color = "#DC2626"
        cal_status = "Not available"

    st.markdown(f"""
    <div style="background:var(--bg-secondary);border:1px solid var(--border-light);border-radius:8px;padding:12px 14px;">
        <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.06em;color:var(--text-tertiary);margin-bottom:8px;">Model Status</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{status_color};"></div>
            <span style="font-size:0.85rem;font-weight:600;color:var(--text-primary);">LightGBM</span>
        </div>
        <div style="font-size:0.78rem;color:var(--text-tertiary);margin-left:15px;">{fc} features</div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
            <div style="width:7px;height:7px;border-radius:50%;background:#D97706;"></div>
            <span style="font-size:0.78rem;color:var(--text-primary);">Calibration: {cal_status}</span>
        </div>
    </div>
    <div style="margin-top:2rem;font-size:0.7rem;color:var(--text-tertiary);">
        {datetime.now().strftime("%d %b %Y, %H:%M")}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE — NEW ASSESSMENT
# ══════════════════════════════════════════════
if menu == "New Assessment":
    st.markdown("# New Assessment")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;margin:-0.4rem 0 2rem;">Complete the form to generate a credit risk profile.</p>', unsafe_allow_html=True)

    # Check if model is loaded before showing form
    model, _, _ = load_model_artifacts()
    if model is None:
        st.error("No model loaded. Please ensure model files exist in the 'models' directory:")
        st.code("""
        models/
        ├── trained_lgbm_model.pkl
        ├── feature_columns.pkl
        └── calibration_model.pkl
        """)
        st.stop()

    with st.form("assessment_form", clear_on_submit=False):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="section-eyebrow">Applicant Profile</div>', unsafe_allow_html=True)
            applicant_id = st.text_input("Applicant ID", placeholder="e.g. APP-00123")
            applicant_name = st.text_input("Full Name", placeholder="e.g. Kwame Mensah")
            applicant_email = st.text_input("Email Address", placeholder="e.g. kwame@example.com")
            age = st.slider("Age", 18, 100, 30)
            employment_status = st.selectbox("Employment Status",
                                             ["Employed", "Self-Employed", "Unemployed", "Retired", "Student"])
            education_level = st.selectbox("Education Level",
                                           ["High School", "Diploma", "Bachelor", "Master", "PhD"])

        with c2:
            st.markdown('<div class="section-eyebrow">Financial Details (GHS)</div>', unsafe_allow_html=True)
            annual_income = st.number_input("Annual Income", min_value=0, value=50000, step=1000, format="%d")
            loan_amount = st.number_input("Loan Amount", min_value=0, value=25000, step=1000, format="%d")
            loan_purpose = st.selectbox("Loan Purpose",
                                        ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"])
            loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
            collateral_present = st.radio("Collateral Present", ["Yes", "No"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2, gap="large")

        with c3:
            st.markdown('<div class="section-eyebrow">Credit History</div>', unsafe_allow_html=True)
            credit_history_length = st.slider("History Length (years)", 0, 30, 5)
            num_previous_loans = st.slider("Previous Loans", 0, 20, 2)
            num_defaults = st.slider("Number of Defaults", 0, 10, 0)
            current_credit_score = st.slider("Credit Score", 300, 850, 650)

        with c4:
            st.markdown('<div class="section-eyebrow">Payment Behaviour</div>', unsafe_allow_html=True)
            avg_payment_delay_days = st.slider("Avg Payment Delay (days)", 0, 60, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Risk Assessment", type="primary", use_container_width=True)

    if submitted:
        missing = [f for f, v in [("Applicant ID", applicant_id),
                                   ("Full Name", applicant_name),
                                   ("Email", applicant_email)] if not v]
        if annual_income <= 0:
            missing.append("Annual Income > 0")
        if loan_amount <= 0:
            missing.append("Loan Amount > 0")

        if missing:
            st.error(f"Required fields missing: {', '.join(missing)}")
        else:
            with st.spinner("Analysing application with LightGBM model..."):
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
                dh = generate_data_hash(application)
                try:
                    proba, score, category, processed = predict_single(application)
                    st.session_state.current_result = {
                        "proba": proba,
                        "score": score,
                        "category": category,
                        "data": application,
                        "hash": dh
                    }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.cursor().execute("""
                        INSERT OR REPLACE INTO assessment_results
                        (applicant_id, applicant_name, applicant_email, age, data_hash,
                         risk_score, probability_of_default, risk_category, timestamp,
                         annual_income, loan_amount, employment_status, credit_score)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (application["applicant_id"], application["applicant_name"],
                         application["applicant_email"], application["age"], dh, score, proba,
                         category, application["submission_timestamp"], application["annual_income"],
                         application["loan_amount"], application["employment_status"],
                         application["current_credit_score"]))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.warning(f"DB save warning: {e}")
                st.rerun()

    # ── Results ──
    if st.session_state.current_result:
        r = st.session_state.current_result
        proba = r["proba"]
        score = r["score"]
        category = r["category"]
        app = r["data"]
        dh = r["hash"]
        pal = RISK.get(category, RISK["Medium Risk"])

        st.markdown("---")
        st.markdown("### Assessment Results")

        ka, kb, kc = st.columns(3, gap="medium")
        sc = risk_color(score)
        pc = "#059669" if proba < 0.2 else "#D97706" if proba < 0.4 else "#DC2626"

        with ka:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Risk Score</div>
                <div class="kpi-value" style="color:{sc};">{score}</div>
                <div class="kpi-sub">out of 1,000</div>
            </div>""", unsafe_allow_html=True)
        with kb:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Default Probability</div>
                <div class="kpi-value" style="color:{pc};">{proba:.1%}</div>
                <div class="kpi-sub">estimated likelihood</div>
            </div>""", unsafe_allow_html=True)
        with kc:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:{pal['border']};background:{pal['bg']};">
                <div class="kpi-label">Risk Category</div>
                <div class="kpi-value" style="font-size:1.5rem;color:{pal['hex']};">{category}</div>
                <div class="kpi-sub">{app['applicant_name']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Financial Summary")

        inc = app["annual_income"]
        lamt = app["loan_amount"]
        ltv = (lamt / inc * 100) if inc > 0 else 0
        f1, f2, f3, f4 = st.columns(4, gap="medium")
        for col, (lbl, val, sub) in zip([f1, f2, f3, f4], [
            ("Annual Income", f"GHS {inc:,.0f}", ""),
            ("Loan Amount", f"GHS {lamt:,.0f}", ""),
            ("Loan-to-Income", f"{ltv:.1f}%", "ratio"),
            ("Credit Score", str(app["current_credit_score"]), "out of 850"),
        ]):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value" style="font-size:1.4rem;">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Data Integrity")
        st.markdown(f"""
        <div class="hash-block">
            <div class="hash-label">SHA-256 Fingerprint</div>
            <div class="hash-value">{dh}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        v1, v2 = st.columns(2, gap="medium")
        with v1:
            if st.button("Verify Integrity", use_container_width=True):
                if verify_data_hash(app, dh):
                    st.success("Integrity verified — fingerprint matches original submission.")
                else:
                    st.error("Integrity check failed — fingerprint mismatch.")
        with v2:
            if st.button("New Assessment", use_container_width=True):
                st.session_state.current_result = None
                st.rerun()

# ══════════════════════════════════════════════
#  PAGE — ASSESSMENT HISTORY
# ══════════════════════════════════════════════
elif menu == "Assessment History":
    st.markdown("# Assessment History")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;margin:-0.4rem 0 2rem;">Portfolio overview of all processed applications.</p>', unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM assessment_results ORDER BY timestamp DESC", conn)
    except Exception as e:
        st.error(f"Load error: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;background:var(--bg-primary);
                    border:1px dashed var(--border-light);border-radius:12px;margin-top:1rem;">
            <div style="font-size:1.1rem;font-weight:600;color:var(--text-secondary);margin-bottom:0.5rem;">No records yet</div>
            <div style="font-size:0.85rem;color:var(--text-tertiary);">Run an assessment to begin building your history.</div>
        </div>""", unsafe_allow_html=True)
    else:
        high = len(df[df["risk_category"].isin(["High Risk", "Very High Risk"])])
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.metric("Total Applications", len(df))
        with m2:
            st.metric("Avg Risk Score", f"{df['risk_score'].mean():.0f}")
        with m3:
            st.metric("High Risk Cases", high)
        with m4:
            st.metric("Avg Default Probability", f"{df['probability_of_default'].mean():.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Application Records")

        disp = df[["applicant_id", "applicant_name", "risk_score", "risk_category",
                   "probability_of_default", "annual_income", "loan_amount", "timestamp"]].copy()
        disp["probability_of_default"] = disp["probability_of_default"].apply(
            lambda x: f"{float(x):.1%}" if pd.notnull(x) else "—")
        disp["annual_income"] = disp["annual_income"].apply(
            lambda x: f"GHS {float(x):,.0f}" if pd.notnull(x) else "—")
        disp["loan_amount"] = disp["loan_amount"].apply(
            lambda x: f"GHS {float(x):,.0f}" if pd.notnull(x) else "—")
        disp["timestamp"] = pd.to_datetime(disp["timestamp"]).dt.strftime("%d %b %Y  %H:%M")

        st.dataframe(disp, use_container_width=True, hide_index=True,
                     column_config={
                         "applicant_id": "ID",
                         "applicant_name": "Name",
                         "risk_score": st.column_config.ProgressColumn(
                             "Risk Score", min_value=0, max_value=1000, format="%d"),
                         "risk_category": "Category",
                         "probability_of_default": "Default Prob.",
                         "annual_income": "Income",
                         "loan_amount": "Loan",
                         "timestamp": "Date",
                     })

        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            st.download_button("Export to CSV",
                               disp.to_csv(index=False).encode("utf-8"),
                               f"creditiq_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               "text/csv", use_container_width=True)
        with e2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.7rem;font-weight:600;text-transform:uppercase;
            letter-spacing:0.06em;color:var(--text-tertiary);">
    CreditIQ — Risk Intelligence Platform
</div>""", unsafe_allow_html=True)
