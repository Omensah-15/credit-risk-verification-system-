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

# Plotting & explainability
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Web3
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except Exception:
    WEB3_AVAILABLE = False

# Dotenv (for optional blockchain credentials)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Credit Risk Verification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Configuration & Paths --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
CONTRACTS_DIR = "contracts"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONTRACTS_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "verification_results.db")
LEDGER_PATH = os.path.join(CONTRACTS_DIR, "ledger.json")
FEATURE_COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR, "calibration_model.pkl")
BASE_MODEL_FILE = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")

# -------------------- Database Initialization --------------------
def init_db():
    """Initialize SQLite database for verification results."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_results (
            applicant_id TEXT PRIMARY KEY,
            applicant_name TEXT,
            applicant_email TEXT,
            age INTEGER,
            data_hash TEXT,
            risk_score INTEGER,
            probability_of_default REAL,
            risk_category TEXT,
            timestamp TEXT,
            tx_hash TEXT
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

@st.cache_resource
def load_base_model():
    """Load base model for feature importance and SHAP."""
    try:
        return joblib.load(BASE_MODEL_FILE)
    except Exception:
        return None

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

# -------------------- SHAP Explanation --------------------
def explain_prediction_with_shap(model, input_df: pd.DataFrame, background_df: Optional[pd.DataFrame] = None, nsample: int = 100):
    """
    Generate SHAP explanations for predictions.
    Returns (explainer, shap_values) with memory-safe background sampling.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP library is not installed. Install with: pip install shap")

    # Extract base model if using calibrated classifier
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].base_estimator
    else:
        base = model

    # Sample background data if provided
    if background_df is not None and len(background_df) > 0:
        background = background_df.sample(min(nsample, len(background_df)))
    else:
        background = None

    # Create explainer and compute SHAP values
    try:
        if background is not None:
            explainer = shap.TreeExplainer(base, data=background, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")
        
        shap_vals = explainer.shap_values(input_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Get positive class SHAP values
    except Exception as e:
        # Fallback to KernelExplainer for non-tree models
        if background is not None:
            explainer = shap.KernelExplainer(base.predict_proba, background)
            shap_vals = explainer.shap_values(input_df)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            raise RuntimeError(f"SHAP explanation failed: {e}")

    return explainer, shap_vals

def plot_shap_waterfall(explainer, shap_values, features: pd.DataFrame, index: int = 0):
    """Create SHAP waterfall plot for a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple)):
            expected = expected[1] if len(expected) > 1 else expected[0]
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[index],
                base_values=expected,
                data=features.iloc[index].values,
                feature_names=features.columns.tolist()
            ),
            show=False
        )
    except Exception:
        # Fallback to summary plot
        try:
            shap.summary_plot(shap_values, features, show=False, plot_type="bar")
        except Exception:
            plt.text(0.5, 0.5, "SHAP visualization unavailable", 
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig

# -------------------- Blockchain Manager --------------------
class BlockchainManager:
    """Manages blockchain interactions and local ledger fallback."""
    
    def __init__(self):
        self.provider_url = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
        self.account_address = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.contract_abi_path = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3: Optional[Web3] = None
        self.contract = None

        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                if self.contract_address and os.path.exists(self.contract_abi_path):
                    with open(self.contract_abi_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        abi = data.get("abi") or data
                    self.contract = self.w3.eth.contract(
                        address=self.contract_address, 
                        abi=abi
                    )
            except Exception:
                self.w3 = None
                self.contract = None

    def is_connected(self) -> bool:
        """Check if connected to blockchain or ledger is available."""
        if self.w3:
            try:
                return self.w3.is_connected()
            except Exception:
                return False
        return True  # Fallback to JSON ledger

    def record_verification(
        self, 
        applicant_id: str, 
        data_hash: str, 
        risk_score: int, 
        risk_category: str, 
        probability_of_default: float
    ) -> str:
        """
        Record verification on blockchain or local ledger.
        Returns transaction hash or status message.
        """
        # Try blockchain storage
        if self.w3 and self.contract and self.account_address and self.private_key:
            try:
                prob_int = int(max(0.0, min(1.0, probability_of_default)) * 10000)
                nonce = self.w3.eth.get_transaction_count(self.account_address)
                
                fn_candidates = [
                    ("storeVerificationResult", (applicant_id, data_hash, int(risk_score), risk_category, prob_int)),
                    ("storeVerification", (applicant_id, int(risk_score), risk_category, data_hash)),
                ]
                
                for fn_name, args in fn_candidates:
                    try:
                        fn = getattr(self.contract.functions, fn_name)
                        txn = fn(*args).build_transaction({
                            "from": self.account_address,
                            "nonce": nonce,
                            "gas": 300000,
                            "gasPrice": self.w3.eth.gas_price
                        })
                        signed = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
                        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
                        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                        return receipt.transactionHash.hex()
                    except Exception:
                        continue
            except Exception:
                pass

        # JSON ledger fallback
        entry = {
            "applicant_id": applicant_id,
            "data_hash": data_hash,
            "risk_score": int(risk_score),
            "risk_category": risk_category,
            "probability_of_default": float(probability_of_default),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        ledger = []
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
            except Exception:
                ledger = []
        
        ledger.append(entry)
        
        try:
            with open(LEDGER_PATH, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_verification(self, applicant_id: str) -> Dict[str, Any]:
        """Retrieve verification record from blockchain or ledger."""
        # Try blockchain
        if self.w3 and self.contract:
            try:
                fn_candidates = ["getVerificationResult", "getVerification"]
                for fn_name in fn_candidates:
                    try:
                        fn = getattr(self.contract.functions, fn_name)
                        res = fn(applicant_id).call()
                        return {
                            "data_hash": res[0],
                            "risk_score": res[1],
                            "risk_category": res[2],
                            "probability_of_default": float(res[3]) / 10000.0 if len(res) > 3 else None,
                            "timestamp": res[4] if len(res) > 4 else None
                        }
                    except Exception:
                        continue
            except Exception:
                pass

        # JSON ledger fallback
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                for entry in reversed(ledger):
                    if entry.get("applicant_id") == applicant_id:
                        return entry
            except Exception:
                pass
        
        return {"error": "Not found"}

@st.cache_resource
def get_blockchain_manager() -> BlockchainManager:
    """Get cached blockchain manager instance."""
    return BlockchainManager()

# -------------------- Session State Initialization --------------------
if 'verification_results' not in st.session_state:
    st.session_state.verification_results = {}
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = get_blockchain_manager()

# -------------------- Main UI --------------------
st.title("Credit Risk Verification System")
st.markdown("---")

# Sidebar navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["New Verification", "Verification History", "Data Insights", "Blockchain Status", "Model Info"]
)

# -------------------- New Verification --------------------
if menu == "New Verification":
    st.header("New Credit Risk Verification")
    
    with st.form("verification_form", clear_on_submit=False):
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

        st.subheader("Blockchain & Verification")
        col3, col4 = st.columns(2)
        
        with col3:
            identity_verified_on_chain = st.radio(
                "Identity Verified On-Chain", 
                [1, 0], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            fraud_alert_flag = st.radio(
                "Fraud Alert Flag", 
                [0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        with col4:
            transaction_consistency_score = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.8)
            on_chain_credit_history = st.slider("On-Chain Credit History", 0, 10, 5)

        submitted = st.form_submit_button("Run Verification", type="primary")

    if submitted:
        if not all([applicant_id, applicant_name, applicant_email]):
            st.error("Please provide Applicant ID, Name, and Email.")
        else:
            with st.spinner("Processing verification..."):
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
                    "identity_verified_on_chain": int(identity_verified_on_chain),
                    "transaction_consistency_score": float(transaction_consistency_score),
                    "fraud_alert_flag": int(fraud_alert_flag),
                    "on_chain_credit_history": int(on_chain_credit_history),
                    "submission_timestamp": datetime.utcnow().isoformat()
                }

                # Generate data hash
                data_hash = generate_data_hash(application)
                
                # Store initial record in database
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO verification_results
                        (applicant_id, applicant_name, applicant_email, age, data_hash, 
                         risk_score, probability_of_default, risk_category, timestamp, tx_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        application['applicant_id'], 
                        application['applicant_name'], 
                        application['applicant_email'], 
                        application['age'],
                        data_hash, None, None, None, 
                        application['submission_timestamp'], None
                    ))
                    conn.commit()
                finally:
                    conn.close()

                # Run prediction
                try:
                    proba, score, category, processed = predict_single(application)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

                # Update database with prediction results
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE verification_results
                        SET risk_score = ?, probability_of_default = ?, risk_category = ?
                        WHERE applicant_id = ?
                    """, (score, proba, category, application['applicant_id']))
                    conn.commit()
                finally:
                    conn.close()

                # Store in session state
                st.session_state.verification_results[applicant_id] = {
                    "data": application,
                    "hash": data_hash,
                    "probability_of_default": proba,
                    "risk_score": score,
                    "risk_category": category,
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Display results
                st.success("Verification completed successfully!")
                st.markdown("### Verification Results")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Score", f"{score}/1000")
                c2.metric("Default Probability", f"{proba:.2%}")
                c3.metric("Risk Category", category)

                st.markdown(f"**Data Hash:** `{data_hash}`")

                # SHAP Explanation
                if SHAP_AVAILABLE:
                    with st.expander("View SHAP Explanation"):
                        try:
                            # Load background data
                            bg = None
                            sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
                            if os.path.exists(sample_csv):
                                try:
                                    bg = pd.read_csv(sample_csv)
                                    if len(bg) > 0:
                                        bg = preprocess_inference_data(bg)
                                except Exception:
                                    bg = None
                            
                            model = load_models()[0]
                            explainer, shap_vals = explain_prediction_with_shap(
                                model,
                                processed,
                                background_df=bg,
                                nsample=100
                            )
                            
                            fig = plot_shap_waterfall(explainer, shap_vals, processed, index=0)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"SHAP explanation unavailable: {e}")
                else:
                    st.info("Install SHAP library for model explanations: pip install shap")

                # Blockchain storage
                if st.button("Store Verification Immutably"):
                    with st.spinner("Storing verification..."):
                        bm = st.session_state.blockchain_manager
                        tx = bm.record_verification(applicant_id, data_hash, score, category, proba)
                        
                        if isinstance(tx, str) and (tx.startswith("0x") or tx == "LOCAL_LEDGER_OK"):
                            try:
                                conn = sqlite3.connect(DB_PATH)
                                cur = conn.cursor()
                                cur.execute(
                                    "UPDATE verification_results SET tx_hash = ? WHERE applicant_id = ?",
                                    (tx, applicant_id)
                                )
                                conn.commit()
                            finally:
                                conn.close()
                            st.success(f"Verification stored: {tx}")
                        else:
                            st.error(f"Storage failed: {tx}")

# -------------------- Verification History --------------------
elif menu == "Verification History":
    st.header("Verification History")
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No verification records yet. Create a new verification to get started.")
    else:
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')

        # Summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Verifications", len(df))
        col2.metric("Average Risk Score", f"{df['risk_score'].mean():.0f}")
        col3.metric("High Risk Count", len(df[df['risk_category'].str.contains('High', na=False)]))
        col4.metric("Blockchain Records", len(df[df['tx_hash'].notnull()]))

        # Display records as cards
        st.subheader("Recent Verifications")
        for idx, row in df.head(10).iterrows():
            cat = row.get('risk_category') or "Unknown"
            score = row.get('risk_score') if row.get('risk_score') is not None else "N/A"
            proba = row.get('probability_of_default')
            proba_str = f"{float(proba):.2%}" if pd.notnull(proba) else "N/A"

            applicant_label = f"{row.get('applicant_name') or 'Unknown'} ({row.get('applicant_id')})"
            data_hash = row.get("data_hash") or "N/A"

            # Color coding based on risk
            if "Very Low" in cat or "Low" in cat:
                bg = "#28a745"
            elif "Medium" in cat:
                bg = "#ffc107"
            elif "High" in cat:
                bg = "#fd7e14"
            elif "Very High" in cat:
                bg = "#dc3545"
            else:
                bg = "#6c757d"

            st.markdown(
                f"""
                <div style="background:{bg}; padding:15px; border-radius:8px; color:white; margin-bottom:10px;">
                    <strong style="font-size:18px;">{applicant_label}</strong><br>
                    <span>Risk Score: {score} | Probability: {proba_str} | Category: {cat}</span><br>
                    <small>Data Hash: {data_hash[:16]}...</small><br>
                    <small>Timestamp: {row.get('timestamp')}</small>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Full data table
        st.subheader("Complete Verification Records")
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else "N/A"
        )

        st.dataframe(
            display_df[[
                'applicant_id', 'applicant_name', 'risk_score', 'risk_category',
                'probability_of_default', 'data_hash', 'timestamp', 'tx_hash'
            ]],
            use_container_width=True
        )

        # Export functionality
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV Export",
            csv,
            "verification_history.csv",
            "text/csv",
            key='download-csv'
        )

        # Blockchain verification
        ids_with_tx = display_df[display_df['tx_hash'].notnull()]['applicant_id'].tolist()
        if ids_with_tx:
            st.subheader("Verify Blockchain Record")
            selected_id = st.selectbox("Select Applicant ID", ids_with_tx)
            if st.button("Fetch from Blockchain/Ledger"):
                bm = st.session_state.blockchain_manager
                res = bm.get_verification(selected_id)
                if 'error' in res:
                    st.error("Verification record not found.")
                else:
                    st.success("Record retrieved successfully")
                    st.json(res)

# -------------------- Data Insights --------------------
elif menu == "Data Insights":
    st.header("Data Insights")
    
    sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(sample_csv):
        try:
            df_sample = pd.read_csv(sample_csv)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df_sample))
            
            if 'default_flag' in df_sample.columns:
                default_rate = df_sample['default_flag'].mean()
                col2.metric("Default Rate", f"{default_rate:.2%}")
            
            if 'current_credit_score' in df_sample.columns:
                avg_score = df_sample['current_credit_score'].mean()
                col3.metric("Avg Credit Score", f"{avg_score:.0f}")
            
            st.subheader("Credit Score Distribution")
            if 'current_credit_score' in df_sample.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df_sample['current_credit_score'].hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
                ax.set_xlabel("Credit Score")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Credit Scores")
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            st.subheader("Loan Amount Distribution")
            if 'loan_amount' in df_sample.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df_sample['loan_amount'].hist(bins=30, ax=ax, color='coral', edgecolor='black')
                ax.set_xlabel("Loan Amount (USD)")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Loan Amounts")
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            st.subheader("Sample Data Preview")
            st.dataframe(df_sample.head(20), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")
    else:
        st.info("Sample dataset not found. Run train.py to generate the dataset.")

# -------------------- Blockchain Status --------------------
elif menu == "Blockchain Status":
    st.header("Blockchain Status")
    
    bm = st.session_state.blockchain_manager
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Connection Details")
        st.write("**Provider URL:**", bm.provider_url)
        
        connected = bm.is_connected()
        if connected:
            st.success("Status: Connected")
        else:
            st.warning("Status: Disconnected")
        
        if WEB3_AVAILABLE and bm.w3:
            try:
                if bm.w3.is_connected():
                    st.write("**Chain ID:**", bm.w3.eth.chain_id)
                    st.write("**Latest Block:**", bm.w3.eth.block_number)
            except Exception as e:
                st.warning(f"Unable to query blockchain: {e}")
    
    with col2:
        st.subheader("Account Information")
        if bm.account_address:
            st.write("**Account Address:**", bm.account_address)
            
            if WEB3_AVAILABLE and bm.w3:
                try:
                    balance = bm.w3.eth.get_balance(bm.account_address)
                    balance_eth = bm.w3.from_wei(balance, "ether")
                    st.metric("Account Balance", f"{balance_eth:.4f} ETH")
                except Exception:
                    st.warning("Unable to retrieve account balance")
        else:
            st.info("No account configured")
        
        if bm.contract_address:
            st.write("**Contract Address:**", bm.contract_address)
        else:
            st.info("No contract configured")
    
    st.markdown("---")
    
    if not (WEB3_AVAILABLE and bm.w3 and bm.contract):
        st.info("**Fallback Mode:** Using local JSON ledger for verification storage.")
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                st.metric("Ledger Records", len(ledger))
            except Exception:
                st.warning("Unable to read ledger file")

# -------------------- Model Info --------------------
elif menu == "Model Info":
    st.header("Model Information")
    
    model, feat_cols = load_models()
    
    if model is None:
        st.warning("No trained model found. Please run train.py first.")
        st.markdown("""
        ### Required Files:
        - `models/calibration_model.pkl` - Calibrated prediction model
        - `models/trained_lgbm_model.pkl` - Base LGBM model
        - `models/feature_columns.pkl` - Feature column names
        """)
    else:
        st.success("Model loaded successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write("**Model Type:**", type(model).__name__)
            st.write("**Number of Features:**", len(feat_cols))
            st.write("**Model File:**", CALIBRATED_MODEL_FILE)
        
        with col2:
            st.subheader("Model Files")
            files_status = {
                "Calibrated Model": os.path.exists(CALIBRATED_MODEL_FILE),
                "Base Model": os.path.exists(BASE_MODEL_FILE),
                "Feature Columns": os.path.exists(FEATURE_COLUMNS_FILE)
            }
            for file_name, exists in files_status.items():
                if exists:
                    st.success(f"{file_name}: Found")
                else:
                    st.error(f"{file_name}: Missing")
        
        st.subheader("Feature List")
        with st.expander("View all features"):
            for i, feat in enumerate(feat_cols, 1):
                st.text(f"{i}. {feat}")
        
        # Feature importance
        base = load_base_model()
        if base is not None and hasattr(base, "feature_importances_"):
            try:
                st.subheader("Top 20 Feature Importances")
                fi = base.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feat_cols,
                    "Importance": fi
                }).sort_values("Importance", ascending=False).head(20)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
                ax.set_xlabel("Importance")
                ax.set_title("Top 20 Feature Importances")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Unable to display feature importances: {e}")
        
        # Model performance info
        st.subheader("Model Usage")
        st.markdown("""
        This model predicts the probability of loan default based on applicant information.
        
        **Risk Score Calculation:**
        - Risk Score = (1 - Probability of Default) × 1000
        - Higher scores indicate lower risk
        - Range: 0 (highest risk) to 1000 (lowest risk)
        
        **Risk Categories:**
        - Very Low Risk: < 10% default probability
        - Low Risk: 10-20% default probability
        - Medium Risk: 20-40% default probability
        - High Risk: 40-60% default probability
        - Very High Risk: > 60% default probability
        """)
