"""
Industry-ready Streamlit app for blockchain-backed credit risk verification.
Features: New verification, history with tx_hash storage, SHAP explanations, CSV export, blockchain/ledger recording.
Requirements: Install dependencies, configure .env, and ensure model/contract files exist.
"""

import os
import json
import hashlib
import sqlite3
import logging
import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, Tuple, Optional, List
from contextlib import contextmanager

import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Plotting & explainability
try:
    import matplotlib.pyplot as plt
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Web3
try:
    from web3 import Web3
    from web3.exceptions import Web3Exception
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

# Dotenv for secure configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -------------------- Configuration & Paths --------------------
DATA_DIR = "data"
MODELS_DIR = "models"
CONTRACTS_DIR = "contracts"
LOG_FILE = os.path.join(DATA_DIR, "app.log")

# Create directories
for directory in [DATA_DIR, MODELS_DIR, CONTRACTS_DIR]:
    os.makedirs(directory, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "verification_results.db")
LEDGER_PATH = os.path.join(CONTRACTS_DIR, "ledger.json")
FEATURE_COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR, "calibration_model.pkl")
BASE_MODEL_FILE = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# -------------------- Dependency Check --------------------
def check_dependencies():
    """Verify all required files and dependencies."""
    required_files = [
        (FEATURE_COLUMNS_FILE, "Feature columns file missing. Run train.py."),
        (CALIBRATED_MODEL_FILE, "Calibrated model missing. Run train.py."),
        (BASE_MODEL_FILE, "Base model missing. Run train.py for SHAP explanations.")
    ]
    for file_path, error_msg in required_files:
        if not os.path.exists(file_path):
            logging.error(error_msg)
            st.error(error_msg)
            st.stop()

    if not WEB3_AVAILABLE:
        logging.warning("web3.py not installed. Using JSON ledger fallback.")
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not installed. Explanations will be disabled.")

# -------------------- Database Management --------------------
@contextmanager
def get_db_connection():
    """Context manager for SQLite connection with pooling."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database with indexes."""
    with get_db_connection() as conn:
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_applicant_id ON verification_results (applicant_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON verification_results (timestamp)")
        conn.commit()
    logging.info("Database initialized.")

init_db()

# -------------------- Utility Functions --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash of applicant data, excluding transient fields."""
    data_copy = dict(data)
    for transient in ("submission_timestamp", "timestamp"):
        data_copy.pop(transient, None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def validate_input(applicant_id: str, applicant_name: str, applicant_email: str, 
                   annual_income: float, loan_amount: float, num_defaults: int, num_previous_loans: int) -> bool:
    """Validate user inputs."""
    if not applicant_id or not re.match(r"^[a-zA-Z0-9_-]{1,50}$", applicant_id):
        st.error("Applicant ID must be 1-50 alphanumeric characters with underscores or hyphens.")
        return False
    if not applicant_name or not re.match(r"^[a-zA-Z\s]{1,100}$", applicant_name):
        st.error("Full name must be 1-100 letters and spaces.")
        return False
    if not applicant_email or not re.match(r"[^@]+@[^@]+\.[^@]+", applicant_email):
        st.error("Valid email is required.")
        return False
    if annual_income < 0 or loan_amount < 0:
        st.error("Income and loan amount must be non-negative.")
        return False
    if num_defaults > num_previous_loans:
        st.error("Number of defaults cannot exceed number of previous loans.")
        return False
    return True

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

@st.cache_data
def preprocess_inference_data(input_data: Any) -> pd.DataFrame:
    """Preprocess input data for model inference."""
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    df['employment_status'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level', pd.Series()).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series()).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    if 'annual_income' not in df.columns:
        df['annual_income'] = 0
    if 'loan_amount' not in df.columns:
        df['loan_amount'] = 0
    if 'num_previous_loans' not in df.columns:
        df['num_previous_loans'] = 0
    if 'credit_history_length' not in df.columns:
        df['credit_history_length'] = 0
    if 'num_defaults' not in df.columns:
        df['num_defaults'] = 0

    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median = df[c].median() if not df[c].isnull().all() else 0
        df[c] = df[c].fillna(median)

    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_FILE}")

    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# -------------------- Model Loading --------------------
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
    """Load trained model and feature columns."""
    try:
        model = joblib.load(CALIBRATED_MODEL_FILE)
        feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
        logging.info("Models loaded successfully.")
        return model, feature_columns
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return None, None

@st.cache_resource
def load_base_model():
    """Load base model for SHAP explanations."""
    try:
        model = joblib.load(BASE_MODEL_FILE)
        logging.info("Base model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Base model loading failed: {e}")
        return None

# -------------------- Prediction Helpers --------------------
def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    """Predict risk for a single applicant."""
    logging.info(f"Starting prediction for applicant_id: {input_dict['applicant_id']}")
    model, feature_columns = load_models()
    if model is None:
        logging.error("Calibrated model not found")
        raise RuntimeError("Calibrated model not found. Run train.py first.")
    processed = preprocess_inference_data(input_dict)
    try:
        proba = float(model.predict_proba(processed)[:, 1][0])
    except Exception as e:
        logging.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Model prediction failed: {e}")
    risk_score = int(round((1 - proba) * 1000))
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
    logging.info(f"Prediction completed: score={risk_score}, category={category}")
    return proba, risk_score, category, processed

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict risk for a batch of applicants."""
    model, feature_columns = load_models()
    if model is None:
        raise RuntimeError("Calibrated model not found. Run train.py first.")
    processed = preprocess_inference_data(df)
    probs = model.predict_proba(processed)[:, 1]
    scores = np.round((1 - probs) * 1000).astype(int)
    def cat(p):
        if p < 0.1: return "Very Low Risk"
        if p < 0.2: return "Low Risk"
        if p < 0.4: return "Medium Risk"
        if p < 0.6: return "High Risk"
        return "Very High Risk"
    cats = [cat(p) for p in probs]
    return pd.DataFrame({"probability_of_default": probs, "risk_score": scores, "risk_category": cats}, index=df.index)

# -------------------- SHAP Helpers --------------------
def get_base_model_for_shap(model):
    """Extract base model for SHAP explanations."""
    if hasattr(model, "base_estimator"):
        return model.base_estimator
    elif hasattr(model, "calibrated_classifiers_"):
        cal = model.calibrated_classifiers_[0]
        return getattr(cal, "estimator", model)
    return model

@lru_cache(maxsize=1)
def get_shap_explainer(model, background_df: Optional[tuple] = None, nsample: int = 100):
    """Create cached SHAP explainer."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed.")
    
    # Convert tuple back to DataFrame if cached
    if background_df is not None:
        background_df = pd.DataFrame(background_df[0], columns=background_df[1])
    
    base = get_base_model_for_shap(model)
    if background_df is not None and len(background_df) > 0:
        if len(background_df) < nsample:
            logging.warning(f"Background dataset too small ({len(background_df)} rows).")
            background = background_df
        else:
            background = background_df.sample(nsample, random_state=42)
        try:
            explainer = shap.TreeExplainer(base, data=background, feature_perturbation="tree_path_dependent")
        except Exception as e:
            logging.warning(f"TreeExplainer failed: {e}. Using KernelExplainer.")
            explainer = shap.KernelExplainer(base.predict_proba, background)
    else:
        try:
            explainer = shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")
        except Exception as e:
            logging.error(f"SHAP explainer creation failed: {e}")
            raise RuntimeError(f"SHAP explainer creation failed: {e}")
    return explainer

def explain_prediction_sampled(model, input_df: pd.DataFrame, background_df: Optional[pd.DataFrame] = None, nsample: int = 100):
    """Generate SHAP explanations with memory safety."""
    # Convert DataFrame to tuple for caching
    bg_tuple = None if background_df is None else (background_df.values, background_df.columns.tolist())
    explainer = get_shap_explainer(model, bg_tuple, nsample)
    try:
        shap_vals = explainer.shap_values(input_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        return explainer, shap_vals
    except Exception as e:
        logging.error(f"SHAP value calculation failed: {e}")
        raise RuntimeError(f"SHAP value calculation failed: {e}")

def plot_shap_decision(explainer, shap_values, features: pd.DataFrame, index: int = 0):
    """Plot SHAP decision plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        expected = explainer.expected_value
        if isinstance(expected, (list, tuple)):
            expected = expected[1] if len(expected) > 1 else expected[0]
        shap.decision_plot(expected, shap_values, features.iloc[index], show=False)
    except Exception:
        try:
            shap.summary_plot(shap_values, features, show=False)
        except Exception:
            plt.text(0.1, 0.5, "SHAP plotting failed", fontsize=12)
    plt.tight_layout()
    return fig

# -------------------- Blockchain Manager --------------------
class BlockchainManager:
    def __init__(self):
        self.provider_url = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
        self.account_address = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.contract_abi_path = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3: Optional[Web3] = None
        self.contract = None
        self.max_retries = 3

        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                if not self.w3.is_connected():
                    raise Web3Exception("Web3 provider not connected")
                if self.contract_address and os.path.exists(self.contract_abi_path):
                    with open(self.contract_abi_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        abi = data.get("abi") or data
                    self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
                logging.info("Blockchain manager initialized.")
            except Exception as e:
                logging.warning(f"Web3 initialization failed: {e}")
                self.w3 = None
                self.contract = None

    def is_connected(self) -> bool:
        """Check blockchain connectivity."""
        if self.w3:
            try:
                return self.w3.is_connected()
            except Exception:
                return False
        return True  # JSON ledger fallback

    def record_verification(self, applicant_id: str, data_hash: str, risk_score: int, 
                            risk_category: str, probability_of_default: float) -> str:
        """Record verification on blockchain or JSON ledger."""
        logging.info(f"Recording verification for applicant_id: {applicant_id}")
        if self.w3 and self.contract and self.account_address and self.private_key:
            for attempt in range(self.max_retries):
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
                            gas_estimate = fn(*args).estimate_gas({"from": self.account_address})
                            txn = fn(*args).build_transaction({
                                "from": self.account_address,
                                "nonce": nonce,
                                "gas": gas_estimate + 10000,
                                "gasPrice": self.w3.eth.gas_price
                            })
                            signed = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
                            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
                            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
                            logging.info(f"Blockchain transaction successful: {receipt.transactionHash.hex()}")
                            return receipt.transactionHash.hex()
                        except Exception as e:
                            logging.warning(f"Transaction attempt {attempt + 1} for {fn_name} failed: {e}")
                            continue
                except Web3Exception as e:
                    logging.error(f"Blockchain storage failed on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        break
                    continue

        # JSON ledger fallback
        entry = {
            "applicant_id": applicant_id,
            "data_hash": data_hash,
            "risk_score": int(risk_score),
            "risk_category": risk_category,
            "probability_of_default": float(probability_of_default),
            "timestamp": datetime.utcnow().isoformat(),
            "ledger_hash": hashlib.sha256(json.dumps({
                "applicant_id": applicant_id,
                "data_hash": data_hash,
                "risk_score": int(risk_score)
            }, sort_keys=True).encode()).hexdigest()
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
            logging.info("Stored in JSON ledger")
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            logging.error(f"JSON ledger storage failed: {e}")
            return f"Error: {str(e)}"

    def get_verification(self, applicant_id: str) -> Dict[str, Any]:
        """Retrieve verification from blockchain or JSON ledger."""
        logging.info(f"Fetching verification for applicant_id: {applicant_id}")
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
            except Web3Exception as e:
                logging.warning(f"Blockchain retrieval failed: {e}")

        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                for entry in reversed(ledger):
                    if entry.get("applicant_id") == applicant_id:
                        expected_hash = hashlib.sha256(json.dumps({
                            "applicant_id": entry["applicant_id"],
                            "data_hash": entry["data_hash"],
                            "risk_score": entry["risk_score"]
                        }, sort_keys=True).encode()).hexdigest()
                        if expected_hash == entry.get("ledger_hash"):
                            return entry
                        else:
                            logging.error("Ledger entry corrupted")
                            return {"error": "Ledger entry corrupted"}
            except Exception as e:
                logging.error(f"Ledger read failed: {e}")
                return {"error": f"Ledger read failed: {e}"}
        return {"error": "Not found"}

# Cached blockchain manager
@st.cache_resource
def get_blockchain_manager() -> BlockchainManager:
    return BlockchainManager()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Credit Risk Verification", layout="wide", page_icon="🔒")
st.markdown("<h1 style='text-align:center;'>🔒 Blockchain-Backed Credit Risk Verification System</h1>", unsafe_allow_html=True)

# Check dependencies on startup
check_dependencies()

# Initialize session state
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = get_blockchain_manager()

menu = st.sidebar.selectbox("Navigation", ["New Verification", "Verification History", "Data Insights", "Blockchain Status", "Model Info"])

# ---------- New Verification ----------
if menu == "New Verification":
    st.header("New Credit Risk Verification")
    with st.form("verification_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            applicant_id = st.text_input("Applicant ID", help="Unique identifier")
            applicant_name = st.text_input("Full name")
            applicant_email = st.text_input("Email")
            age = st.slider("Age", 18, 100, 30)
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed", "student"])
            education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
            credit_history_length = st.slider("Credit history (years)", 0, 30, 5)
        with col2:
            num_previous_loans = st.slider("Number of previous loans", 0, 20, 2)
            num_defaults = st.slider("Number of defaults", 0, 10, 0)
            avg_payment_delay_days = st.slider("Avg payment delay (days)", 0, 60, 5)
            current_credit_score = st.slider("Current credit score", 300, 850, 650)
            loan_amount = st.number_input("Loan amount ($)", min_value=0, value=25000, step=1000)
            loan_term_months = st.slider("Loan term (months)", 12, 84, 36)
            loan_purpose = st.selectbox("Loan purpose", ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"])
            collateral_present = st.radio("Collateral present", ["Yes", "No"])

        st.markdown("**Blockchain & Fraud indicators**")
        col3, col4 = st.columns(2)
        with col3:
            identity_verified_on_chain = st.radio("Identity verified on chain", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            fraud_alert_flag = st.radio("Fraud alert flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col4:
            transaction_consistency_score = st.slider("Transaction consistency", 0.0, 1.0, 0.8)
            on_chain_credit_history = st.slider("On-chain credit history", 0, 10, 5)

        submitted = st.form_submit_button("Run Verification")

    if submitted:
        if not validate_input(applicant_id, applicant_name, applicant_email, annual_income, loan_amount, num_defaults, num_previous_loans):
            st.stop()

        application = {
            "applicant_id": applicant_id,
            "applicant_name": hashlib.sha256(applicant_name.encode()).hexdigest(),
            "applicant_email": hashlib.sha256(applicant_email.encode()).hexdigest(),
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

        data_hash = generate_data_hash(application)
        try:
            with get_db_connection() as conn:
                conn.execute("BEGIN TRANSACTION")
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO verification_results
                    (applicant_id, applicant_name, applicant_email, age, data_hash, risk_score, probability_of_default, risk_category, timestamp, tx_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (application['applicant_id'], application['applicant_name'], application['applicant_email'], application['age'],
                      data_hash, None, None, None, application['submission_timestamp'], None))
                conn.commit()

                proba, score, category, processed = predict_single(application)
                
                cur.execute("""
                    UPDATE verification_results
                    SET risk_score = ?, probability_of_default = ?, risk_category = ?
                    WHERE applicant_id = ?
                """, (score, proba, category, application['applicant_id']))
                conn.commit()
        except FileNotFoundError as e:
            st.error(str(e))
            logging.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Operation failed: {e}")
            logging.error(f"Operation failed: {e}")
            st.stop()

        st.markdown("### Result")
        c1, c2, c3 = st.columns([1,1,1])
        c1.metric("Risk Score", score)
        c2.metric("Probability of Default", f"{proba:.2%}")
        c3.metric("Risk Category", category)
        st.markdown(f"**Data Hash:** `{data_hash[:12]}...`")

        if SHAP_AVAILABLE:
            with st.expander("View SHAP explanation (sampled)"):
                sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
                bg = pd.read_csv(sample_csv) if os.path.exists(sample_csv) else None
                try:
                    model = load_models()[0]
                    explainer, shap_vals = explain_prediction_sampled(model, processed, background_df=bg, nsample=100)
                    fig = plot_shap_decision(explainer, shap_vals, processed, index=0)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP explanation not available: {e}")
                    logging.warning(f"SHAP explanation failed: {e}")
        else:
            st.info("SHAP not installed — install shap for explanations.")

        bm = st.session_state.blockchain_manager
        if st.button("Store immutably (blockchain/ledger)"):
            with st.spinner("Storing verification..."):
                tx = bm.record_verification(applicant_id, data_hash, score, category, proba)
                if isinstance(tx, str) and (tx.startswith("0x") or tx == "LOCAL_LEDGER_OK"):
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("UPDATE verification_results SET tx_hash = ? WHERE applicant_id = ?", (tx, applicant_id))
                        conn.commit()
                    st.success(f"Stored with transaction hash: {tx}")
                else:
                    st.error(f"Storage failed: {tx}")

# ---------- Verification History ----------
elif menu == "Verification History":
    st.header("Verification History")
    with get_db_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)

    if df.empty:
        st.info("No verification records yet. Run 'New Verification' to add.")
    else:
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')

        st.markdown("### Summary Cards")
        for idx, row in df.iterrows():
            cat = row.get('risk_category') or "Unknown"
            score = row.get('risk_score') if row.get('risk_score') is not None else ""
            proba = row.get('probability_of_default')
            proba_str = f"{float(proba):.2%}" if pd.notnull(proba) else ""
            applicant_label = f"{row.get('applicant_name') or ''} ({row.get('applicant_id')})"
            data_hash = row.get("data_hash") or "N/A"
            tx_hash = row.get("tx_hash") or "N/A"
            
            display_data_hash = data_hash[:12] + "..." if len(data_hash) > 12 else data_hash
            display_tx_hash = tx_hash[:12] + "..." if tx_hash and len(tx_hash) > 12 else tx_hash

            if "Very Low" in cat or "Low" in cat:
                bg = "#1e7e34"
            elif "Medium" in cat:
                bg = "#f1c40f"
            elif "High" in cat:
                bg = "#e67e22"
            elif "Very High" in cat:
                bg = "#c0392b"
            else:
                bg = "#7f8c8d"

            st.markdown(
                f"""
                <div style="background:{bg}; padding:12px; border-radius:10px; color:white; margin-bottom:8px;">
                    <strong style="font-size:16px;">{applicant_label}</strong><br>
                    <span>Risk Score: {score} &nbsp; | &nbsp; Probability: {proba_str} &nbsp; | &nbsp; Category: {cat}</span><br>
                    <small>Data Hash: {display_data_hash}</small><br>
                    <small>Tx Hash: {display_tx_hash}</small><br>
                    <small>Timestamp: {row.get('timestamp')}</small>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("### Full Table")
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else ""
        )
        display_df['data_hash'] = display_df['data_hash'].apply(lambda x: x[:12] + "..." if x and len(x) > 12 else x)
        display_df['tx_hash'] = display_df['tx_hash'].apply(lambda x: x[:12] + "..." if x and len(x) > 12 else x)

        st.dataframe(
            display_df[['applicant_id', 'applicant_name', 'risk_score', 'risk_category',
                        'probability_of_default', 'data_hash', 'timestamp', 'tx_hash']],
            use_container_width=True
        )

        st.markdown("### Export Options")
        export_cols = st.multiselect(
            "Select columns to export",
            options=df.columns.tolist(),
            default=['applicant_id', 'risk_score', 'risk_category', 'probability_of_default', 'data_hash', 'timestamp', 'tx_hash']
        )
        csv = df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV of History", csv, "verification_history.csv", "text/csv")

        ids_with_tx = df[df['tx_hash'].notnull()]['applicant_id'].tolist()
        if ids_with_tx:
            st.markdown("### Verify Stored Record")
            selected_id = st.selectbox("Select applicant ID with a tx", ids_with_tx)
            if st.button("Fetch verification from chain/ledger"):
                bm = st.session_state.blockchain_manager
                res = bm.get_verification(selected_id)
                if 'error' in res:
                    st.error(res['error'])
                else:
                    db_row = df[df['applicant_id'] == selected_id].iloc[0]
                    if res['data_hash'] == db_row['data_hash'] and res['risk_score'] == db_row['risk_score']:
                        st.success("Record verified: matches blockchain/ledger.")
                    else:
                        st.error("Verification mismatch between database and blockchain/ledger.")
                    st.json(res)

# ---------- Data Insights ----------
elif menu == "Data Insights":
    st.header("Data Insights")
    sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(sample_csv):
        try:
            df_sample = pd.read_csv(sample_csv)
            st.metric("Total Records", len(df_sample))
            st.metric("Default Rate", f"{df_sample['default_flag'].mean():.2%}" if 'default_flag' in df_sample.columns else "N/A")
            st.subheader("Credit Score Distribution")
            fig, ax = plt.subplots()
            df_sample['current_credit_score'].hist(bins=30, ax=ax)
            ax.set_xlabel("Credit Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not load sample dataset: {e}")
            logging.error(f"Sample dataset loading failed: {e}")
    else:
        st.info("Sample dataset not found. Run train.py to generate one.")

# ---------- Blockchain Status ----------
elif menu == "Blockchain Status":
    st.header("Blockchain Status")
    bm = st.session_state.blockchain_manager
    st.write("Provider URL:", bm.provider_url)
    connected = bm.is_connected()
    st.write("Connected:", connected)
    if connected and WEB3_AVAILABLE and bm.w3:
        try:
            st.write("Chain ID:", bm.w3.eth.chain_id)
            if bm.account_address:
                bal = bm.w3.eth.get_balance(bm.account_address)
                st.write("Account balance (ETH):", bm.w3.from_wei(bal, "ether"))
        except Exception as e:
            st.warning(f"Could not query chain details: {e}")
            logging.warning(f"Chain details query failed: {e}")
    else:
        st.info("Using JSON ledger fallback (no web3 or contract configured).")

# ---------- Model Info ----------
elif menu == "Model Info":
    st.header("Model Information")
    model, feat_cols = load_models()
    if model is None:
        st.info("No trained/calibrated model found. Run train.py first.")
    else:
        st.success("Model loaded")
        st.write("Feature columns (count = %d):" % len(feat_cols))
        st.write(feat_cols)
        base = load_base_model()
        if base is not None and hasattr(base, "feature_importances_"):
            try:
                fi = getattr(base, "feature_importances_")
                fi_df = pd.DataFrame({"feature": feat_cols, "importance": fi}).sort_values("importance", ascending=False).head(30)
                st.subheader("Top feature importances")
                st.bar_chart(fi_df.set_index("feature")["importance"])
            except Exception as e:
                st.warning(f"Feature importance plotting failed: {e}")
                logging.warning(f"Feature importance plotting failed: {e}")

if __name__ == "__main__":
    main()
