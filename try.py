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

# plotting & explainability
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# web3
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except Exception:
    WEB3_AVAILABLE = False

# dotenv (for optional blockchain credentials)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- Configuration & paths --------------------
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
BASE_MODEL_FILE = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")  # optional for SHAP

# -------------------- DB init --------------------
def init_db():
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

# -------------------- Utility: deterministic hashing --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Deterministic SHA-256 hash of applicant data.
    Excludes transient fields like submission_timestamp/timestamp.
    """
    data_copy = dict(data)
    for transient in ("submission_timestamp", "timestamp"):
        data_copy.pop(transient, None)
    # canonical JSON
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    return generate_data_hash(data) == original_hash

# -------------------- Preprocessing (must match training) --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    """
    Accepts dict (single row) or DataFrame (batch). Returns DataFrame aligned with training features.
    Will raise FileNotFoundError if FEATURE_COLUMNS_FILE is missing.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Safe mapping (if column missing, create it)
    df['employment_status'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level', pd.Series()).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAPPING).fillna(0).astize(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series()).map({'Yes': 1, 'No': 0}).fillna(0).astize(int)

    # engineered features
    # provide defaults if columns missing
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

    # ensure numeric columns are numeric and fillna with median
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median = df[c].median() if not df[c].isnull().all() else 0
        df[c] = df[c].fillna(median)

    # load expected feature columns
    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_FILE}. Run train.py first.")

    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    # add missing cols with zero
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # return aligned DF
    return df[feature_columns]

# -------------------- Model loading (cached) --------------------
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
    """
    Returns (calibrated_model, feature_columns) or (None, None)
    """
    try:
        model = joblib.load(CALIBRATED_MODEL_FILE)
        feature_columns = joblib.load(FEATURE_COLUMNS_FILE)
        return model, feature_columns
    except Exception:
        return None, None

def load_base_model():
    try:
        return joblib.load(BASE_MODEL_FILE)
    except Exception:
        return None

# -------------------- Prediction helpers --------------------
def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    model, feature_columns = load_models()
    if model is None:
        raise RuntimeError("Calibrated model not found. Run train.py first.")
    processed = preprocess_inference_data(input_dict)
    # predict_proba -> class 1 probability
    try:
        proba = float(model.predict_proba(processed)[:, 1][0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")
    risk_score = int(round((1 - proba) * 1000))  # higher = safer
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

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
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

# -------------------- SHAP helpers (robust) --------------------
def get_base_model_for_shap(model):
    """
    Safely extract the underlying model for SHAP.
    Works with CalibratedClassifierCV or normal models.
    """
    # For CalibratedClassifierCV, extract the base estimator
    if hasattr(model, "calibrated_classifiers_"):
        # Access the base estimator from the first calibrated classifier
        return model.calibrated_classifiers_[0].estimator
    # For normal models, just return the model
    return model

def explain_prediction_sampled(model, input_df: pd.DataFrame, background_df: Optional[pd.DataFrame] = None, nsample: int = 100):
    """
    Returns (explainer, shap_values) with memory-safe background sampling.
    Works with tree-based or non-tree models.
    """
    if not SHAP_AVAILABLE:
        raise RuntimeError("shap is not installed or failed to import.")

    # Get base model for SHAP explanation
    base_model = get_base_model_for_shap(model)
    
    # Sample background
    if background_df is not None and len(background_df) > 0:
        background = background_df.sample(min(nsample, len(background_df)))
    else:
        background = None

    # Select proper TreeExplainer config
    try:
        if background is not None:
            explainer = shap.TreeExplainer(base_model, data=background, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.TreeExplainer(base_model, feature_perturbation="tree_path_dependent")
        shap_vals = explainer.shap_values(input_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception as e:
        # fallback to KernelExplainer for non-tree models
        if background is not None:
            explainer = shap.KernelExplainer(base_model.predict_proba, background)
            shap_vals = explainer.shap_values(input_df)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            raise RuntimeError(f"SHAP explanation failed: {e}")

    return explainer, shap_vals


def plot_shap_decision(explainer, shap_values, features: pd.DataFrame, index: int = 0):
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

# -------------------- Blockchain manager (web3 + JSON fallback) --------------------
class BlockchainManager:
    def __init__(self):
        # config from env or defaults
        self.provider_url = os.getenv("WEB3_PROVIDER_URL", "https://mainnet.infura.io/v3/your-project-id")
        self.account_address = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.contract_abi_path = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3: Optional[Web3] = None
        self.contract = None

        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                
                # Add middleware for POA chains if needed
                try:
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except Exception:
                    pass
                
                if self.contract_address and os.path.exists(self.contract_abi_path):
                    with open(self.contract_abi_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        abi = data.get("abi") or data
                    self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
            except Exception as e:
                st.error(f"Web3 connection error: {e}")
                self.w3 = None
                self.contract = None

    def is_connected(self) -> bool:
        if self.w3:
            try:
                return self.w3.is_connected()
            except Exception:
                return False
        return False

    def record_verification(self, applicant_id: str, data_hash: str, risk_score: int, risk_category: str, probability_of_default: float) -> str:
        """
        Tries to store on-chain. If not available or fails, appends to local JSON ledger.
        Returns transaction hash (hex) or "LOCAL_LEDGER_OK" or error message.
        """
        # try on-chain if we have contract + keys
        if self.w3 and self.contract and self.account_address and self.private_key:
            try:
                # Convert probability to integer (scale by 10000 to preserve decimals)
                prob_int = int(probability_of_default * 10000)
                
                # Get nonce
                nonce = self.w3.eth.get_transaction_count(self.account_address)
                
                # Build transaction
                txn = self.contract.functions.storeVerification(
                    applicant_id, 
                    data_hash, 
                    int(risk_score), 
                    risk_category, 
                    prob_int
                ).build_transaction({
                    "from": self.account_address,
                    "nonce": nonce,
                    "gas": 300000,
                    "gasPrice": self.w3.eth.gas_price
                })
                
                # Sign transaction
                signed_txn = self.w3.eth.account.sign_transaction(txn, private_key=self.private_key)
                
                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                
                if receipt.status == 1:
                    return tx_hash.hex()
                else:
                    return f"Transaction failed with status {receipt.status}"
                    
            except Exception as e:
                error_msg = f"Blockchain transaction failed: {str(e)}"
                st.error(error_msg)
                # Fall through to local ledger

        # JSON fallback ledger (only if blockchain fails)
        entry = {
            "applicant_id": applicant_id,
            "data_hash": data_hash,
            "risk_score": int(risk_score),
            "risk_category": risk_category,
            "probability_of_default": float(probability_of_default),
            "timestamp": datetime.utcnow().isoformat(),
            "stored_locally": True
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
        # try on-chain first
        if self.w3 and self.contract:
            try:
                result = self.contract.functions.getVerification(applicant_id).call()
                
                # Parse result based on contract structure
                # Assuming structure: (dataHash, riskScore, category, probInt, timestamp)
                return {
                    "data_hash": result[0],
                    "risk_score": result[1],
                    "risk_category": result[2],
                    "probability_of_default": float(result[3]) / 10000.0,
                    "timestamp": result[4],
                    "on_chain": True
                }
            except Exception as e:
                st.warning(f"On-chain retrieval failed: {e}")
                # Fall through to local ledger

        # JSON fallback:
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    ledger = json.load(f)
                for entry in reversed(ledger):
                    if entry.get("applicant_id") == applicant_id:
                        entry["on_chain"] = False
                        return entry
            except Exception as e:
                st.error(f"Ledger read error: {e}")
        return {"error": "Not found"}

# cached blockchain manager
@st.cache_resource
def get_blockchain_manager() -> BlockchainManager:
    return BlockchainManager()

# initialize session-state
if 'verification_results' not in st.session_state:
    st.session_state.verification_results = {}
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = get_blockchain_manager()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Credit Risk Verification", layout="wide", page_icon="🔒")
st.markdown("<h1 style='text-align:center;'>🔒 Credit Risk Verification</h1>", unsafe_allow_html=True)

menu = st.sidebar.selectbox("Navigation", ["New Verification", "Verification History", "Data Insights", "Blockchain Status", "Model Info"])

# ---------- New Verification ----------
if menu == "New Verification":
    st.header("New Credit Risk Verification")
    
    # Blockchain status indicator
    bm = st.session_state.blockchain_manager
    if bm.is_connected():
        st.success("✅ Connected to Blockchain")
    else:
        st.warning("⚠️ Using Local Ledger (Blockchain not connected)")
    
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
        # basic validation
        if not all([applicant_id, applicant_name, applicant_email]):
            st.error("Please provide Applicant ID, Name and Email.")
        else:
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

            # generate data hash and pre-store minimal row in sqlite
            data_hash = generate_data_hash(application)
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR REPLACE INTO verification_results
                    (applicant_id, applicant_name, applicant_email, age, data_hash, risk_score, probability_of_default, risk_category, timestamp, tx_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (application['applicant_id'], application['applicant_name'], application['applicant_email'], application['age'],
                      data_hash, None, None, None, application['submission_timestamp'], ""))  # Initialize tx_hash as empty string
                conn.commit()
            finally:
                conn.close()

            # run prediction
            try:
                proba, score, category, processed = predict_single(application)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            # update sqlite with prediction
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

            # store in session for immediate UI
            st.session_state.verification_results[applicant_id] = {
                "data": application,
                "hash": data_hash,
                "probability_of_default": proba,
                "risk_score": score,
                "risk_category": category,
                "timestamp": datetime.utcnow().isoformat()
            }

            # pretty metric cards
            st.markdown("### Result")
            c1, c2, c3 = st.columns([1,1,1])
            c1.metric("Risk Score", score)
            c2.metric("Probability of Default", f"{proba:.2%}")
            c3.metric("Risk Category", category)

            st.markdown(f"**Data Hash:** `{data_hash[:12]}...`")

            # SHAP explanation - optional & memory-safe
            if SHAP_AVAILABLE:
                with st.expander("View SHAP explanation (sampled)"):
                    # try to load sample background
                    bg = None
                    sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
                    if os.path.exists(sample_csv):
                        try:
                            bg_df = pd.read_csv(sample_csv)
                            # Preprocess the background data to match the training format
                            bg = preprocess_inference_data(bg_df)
                        except Exception as e:
                            st.warning(f"Could not load background data: {e}")
                            bg = None
                    try:
                        model = load_models()[0]  # get your trained model
                        # explain prediction safely
                        explainer, shap_vals = explain_prediction_sampled(
                            model,
                            processed,
                            background_df=bg,
                            nsample=100
                        )
                        fig = plot_shap_decision(explainer, shap_vals, processed, index=0)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP explanation not available: {e}")
            else:
                st.info("SHAP not installed — install shap for explanations.")

            # Store to blockchain/ledger immediately after prediction
            bm = st.session_state.blockchain_manager
            with st.spinner("Storing verification on blockchain..."):
                tx = bm.record_verification(applicant_id, data_hash, score, category, proba)
                if tx.startswith("0x"):
                    # update tx_hash in sqlite
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute("UPDATE verification_results SET tx_hash = ? WHERE applicant_id = ?", (tx, applicant_id))
                        conn.commit()
                    finally:
                        conn.close()
                    st.success(f"✅ Successfully stored on blockchain: {tx}")
                elif tx == "LOCAL_LEDGER_OK":
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute("UPDATE verification_results SET tx_hash = ? WHERE applicant_id = ?", ("LOCAL", applicant_id))
                        conn.commit()
                    finally:
                        conn.close()
                    st.warning("⚠️ Stored in local ledger (blockchain unavailable)")
                else:
                    st.error(f"❌ Storage failed: {tx}")

# ---------- Verification History ----------
elif menu == "Verification History" or menu == "Verification History".replace(" ", ""):
    st.header("Verification History")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No verification records yet. Run 'New Verification' to add.")
    else:
        # normalize col types
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')

        # pretty cards
        st.markdown("### Summary Cards")
        for idx, row in df.iterrows():
            cat = row.get('risk_category') or "Unknown"
            score = row.get('risk_score') if row.get('risk_score') is not None else ""
            proba = row.get('probability_of_default')
            if pd.isna(proba):
                proba_str = ""
            else:
                proba_str = f"{float(proba):.2%}"

            applicant_label = f"{row.get('applicant_name') or ''} ({row.get('applicant_id')})"
            data_hash = row.get("data_hash") or "N/A"
            tx_hash = row.get("tx_hash") or "Not stored"

            # color selection
            if "Very Low" in cat or "Low" in cat:
                bg = "#1e7e34"  # green
            elif "Medium" in cat:
                bg = "#f1c40f"  # yellow
            elif "High" in cat:
                bg = "#e67e22"  # orange
            elif "Very High" in cat:
                bg = "#c0392b"  # red
            else:
                bg = "#7f8c8d"  # gray

            # Different border for blockchain vs local storage
            border_style = "5px solid #27ae60" if tx_hash.startswith("0x") else "5px solid #f39c12" if tx_hash == "LOCAL" else "5px solid #95a5a6"
            
            st.markdown(
                f"""
                <div style="background:{bg}; padding:12px; border-radius:10px; color:white; margin-bottom:8px; border:{border_style}">
                    <strong style="font-size:16px;">{applicant_label}</strong><br>
                    <span>Risk Score: {score} &nbsp; | &nbsp; Probability: {proba_str} &nbsp; | &nbsp; Category: {cat}</span><br>
                    <small>Data Hash: {data_hash}</small><br>
                    <small>Transaction Hash: {tx_hash}</small><br>
                    <small>Timestamp: {row.get('timestamp')}</small>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("### Full Table")
        # show table including hash
        display_df = df.copy()
        display_df['probability_of_default'] = display_df['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else ""
        )

        st.dataframe(
            display_df[['applicant_id','applicant_name','risk_score','risk_category',
                        'probability_of_default','data_hash','timestamp','tx_hash']],
            use_container_width=True
        )

        # CSV export
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV of History", csv, "verification_history.csv", "text/csv")

        # select & verify on-chain/ledger if tx exists
        ids_with_tx = display_df[display_df['tx_hash'].notnull() & (display_df['tx_hash'] != "")]['applicant_id'].tolist()
        if ids_with_tx:
            st.markdown("### Verify Stored Record")
            selected_id = st.selectbox("Select applicant ID with a tx", ids_with_tx)
            if st.button("Fetch verification from chain/ledger"):
                bm = st.session_state.blockchain_manager
                res = bm.get_verification(selected_id)
                if 'error' in res:
                    st.error("Verification not found on chain/ledger.")
                else:
                    st.success("Record retrieved")
                    st.json(res)

# ---------- Data Insights ----------
elif menu == "Data Insights":
    st.header("Data Insights")
    sample_csv = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(sample_csv):
        try:
            df_sample = pd.read_csv(sample_csv)
            st.metric("Total Records", len(df_sample))
            st.metric("Default Rate", f"{df_sample['default_flag'].mean():.2%}")
            st.subheader("Credit Score Distribution")
            fig, ax = plt.subplots()
            df_sample['current_credit_score'].hist(bins=30, ax=ax)
            ax.set_xlabel("Credit Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not load sample dataset: {e}")
    else:
        st.info("Sample dataset not found. Run train.py to generate one.")

# ---------- Blockchain Status ----------
elif menu == "Blockchain Status":
    st.header("Blockchain Status")
    bm = st.session_state.blockchain_manager
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Provider URL:**", bm.provider_url)
        connected = bm.is_connected()
        status = "✅ Connected" if connected else "❌ Disconnected"
        st.write("**Connection Status:**", status)
        
        if connected and WEB3_AVAILABLE and bm.w3:
            try:
                st.write("**Chain ID:**", bm.w3.eth.chain_id)
                if bm.account_address:
                    bal = bm.w3.eth.get_balance(bm.account_address)
                    st.write("**Account balance (ETH):**", bm.w3.from_wei(bal, "ether"))
            except Exception as e:
                st.warning(f"Could not query chain details: {e}")
    
    with col2:
        if bm.contract_address:
            st.write("**Contract Address:**", bm.contract_address)
        else:
            st.write("**Contract Address:** Not configured")
            
        if bm.account_address:
            st.write("**Account Address:**", bm.account_address)
        else:
            st.write("**Account Address:** Not configured")
    
    # Test connection button
    if st.button("Test Blockchain Connection"):
        if bm.is_connected():
            st.success("✅ Blockchain connection successful!")
            try:
                block = bm.w3.eth.get_block('latest')
                st.write(f"Latest block number: {block.number}")
                st.write(f"Block timestamp: {datetime.fromtimestamp(block.timestamp)}")
            except Exception as e:
                st.error(f"Error getting block info: {e}")
        else:
            st.error("❌ Could not connect to blockchain")
            st.info("Make sure to set WEB3_PROVIDER_URL, ACCOUNT_ADDRESS, PRIVATE_KEY, and CONTRACT_ADDRESS in your environment variables")

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
            except Exception:
                pass
