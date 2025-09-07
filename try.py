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

# Plotting & explainability
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Web3 for blockchain
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

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
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")

# -------------------- DB Initialization --------------------
def init_db():
    """Initialize the SQLite database."""
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

# -------------------- Utility: Data Hashing --------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash of applicant data."""
    data_copy = dict(data)
    for transient in ("submission_timestamp", "timestamp"):
        data_copy.pop(transient, None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# -------------------- Preprocessing --------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess input data for model prediction."""
    df = pd.DataFrame([input_data])
    
    # Map categorical variables
    df['employment_status'] = df['employment_status'].map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df['education_level'].map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df['loan_purpose'].map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df['collateral_present'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Engineered features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    
    # Ensure all expected features are present
    expected_features = [
        'age', 'annual_income', 'employment_status', 'education_level', 
        'credit_history_length', 'num_previous_loans', 'num_defaults', 
        'current_credit_score', 'loan_amount', 'loan_term_months', 
        'loan_purpose', 'collateral_present', 'identity_verified_on_chain',
        'transaction_consistency_score', 'fraud_alert_flag', 
        'on_chain_credit_history', 'income_to_loan_ratio', 
        'credit_utilization', 'default_rate'
    ]
    
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[expected_features]

# -------------------- Model Loading --------------------
@st.cache_resource
def load_model():
    """Load the trained model or create a demo model if not available."""
    try:
        if os.path.exists(MODEL_FILE):
            return joblib.load(MODEL_FILE)
        else:
            # Create a simple demo model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=1000, n_features=19, n_informative=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Save demo model
            joblib.dump(model, MODEL_FILE)
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------- Prediction --------------------
def predict_default(input_dict: Dict[str, Any]) -> Tuple[float, int, str, pd.DataFrame]:
    """Predict probability of default."""
    model = load_model()
    if model is None:
        return 0.5, 500, "Medium Risk", pd.DataFrame()
    
    processed = preprocess_inference_data(input_dict)
    
    # Predict probability
    try:
        proba = float(model.predict_proba(processed)[0, 1])
    except Exception:
        # Fallback calculation for demo purposes
        proba = min(0.95, max(0.05, 
            (input_dict['num_defaults'] * 0.2 + 
             (850 - input_dict['current_credit_score']) / 1000 +
             (input_dict['loan_amount'] / input_dict['annual_income']) * 0.3)
        ))
    
    # Calculate risk score (0-1000, higher is better)
    risk_score = int(round((1 - proba) * 1000))
    
    # Determine risk category
    if proba < 0.2:
        category = "Very Low Risk"
    elif proba < 0.4:
        category = "Low Risk"
    elif proba < 0.6:
        category = "Medium Risk"
    elif proba < 0.8:
        category = "High Risk"
    else:
        category = "Very High Risk"
    
    return proba, risk_score, category, processed

# -------------------- SHAP Explanation --------------------
def explain_prediction_shap(input_df: pd.DataFrame):
    """Generate SHAP explanation for the prediction."""
    if not SHAP_AVAILABLE:
        return None
    
    model = load_model()
    if model is None:
        return None
    
    try:
        # Create a TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # For binary classification, we take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, input_df, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"SHAP explanation error: {e}")
        return None

# -------------------- Blockchain Manager --------------------
class BlockchainManager:
    def __init__(self):
        self.provider_url = os.getenv("WEB3_PROVIDER_URL", "https://mainnet.infura.io/v3/your-project-id")
        self.account_address = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.contract_abi_path = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3 = None
        self.contract = None
        
        if WEB3_AVAILABLE:
            self.setup_web3()
    
    def setup_web3(self):
        """Set up Web3 connection."""
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
            
            # Add middleware for POA chains if needed
            try:
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except Exception:
                pass
            
            # Load contract if available
            if self.contract_address and os.path.exists(self.contract_abi_path):
                with open(self.contract_abi_path, "r") as f:
                    abi = json.load(f)
                self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
        except Exception as e:
            st.error(f"Web3 setup error: {e}")
    
    def is_connected(self):
        """Check if connected to blockchain."""
        return self.w3 and self.w3.is_connected() if self.w3 else False
    
    def record_verification(self, applicant_id: str, data_hash: str, 
                           risk_score: int, risk_category: str, 
                           probability_of_default: float) -> str:
        """Record verification on blockchain or local ledger."""
        # Try blockchain first
        if self.is_connected() and self.contract and self.account_address and self.private_key:
            try:
                # Convert probability to integer (scale by 10000)
                prob_int = int(probability_of_default * 10000)
                
                # Build transaction
                nonce = self.w3.eth.get_transaction_count(self.account_address)
                txn = self.contract.functions.storeVerification(
                    applicant_id, data_hash, risk_score, risk_category, prob_int
                ).build_transaction({
                    "from": self.account_address,
                    "nonce": nonce,
                    "gas": 300000,
                    "gasPrice": self.w3.eth.gas_price
                })
                
                # Sign and send transaction
                signed_txn = self.w3.eth.account.sign_transaction(txn, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for receipt
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt.status == 1:
                    return tx_hash.hex()
                else:
                    return "TX_FAILED"
            except Exception as e:
                error_msg = f"Blockchain error: {str(e)}"
                st.error(error_msg)
                # Fall through to local ledger
        
        # Local JSON ledger fallback
        return self.record_local(applicant_id, data_hash, risk_score, risk_category, probability_of_default)
    
    def record_local(self, applicant_id: str, data_hash: str, 
                    risk_score: int, risk_category: str, 
                    probability_of_default: float) -> str:
        """Record verification in local JSON ledger."""
        entry = {
            "applicant_id": applicant_id,
            "data_hash": data_hash,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "probability_of_default": probability_of_default,
            "timestamp": datetime.utcnow().isoformat(),
            "stored_locally": True
        }
        
        ledger = []
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
            except Exception:
                ledger = []
        
        ledger.append(entry)
        
        try:
            with open(LEDGER_PATH, "w") as f:
                json.dump(ledger, f, indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            return f"LOCAL_ERROR: {str(e)}"
    
    def get_verification(self, applicant_id: str) -> Dict[str, Any]:
        """Get verification from blockchain or local ledger."""
        # Try blockchain first
        if self.is_connected() and self.contract:
            try:
                result = self.contract.functions.getVerification(applicant_id).call()
                # Convert back from stored format
                return {
                    "applicant_id": result[0],
                    "data_hash": result[1],
                    "risk_score": result[2],
                    "risk_category": result[3],
                    "probability_of_default": result[4] / 10000.0,
                    "stored_on_chain": True
                }
            except Exception:
                # Fall through to local ledger
                pass
        
        # Check local ledger
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
                
                for entry in ledger:
                    if entry.get("applicant_id") == applicant_id:
                        return entry
            except Exception:
                pass
        
        return {"error": "Verification not found"}

# -------------------- Database Operations --------------------
def save_to_db(applicant_id: str, applicant_name: str, applicant_email: str, 
               age: int, data_hash: str, risk_score: int, 
               probability_of_default: float, risk_category: str, 
               timestamp: str, tx_hash: str):
    """Save verification result to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO verification_results 
            (applicant_id, applicant_name, applicant_email, age, data_hash, 
             risk_score, probability_of_default, risk_category, timestamp, tx_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (applicant_id, applicant_name, applicant_email, age, data_hash,
              risk_score, probability_of_default, risk_category, timestamp, tx_hash))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def load_from_db():
    """Load all verification results from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database load error: {e}")
        return pd.DataFrame()

# -------------------- Streamlit UI --------------------
def main():
    # Initialize session state
    if 'verification_results' not in st.session_state:
        st.session_state.verification_results = {}
    if 'blockchain_manager' not in st.session_state:
        st.session_state.blockchain_manager = BlockchainManager()
    
    # Page configuration
    st.set_page_config(
        page_title="Credit Risk Verification",
        layout="wide",
        page_icon="🔒",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .risk-low {
            color: #2ecc71;
            font-weight: bold;
        }
        .risk-medium {
            color: #f39c12;
            font-weight: bold;
        }
        .risk-high {
            color: #e74c3c;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .history-item {
            border-left: 4px solid #1f77b4;
            padding-left: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'> Credit Risk Verification</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation
    menu = st.sidebar.selectbox(
        "Navigation",
        ["New Verification", "Verification History", "Blockchain Status", "Model Info"]
    )
    
    # Get blockchain manager
    bm = st.session_state.blockchain_manager
    
    # ---------- New Verification Page ----------
    if menu == "New Verification":
        st.header("📝 New Credit Risk Verification")
        
        # Blockchain status
        if bm.is_connected():
            st.success("Connected to Blockchain")
        else:
            st.warning("Using Local Storage (Blockchain not connected)")
        
        with st.form("verification_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Applicant Information")
                applicant_id = st.text_input("Applicant ID*", help="Unique identifier for the applicant")
                applicant_name = st.text_input("Full Name*")
                applicant_email = st.text_input("Email Address*")
                age = st.slider("Age", 18, 100, 30)
                annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
                employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed", "student"])
                education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
                credit_history_length = st.slider("Credit History (years)", 0, 30, 5)
            
            with col2:
                st.subheader("Loan Information")
                num_previous_loans = st.slider("Number of Previous Loans", 0, 20, 2)
                num_defaults = st.slider("Number of Defaults", 0, 10, 0)
                current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
                loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=25000, step=1000)
                loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
                loan_purpose = st.selectbox("Loan Purpose", ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"])
                collateral_present = st.radio("Collateral Present", ["Yes", "No"])
            
            st.subheader("Blockchain Indicators")
            col3, col4 = st.columns(2)
            with col3:
                identity_verified_on_chain = st.radio("Identity Verified on Chain", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            with col4:
                on_chain_credit_history = st.slider("On-Chain Credit History", 0, 10, 5)
            
            # Add missing fields with default values
            transaction_consistency_score = 0.8
            fraud_alert_flag = 0
            
            submitted = st.form_submit_button("Run Verification")
        
        if submitted:
            # Validate required fields
            if not all([applicant_id, applicant_name, applicant_email]):
                st.error("Please fill in all required fields (marked with *)")
            else:
                # Create application data
                application = {
                    "applicant_id": applicant_id,
                    "applicant_name": applicant_name,
                    "applicant_email": applicant_email,
                    "age": age,
                    "annual_income": annual_income,
                    "employment_status": employment_status,
                    "education_level": education_level,
                    "credit_history_length": credit_history_length,
                    "num_previous_loans": num_previous_loans,
                    "num_defaults": num_defaults,
                    "current_credit_score": current_credit_score,
                    "loan_amount": loan_amount,
                    "loan_term_months": loan_term_months,
                    "loan_purpose": loan_purpose,
                    "collateral_present": collateral_present,
                    "identity_verified_on_chain": identity_verified_on_chain,
                    "on_chain_credit_history": on_chain_credit_history,
                    "transaction_consistency_score": transaction_consistency_score,
                    "fraud_alert_flag": fraud_alert_flag,
                    "submission_timestamp": datetime.utcnow().isoformat()
                }
                
                # Generate data hash
                data_hash = generate_data_hash(application)
                
                # Run prediction
                with st.spinner("Calculating risk assessment..."):
                    proba, score, category, processed = predict_default(application)
                
                # Display results
                st.success("Risk assessment completed!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{score}/1000")
                with col2:
                    st.metric("Probability of Default", f"{proba:.2%}")
                with col3:
                    # Color code based on risk
                    if "Low" in category:
                        st.markdown(f"<p class='risk-low'>Risk Category: {category}</p>", unsafe_allow_html=True)
                    elif "Medium" in category:
                        st.markdown(f"<p class='risk-medium'>Risk Category: {category}</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p class='risk-high'>Risk Category: {category}</p>", unsafe_allow_html=True)
                
                st.markdown(f"**Data Hash:** `{data_hash[:16]}...`")
                
                # SHAP explanation
                if SHAP_AVAILABLE:
                    with st.expander("View Explanation (SHAP)"):
                        shap_fig = explain_prediction_shap(processed)
                        if shap_fig:
                            st.pyplot(shap_fig)
                        else:
                            st.info("SHAP explanation not available for this prediction.")
                else:
                    st.info("Install SHAP package for model explanations.")
                
                # Save options
                st.subheader("Save Results")
                save_col1, save_col2 = st.columns(2)
                
                with save_col1:
                    if st.button("💾 Save to Database", use_container_width=True):
                        timestamp = datetime.utcnow().isoformat()
                        if save_to_db(applicant_id, applicant_name, applicant_email, age, 
                                     data_hash, score, proba, category, timestamp, ""):
                            st.success("Results saved to database!")
                        else:
                            st.error("Failed to save to database.")
                
                with save_col2:
                    if st.button("🔗 Save to Blockchain", use_container_width=True):
                        with st.spinner("Saving to blockchain..."):
                            tx_hash = bm.record_verification(applicant_id, data_hash, score, category, proba)
                            
                            if tx_hash.startswith("0x"):
                                # Update database with tx hash
                                timestamp = datetime.utcnow().isoformat()
                                save_to_db(applicant_id, applicant_name, applicant_email, age, 
                                          data_hash, score, proba, category, timestamp, tx_hash)
                                st.success(f"Saved to blockchain! Transaction: {tx_hash[:16]}...")
                            elif tx_hash == "LOCAL_LEDGER_OK":
                                timestamp = datetime.utcnow().isoformat()
                                save_to_db(applicant_id, applicant_name, applicant_email, age, 
                                          data_hash, score, proba, category, timestamp, "LOCAL")
                                st.info("📝 Saved to local ledger (blockchain not available)")
                            else:
                                st.error(f"Failed to save: {tx_hash}")
    
    # ---------- Verification History ----------
    elif menu == "Verification History":
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

                st.markdown(
                    f"""
                    <div style="background:{bg}; padding:12px; border-radius:10px; color:white; margin-bottom:8px;">
                        <strong style="font-size:16px;">{applicant_label}</strong><br>
                        <span>Risk Score: {score} &nbsp; | &nbsp; Probability: {proba_str} &nbsp; | &nbsp; Category: {cat}</span><br>
                        <small>Data Hash: {data_hash}</small><br>
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
            ids_with_tx = display_df[display_df['tx_hash'].notnull()]['applicant_id'].tolist()
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
    
    # ---------- Blockchain Status Page ----------
    elif menu == "Blockchain Status":
        st.header("🔗 Blockchain Status")
        
        bm = st.session_state.blockchain_manager
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection")
            st.write(f"**Provider:** {bm.provider_url}")
            
            if bm.is_connected():
                st.success("Connected to Blockchain")
                try:
                    block_number = bm.w3.eth.block_number
                    st.write(f"**Latest Block:** {block_number}")
                except Exception as e:
                    st.error(f"Error retrieving blockchain data: {e}")
            else:
                st.error("Not connected to Blockchain")
        
        with col2:
            st.subheader("Contract")
            if bm.contract_address:
                st.write(f"**Address:** {bm.contract_address}")
                
                if bm.contract and bm.is_connected():
                    try:
                        st.success("Contract is accessible")
                    except Exception as e:
                        st.error(f"Contract error: {e}")
                else:
                    st.warning("Contract not loaded")
            else:
                st.info("No contract address configured")
        
        # Test connection button
        if st.button("Test Connection"):
            if bm.is_connected():
                st.success("Blockchain connection successful!")
            else:
                st.error("Could not connect to blockchain")
    
    # ---------- Model Info Page ----------
    elif menu == "Model Info":
        st.header("Model Information")
        
        model = load_model()
        if model:
            st.subheader("Model Details")
            st.write(f"**Model Type:** {type(model).__name__}")
            
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importances")
                try:
                    # Get feature names
                    feature_names = [
                        'age', 'annual_income', 'employment_status', 'education_level', 
                        'credit_history_length', 'num_previous_loans', 'num_defaults', 
                        'current_credit_score', 'loan_amount', 'loan_term_months', 
                        'loan_purpose', 'collateral_present', 'identity_verified_on_chain',
                        'transaction_consistency_score', 'fraud_alert_flag', 
                        'on_chain_credit_history', 'income_to_loan_ratio', 
                        'credit_utilization', 'default_rate'
                    ]
                    
                    # Create feature importance chart
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.set_title("Feature Importances")
                    ax.barh(range(len(indices)), importances[indices], align="center")
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels([feature_names[i] for i in indices])
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate feature importance chart: {e}")
        else:
            st.error("Model not available")

if __name__ == "__main__":
    main()
