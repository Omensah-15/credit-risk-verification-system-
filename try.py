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

# -------------------- Sample Data for Demo --------------------
def create_sample_data():
    """Create sample data for demonstration purposes."""
    sample_data = {
        "age": [35, 42, 29, 51, 37],
        "annual_income": [75000, 52000, 89000, 63000, 71000],
        "employment_status": ["employed", "self-employed", "employed", "unemployed", "employed"],
        "education_level": ["Bachelor", "High School", "Master", "Bachelor", "PhD"],
        "credit_history_length": [7, 3, 5, 12, 9],
        "num_previous_loans": [2, 1, 4, 3, 2],
        "num_defaults": [0, 1, 0, 2, 0],
        "current_credit_score": [720, 650, 780, 600, 750],
        "loan_amount": [25000, 15000, 35000, 20000, 30000],
        "loan_term_months": [36, 24, 48, 36, 60],
        "loan_purpose": ["Home Loan", "Car Loan", "Business", "Education", "Home Loan"],
        "collateral_present": ["Yes", "No", "Yes", "No", "Yes"],
        "identity_verified_on_chain": [1, 0, 1, 0, 1],
        "transaction_consistency_score": [0.9, 0.7, 0.8, 0.6, 0.9],
        "fraud_alert_flag": [0, 0, 0, 1, 0],
        "on_chain_credit_history": [5, 2, 7, 1, 6]
    }
    return pd.DataFrame(sample_data)

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

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    """Verify data against its hash."""
    return generate_data_hash(data) == original_hash

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
def explain_prediction(input_df: pd.DataFrame):
    """Generate SHAP explanation for the prediction."""
    if not SHAP_AVAILABLE:
        return None, None
    
    model = load_model()
    if model is None:
        return None, None
    
    try:
        # Create a TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # For binary classification, we take the values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return explainer, shap_values
    except Exception as e:
        st.error(f"SHAP explanation error: {e}")
        return None, None

def plot_shap_summary(explainer, shap_values, features: pd.DataFrame):
    """Plot SHAP summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    return fig

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
        """Get verification record from blockchain or local ledger."""
        # Try blockchain first
        if self.is_connected() and self.contract:
            try:
                result = self.contract.functions.getVerification(applicant_id).call()
                return {
                    "data_hash": result[0],
                    "risk_score": result[1],
                    "risk_category": result[2],
                    "probability_of_default": float(result[3]) / 10000.0,
                    "timestamp": result[4],
                    "on_chain": True
                }
            except Exception:
                # Fall through to local ledger
                pass
        
        # Local JSON ledger fallback
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
                for entry in reversed(ledger):
                    if entry.get("applicant_id") == applicant_id:
                        entry["on_chain"] = False
                        return entry
            except Exception:
                pass
        
        return {"error": "Not found"}

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
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
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
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🔒 Credit Risk Verification System</h1>", unsafe_allow_html=True)
    
    # Sidebar navigation
    menu = st.sidebar.selectbox(
        "Navigation",
        ["New Verification", "Verification History", "Data Insights", "Blockchain Status", "Model Info"]
    )
    
    # Get blockchain manager
    bm = st.session_state.blockchain_manager
    
    # ---------- New Verification Page ----------
    if menu == "New Verification":
        st.header("📝 New Credit Risk Verification")
        
        # Blockchain status
        if bm.is_connected():
            st.success("✅ Connected to Blockchain")
        else:
            st.warning("⚠️ Using Local Storage (Blockchain not connected)")
        
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
                avg_payment_delay_days = st.slider("Average Payment Delay (days)", 0, 60, 5)
                current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
                loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=25000, step=1000)
                loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
                loan_purpose = st.selectbox("Loan Purpose", ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"])
                collateral_present = st.radio("Collateral Present", ["Yes", "No"])
            
            st.subheader("Blockchain & Fraud Indicators")
            col3, col4 = st.columns(2)
            with col3:
                identity_verified_on_chain = st.radio("Identity Verified on Chain", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
                fraud_alert_flag = st.radio("Fraud Alert Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with col4:
                transaction_consistency_score = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.8)
                on_chain_credit_history = st.slider("On-Chain Credit History", 0, 10, 5)
            
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
                    "avg_payment_delay_days": avg_payment_delay_days,
                    "current_credit_score": current_credit_score,
                    "loan_amount": loan_amount,
                    "loan_term_months": loan_term_months,
                    "loan_purpose": loan_purpose,
                    "collateral_present": collateral_present,
                    "identity_verified_on_chain": identity_verified_on_chain,
                    "transaction_consistency_score": transaction_consistency_score,
                    "fraud_alert_flag": fraud_alert_flag,
                    "on_chain_credit_history": on_chain_credit_history,
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
                
                st.markdown(f"**Data Hash:** `{data_hash}`")
                
                # SHAP explanation
                if SHAP_AVAILABLE:
                    with st.expander("View Explanation (SHAP)"):
                        explainer, shap_values = explain_prediction(processed)
                        if explainer and shap_values is not None:
                            fig = plot_shap_summary(explainer, shap_values, processed)
                            st.pyplot(fig)
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
                                st.success(f"✅ Saved to blockchain! Transaction: {tx_hash}")
                            elif tx_hash == "LOCAL_LEDGER_OK":
                                timestamp = datetime.utcnow().isoformat()
                                save_to_db(applicant_id, applicant_name, applicant_email, age, 
                                          data_hash, score, proba, category, timestamp, "LOCAL")
                                st.info("📝 Saved to local ledger (blockchain not available)")
                            else:
                                st.error(f"❌ Failed to save: {tx_hash}")
    
    # ---------- Verification History Page ----------
    elif menu == "Verification History":
        st.header("📋 Verification History")
        
        # Load data from database
        df = load_from_db()
        
        if df.empty:
            st.info("No verification records found. Run a verification first.")
        else:
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Verifications", len(df))
            with col2:
                avg_score = df['risk_score'].mean() if 'risk_score' in df.columns else 0
                st.metric("Average Risk Score", f"{avg_score:.0f}/1000")
            with col3:
                on_blockchain = len(df[df['tx_hash'].str.startswith('0x', na=False)]) if 'tx_hash' in df.columns else 0
                st.metric("On Blockchain", f"{on_blockchain}/{len(df)}")
            
            # Filter options
            st.subheader("Filter Results")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                risk_filter = st.multiselect(
                    "Risk Category",
                    options=df['risk_category'].unique() if 'risk_category' in df.columns else [],
                    default=df['risk_category'].unique() if 'risk_category' in df.columns else []
                )
            
            with filter_col2:
                blockchain_filter = st.selectbox(
                    "Blockchain Status",
                    options=["All", "On Blockchain", "Local Only"]
                )
            
            # Apply filters
            filtered_df = df.copy()
            if risk_filter and 'risk_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['risk_category'].isin(risk_filter)]
            
            if blockchain_filter == "On Blockchain" and 'tx_hash' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['tx_hash'].str.startswith('0x', na=False)]
            elif blockchain_filter == "Local Only" and 'tx_hash' in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df['tx_hash'].str.startswith('0x', na=False)]
            
            # Display results
            st.subheader("Verification Records")
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    # Determine risk color
                    risk_class = "risk-medium"
                    if "Low" in row.get('risk_category', ''):
                        risk_class = "risk-low"
                    elif "High" in row.get('risk_category', ''):
                        risk_class = "risk-high"
                    
                    # Determine blockchain status
                    blockchain_status = "🔗 On Blockchain" if row.get('tx_hash', '').startswith('0x') else "📝 Local Storage"
                    
                    with st.container():
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>{row.get('applicant_name', 'N/A')} ({row.get('applicant_id', 'N/A')})</h3>
                                <p><b>Risk Score:</b> {row.get('risk_score', 'N/A')}/1000 | 
                                <b>Probability of Default:</b> {float(row.get('probability_of_default', 0)):.2%} | 
                                <span class="{risk_class}"><b>Category:</b> {row.get('risk_category', 'N/A')}</span></p>
                                <p><b>Data Hash:</b> {row.get('data_hash', 'N/A')[:16]}... | 
                                <b>Status:</b> {blockchain_status}</p>
                                <p><b>Timestamp:</b> {row.get('timestamp', 'N/A')}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons for each record
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("View Details", key=f"view_{row['applicant_id']}"):
                                st.session_state.selected_applicant = row['applicant_id']
                        with btn_col2:
                            if st.button("Verify on Blockchain", key=f"verify_{row['applicant_id']}"):
                                if not row.get('tx_hash', '').startswith('0x'):
                                    # Try to save to blockchain
                                    with st.spinner("Saving to blockchain..."):
                                        tx_hash = bm.record_verification(
                                            row['applicant_id'], 
                                            row['data_hash'], 
                                            row['risk_score'], 
                                            row['risk_category'], 
                                            row['probability_of_default']
                                        )
                                        
                                        if tx_hash.startswith("0x"):
                                            # Update database
                                            save_to_db(
                                                row['applicant_id'], row['applicant_name'], row['applicant_email'],
                                                row['age'], row['data_hash'], row['risk_score'],
                                                row['probability_of_default'], row['risk_category'],
                                                row['timestamp'], tx_hash
                                            )
                                            st.success(f"Saved to blockchain! Transaction: {tx_hash}")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to save: {tx_hash}")
                                else:
                                    st.info("Already on blockchain")
                        
                        st.divider()
                
                # Display selected applicant details
                if 'selected_applicant' in st.session_state:
                    selected = df[df['applicant_id'] == st.session_state.selected_applicant].iloc[0]
                    with st.expander("Selected Application Details", expanded=True):
                        st.json(selected.to_dict())
            else:
                st.info("No records match the selected filters.")
            
            # Export data
            st.subheader("Export Data")
            if st.button("Download as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="verification_history.csv",
                    mime="text/csv"
                )
    
    # ---------- Data Insights Page ----------
    elif menu == "Data Insights":
        st.header("📊 Data Insights")
        
        # Load sample data for demonstration
        sample_df = create_sample_data()
        
        st.subheader("Sample Data Overview")
        st.dataframe(sample_df, use_container_width=True)
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.write(sample_df.describe())
        
        # Visualizations
        st.subheader("Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig, ax = plt.subplots()
            sample_df['age'].hist(bins=10, ax=ax)
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            # Credit score distribution
            fig, ax = plt.subplots()
            sample_df['current_credit_score'].hist(bins=10, ax=ax)
            ax.set_title('Credit Score Distribution')
            ax.set_xlabel('Credit Score')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = sample_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = numeric_df.corr()
            im = ax.imshow(corr, cmap='coolwarm', interpolation='nearest')
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            plt.colorbar(im)
            st.pyplot(fig)
        else:
            st.info("No numeric data available for correlation analysis.")
    
    # ---------- Blockchain Status Page ----------
    elif menu == "Blockchain Status":
        st.header("🔗 Blockchain Status")
        
        bm = st.session_state.blockchain_manager
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection Details")
            st.write(f"**Provider URL:** {bm.provider_url}")
            
            if bm.is_connected():
                st.success("✅ Connected to Blockchain")
                try:
                    block_number = bm.w3.eth.block_number
                    st.write(f"**Latest Block:** {block_number}")
                    
                    if bm.account_address:
                        balance = bm.w3.eth.get_balance(bm.account_address)
                        st.write(f"**Account Balance:** {bm.w3.from_wei(balance, 'ether'):.4f} ETH")
                except Exception as e:
                    st.error(f"Error retrieving blockchain data: {e}")
            else:
                st.error("❌ Not connected to Blockchain")
        
        with col2:
            st.subheader("Contract Details")
            if bm.contract_address:
                st.write(f"**Contract Address:** {bm.contract_address}")
                
                if bm.contract and bm.is_connected():
                    try:
                        # Try to call a simple contract method to verify it works
                        dummy_call = bm.contract.functions.getVerification("dummy_id").call()
                        st.success("✅ Contract is accessible")
                    except Exception as e:
                        st.error(f"❌ Contract error: {e}")
                else:
                    st.warning("⚠️ Contract not loaded")
            else:
                st.info("No contract address configured")
        
        # Test connection button
        if st.button("Test Connection"):
            if bm.is_connected():
                st.success("✅ Blockchain connection successful!")
            else:
                st.error("❌ Could not connect to blockchain")
        
        # Local ledger info
        st.subheader("Local Ledger")
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH, "r") as f:
                    ledger = json.load(f)
                st.write(f"**Entries in local ledger:** {len(ledger)}")
                
                if st.button("View Local Ledger"):
                    st.json(ledger)
            except Exception as e:
                st.error(f"Error reading local ledger: {e}")
        else:
            st.info("Local ledger is empty")
    
    # ---------- Model Info Page ----------
    elif menu == "Model Info":
        st.header("🤖 Model Information")
        
        model = load_model()
        
        if model:
            st.success("✅ Model loaded successfully")
            
            # Model details
            st.subheader("Model Details")
            st.write(f"**Model Type:** {type(model).__name__}")
            
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importances")
                
                # Create feature importance plot
                feature_names = [
                    'age', 'annual_income', 'employment_status', 'education_level', 
                    'credit_history_length', 'num_previous_loans', 'num_defaults', 
                    'current_credit_score', 'loan_amount', 'loan_term_months', 
                    'loan_purpose', 'collateral_present', 'identity_verified_on_chain',
                    'transaction_consistency_score', 'fraud_alert_flag', 
                    'on_chain_credit_history', 'income_to_loan_ratio', 
                    'credit_utilization', 'default_rate'
                ]
                
                # Ensure we have the right number of features
                if len(model.feature_importances_) == len(feature_names):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importances')
                    st.pyplot(fig)
                else:
                    st.info("Feature importance dimensions don't match feature names")
            
            # Model parameters
            if hasattr(model, 'get_params'):
                st.subheader("Model Parameters")
                params = model.get_params()
                st.json(params)
        else:
            st.error("❌ Could not load model")
        
        # SHAP availability
        st.subheader("Explanation Capabilities")
        if SHAP_AVAILABLE:
            st.success("✅ SHAP available for model explanations")
        else:
            st.warning("⚠️ SHAP not installed. Install with: pip install shap")

if __name__ == "__main__":
    main()
