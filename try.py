import os, json, hashlib, sqlite3
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import joblib, pandas as pd, numpy as np, streamlit as st
import matplotlib.pyplot as plt

# ------------------- SHAP -------------------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ------------------- Web3 -------------------
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
except Exception:
    WEB3_AVAILABLE = False

# ------------------- dotenv -------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------- Paths -------------------
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

# ------------------- DB -------------------
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
    )""")
    conn.commit()
    conn.close()

init_db()

# ------------------- Utility -------------------
def generate_data_hash(data: Dict[str, Any]) -> str:
    data_copy = dict(data)
    for transient in ("submission_timestamp", "timestamp"):
        data_copy.pop(transient, None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    return generate_data_hash(data) == original_hash

# ------------------- Preprocessing -------------------
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAPPING = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    df['employment_status'] = df.get('employment_status', pd.Series()).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level'] = df.get('education_level', pd.Series()).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose'] = df.get('loan_purpose', pd.Series()).map(LOAN_PURPOSE_MAPPING).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series()).map({'Yes':1,'No':0}).fillna(0).astype(int)

    for col in ['annual_income','loan_amount','num_previous_loans','credit_history_length','num_defaults']:
        if col not in df.columns: df[col] = 0

    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median = df[c].median() if not df[c].isnull().all() else 0
        df[c] = df[c].fillna(median)

    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(f"{FEATURE_COLUMNS_FILE} missing. Run train.py first.")

    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    for col in feature_columns:
        if col not in df.columns: df[col] = 0

    return df[feature_columns]

# ------------------- Models -------------------
@st.cache_resource
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
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

def predict_single(input_dict: dict) -> Tuple[float,int,str,pd.DataFrame]:
    model,_ = load_models()
    if model is None: raise RuntimeError("Calibrated model not found. Run train.py first.")
    processed = preprocess_inference_data(input_dict)
    proba = float(model.predict_proba(processed)[:,1][0])
    score = int(round((1 - proba) * 1000))
    if proba < 0.1: category="Very Low Risk"
    elif proba<0.2: category="Low Risk"
    elif proba<0.4: category="Medium Risk"
    elif proba<0.6: category="High Risk"
    else: category="Very High Risk"
    return proba,score,category,processed

# ------------------- Blockchain Manager -------------------
class BlockchainManager:
    def __init__(self):
        self.provider_url = os.getenv("WEB3_PROVIDER_URL","")
        self.account_address = os.getenv("ACCOUNT_ADDRESS","")
        self.private_key = os.getenv("PRIVATE_KEY","")
        self.contract_address = os.getenv("CONTRACT_ADDRESS","")
        self.contract_abi_path = os.path.join(CONTRACTS_DIR,"VerificationContract.json")
        self.w3 = None
        self.contract = None
        if WEB3_AVAILABLE and self.provider_url:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                try: self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                except: pass
                if self.contract_address and os.path.exists(self.contract_abi_path):
                    with open(self.contract_abi_path,"r") as f:
                        data=json.load(f)
                        abi=data.get("abi") or data
                        self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
            except Exception as e:
                st.error(f"Web3 init error: {e}")
                self.w3=None; self.contract=None

    def is_connected(self)->bool:
        if self.w3:
            try: return self.w3.is_connected()
            except: return False
        return False

    def record_verification(self, applicant_id:str, data_hash:str, risk_score:int, risk_category:str, probability_of_default:float)->str:
        # fallback local ledger
        entry={"applicant_id":applicant_id,"data_hash":data_hash,"risk_score":risk_score,
               "risk_category":risk_category,"probability_of_default":probability_of_default,
               "timestamp":datetime.utcnow().isoformat(),"stored_locally":True}
        ledger=[]
        if os.path.exists(LEDGER_PATH):
            try: ledger=json.load(open(LEDGER_PATH,"r"))
            except: ledger=[]
        ledger.append(entry)
        try:
            json.dump(ledger, open(LEDGER_PATH,"w"), indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            return f"Error: {e}"

@st.cache_resource
def get_blockchain_manager(): return BlockchainManager()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Credit Risk Verification", layout="wide", page_icon="🔒")
st.title("🔒 Credit Risk Verification")

if 'verification_results' not in st.session_state: st.session_state['verification_results']={}
if 'blockchain_manager' not in st.session_state: st.session_state['blockchain_manager']=get_blockchain_manager()

menu = st.sidebar.selectbox("Navigation", ["New Verification","Verification History","Data Insights","Blockchain Status","Model Info"])

# ------------------- Flexible cards (Samsung tabs style) -------------------
def display_result_cards(results:Dict[str,Any]):
    # Use reorderable cards (simulated with multiselect for flexibility)
    cards = ["Risk Score","Probability of Default","Risk Category"]
    order = st.multiselect("Arrange Cards", cards, default=cards)
    col_widths = [1]*len(order)
    cols = st.columns(col_widths)
    for idx, card in enumerate(order):
        value = results.get(card.replace(" ","_").lower(),"")
        if card=="Probability of Default" and value!="": value=f"{float(value):.2%}"
        cols[idx].metric(card,value)

# ------------------- New Verification -------------------
if menu=="New Verification":
    st.header("New Credit Risk Verification")
    bm = st.session_state.blockchain_manager
    st.success("✅ Connected to Blockchain") if bm.is_connected() else st.warning("⚠️ Using Local Ledger")

    with st.form("verification_form"):
        applicant_id = st.text_input("Applicant ID")
        applicant_name = st.text_input("Full Name")
        applicant_email = st.text_input("Email")
        age = st.slider("Age",18,100,30)
        annual_income = st.number_input("Annual Income",0,1000000,50000)
        loan_amount = st.number_input("Loan Amount",0,1000000,25000)
        employment_status = st.selectbox("Employment Status", ["employed","self-employed","unemployed","student"])
        education_level = st.selectbox("Education Level", ["High School","Diploma","Bachelor","Master","PhD"])
        loan_purpose = st.selectbox("Loan Purpose", ["Business","Crypto-Backed","Car Loan","Education","Home Loan"])
        collateral_present = st.radio("Collateral Present", ["Yes","No"])
        submitted = st.form_submit_button("Run Verification")

    if submitted:
        if not all([applicant_id,applicant_name,applicant_email]): st.error("Provide ID, Name, Email"); st.stop()
        application={"applicant_id":applicant_id,"applicant_name":applicant_name,"applicant_email":applicant_email,
                     "age":age,"annual_income":annual_income,"loan_amount":loan_amount,
                     "employment_status":employment_status,"education_level":education_level,
                     "loan_purpose":loan_purpose,"collateral_present":collateral_present,
                     "submission_timestamp":datetime.utcnow().isoformat()}
        proba,score,category,processed = predict_single(application)
        results={"risk_score":score,"probability_of_default":proba,"risk_category":category}
        display_result_cards(results)
