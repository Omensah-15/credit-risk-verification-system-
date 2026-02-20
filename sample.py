"""
AI Credit Risk Verification System
Run: streamlit run app.py
"""

import os
import json
import hashlib
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except Exception:
    WEB3_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(
    page_title="Credit Risk Verification",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; }

.page-title { font-size: 1.35rem; font-weight: 600; color: #0F172A; margin-bottom: 0.2rem; }
.page-sub   { font-size: 0.85rem; color: #94A3B8; margin-bottom: 1.5rem; }

.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
.metric-label { font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
.metric-value { font-size: 1.6rem; font-weight: 600; color: #0F172A; line-height: 1.1; }
.metric-sub   { font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; }

.result-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.score-big { font-size: 3rem; font-weight: 600; color: #0F172A; line-height: 1; }

.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-green  { background: #DCFCE7; color: #166534; }
.badge-lime   { background: #F0FDF4; color: #15803D; }
.badge-yellow { background: #FEF9C3; color: #854D0E; }
.badge-orange { background: #FFF7ED; color: #9A3412; }
.badge-red    { background: #FEF2F2; color: #991B1B; }

.gauge-bg { height: 5px; background: #E2E8F0; border-radius: 99px; overflow: hidden; margin: 0.75rem 0 1.25rem; }
.gauge-fg { height: 100%; border-radius: 99px; }

.hash-box {
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    color: #64748B;
    background: #F1F5F9;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    word-break: break-all;
    margin-top: 0.75rem;
}

.hist-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.7rem 1rem;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    background: #FFFFFF;
}
.hist-dot   { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.hist-name  { font-weight: 500; font-size: 0.9rem; color: #0F172A; }
.hist-id    { font-size: 0.72rem; color: #94A3B8; }
.hist-right { margin-left: auto; text-align: right; }
.hist-score { font-weight: 600; font-size: 1rem; color: #0F172A; }
.hist-ts    { font-size: 0.68rem; color: #94A3B8; }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94A3B8;
    margin: 1rem 0 0.4rem;
}

.stButton > button {
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    color: #0F172A !important;
    padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover { background: #F8FAFC !important; border-color: #CBD5E1 !important; }
.stButton > button[kind="primary"] {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border-color: #0F172A !important;
}
.stButton > button[kind="primary"]:hover { background: #1E293B !important; }
.stExpander { border: 1px solid #E2E8F0 !important; border-radius: 8px !important; }
hr { border-color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
MODELS_DIR    = "models"
CONTRACTS_DIR = "contracts"
for d in [DATA_DIR, MODELS_DIR, CONTRACTS_DIR]:
    os.makedirs(d, exist_ok=True)

DB_PATH               = os.path.join(DATA_DIR,      "verification_results.db")
LEDGER_PATH           = os.path.join(CONTRACTS_DIR, "ledger.json")
FEATURE_COLUMNS_FILE  = os.path.join(MODELS_DIR,    "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR,    "calibration_model.pkl")
BASE_MODEL_FILE       = os.path.join(MODELS_DIR,    "trained_lgbm_model.pkl")

# ── Database ───────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verification_results (
            applicant_id TEXT PRIMARY KEY, applicant_name TEXT,
            applicant_email TEXT, age INTEGER, data_hash TEXT,
            risk_score INTEGER, probability_of_default REAL,
            risk_category TEXT, timestamp TEXT, tx_hash TEXT
        )
    """)
    conn.commit(); conn.close()

init_db()

# ── Helpers ────────────────────────────────────────────────────────────────────
def generate_data_hash(data: Dict[str, Any]) -> str:
    copy = {k: v for k, v in data.items() if k not in ("submission_timestamp","timestamp")}
    return hashlib.sha256(
        json.dumps(copy, sort_keys=True, separators=(",",":"), default=str).encode()
    ).hexdigest()

def badge_html(category: str) -> str:
    c = category.lower()
    if "very low"  in c: cls = "badge-green"
    elif "very high" in c: cls = "badge-red"
    elif "low"     in c: cls = "badge-lime"
    elif "medium"  in c: cls = "badge-yellow"
    else:                cls = "badge-orange"
    return f'<span class="badge {cls}">{category}</span>'

def dot_color(category: str) -> str:
    c = category.lower()
    if "very low"  in c: return "#22C55E"
    if "very high" in c: return "#EF4444"
    if "low"       in c: return "#4ADE80"
    if "medium"    in c: return "#EAB308"
    return "#F97316"

def gauge_color(p: float) -> str:
    if p < 0.2: return "#22C55E"
    if p < 0.4: return "#EAB308"
    if p < 0.6: return "#F97316"
    return "#EF4444"

# ── Preprocessing ──────────────────────────────────────────────────────────────
EMP_MAP  = {'employed':0,'self-employed':1,'unemployed':2,'student':3}
EDU_MAP  = {'High School':0,'Diploma':1,'Bachelor':2,'Master':3,'PhD':4}
PURP_MAP = {'Business':0,'Crypto-Backed':1,'Car Loan':2,'Education':3,'Home Loan':4}

def preprocess(input_data) -> pd.DataFrame:
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    df['employment_status']  = df.get('employment_status',  pd.Series(dtype=str)).map(EMP_MAP).fillna(0).astype(int)
    df['education_level']    = df.get('education_level',    pd.Series(dtype=str)).map(EDU_MAP).fillna(0).astype(int)
    df['loan_purpose']       = df.get('loan_purpose',       pd.Series(dtype=str)).map(PURP_MAP).fillna(0).astype(int)
    df['collateral_present'] = df.get('collateral_present', pd.Series(dtype=str)).map({'Yes':1,'No':0}).fillna(0).astype(int)
    for col in ['annual_income','loan_amount','num_previous_loans','credit_history_length','num_defaults']:
        if col not in df.columns: df[col] = 0
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization']   = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate']         = df['num_defaults'] / (df['num_previous_loans'] + 1)
    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median() if not df[c].isnull().all() else 0)
    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError("Feature columns not found. Run train.py first.")
    feat: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    for col in feat:
        if col not in df.columns: df[col] = 0
    return df[feat]

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(CALIBRATED_MODEL_FILE), joblib.load(FEATURE_COLUMNS_FILE)
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_base_model():
    try: return joblib.load(BASE_MODEL_FILE)
    except Exception: return None

def models_ready() -> bool:
    return os.path.exists(CALIBRATED_MODEL_FILE) and os.path.exists(FEATURE_COLUMNS_FILE)

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    model, _ = load_model()
    if model is None: raise RuntimeError("Model not loaded.")
    proc  = preprocess(input_dict)
    proba = float(model.predict_proba(proc)[:, 1][0])
    score = int(round((1 - proba) * 1000))
    if proba < 0.10:   cat = "Very Low Risk"
    elif proba < 0.20: cat = "Low Risk"
    elif proba < 0.40: cat = "Medium Risk"
    elif proba < 0.60: cat = "High Risk"
    else:              cat = "Very High Risk"
    return proba, score, cat, proc

# ── SHAP ───────────────────────────────────────────────────────────────────────
def shap_explain(model, proc, bg=None):
    base = model.calibrated_classifiers_[0].base_estimator if hasattr(model,"calibrated_classifiers_") else model
    try:
        exp  = shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")
        vals = exp.shap_values(proc)
        if isinstance(vals, list): vals = vals[1]
        return vals
    except Exception:
        if bg is not None:
            exp  = shap.KernelExplainer(base.predict_proba, bg.sample(min(50, len(bg))))
            vals = exp.shap_values(proc)
            if isinstance(vals, list): vals = vals[1]
            return vals
        raise

def plot_shap(shap_values, features: pd.DataFrame) -> plt.Figure:
    sv    = shap_values[0]
    pairs = sorted(zip(features.columns.tolist(), sv), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [p[0].replace("_"," ").title() for p in pairs]
    vals  = [p[1] for p in pairs]
    clrs  = ["#EF4444" if v > 0 else "#22C55E" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#F8FAFC")
    ax.barh(names[::-1], vals[::-1], color=clrs[::-1], height=0.55)
    ax.axvline(0, color="#CBD5E1", linewidth=1)
    ax.set_xlabel("SHAP Value", fontsize=9, color="#64748B")
    ax.tick_params(colors="#475569", labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#E2E8F0")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)
    plt.tight_layout(); return fig

def plot_feature_importance(feat_cols: List[str]) -> Optional[plt.Figure]:
    base = load_base_model()
    if base is None or not hasattr(base, "feature_importances_"): return None
    df = pd.DataFrame({"Feature": feat_cols, "Importance": base.feature_importances_}) \
           .sort_values("Importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#F8FAFC")
    ax.barh(df["Feature"][::-1], df["Importance"][::-1], color="#3B82F6", height=0.55)
    ax.tick_params(colors="#475569", labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#E2E8F0")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=9, color="#64748B")
    plt.tight_layout(); return fig

# ── Blockchain / Ledger ────────────────────────────────────────────────────────
class BlockchainManager:
    def __init__(self):
        self.provider_url     = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
        self.account_address  = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key      = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.abi_path         = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3 = None; self.contract = None
        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                if self.contract_address and os.path.exists(self.abi_path):
                    with open(self.abi_path) as f: data = json.load(f)
                    self.contract = self.w3.eth.contract(
                        address=self.contract_address, abi=data.get("abi", data))
            except Exception: self.w3 = None

    def is_connected(self):
        if self.w3:
            try: return self.w3.is_connected()
            except: return False
        return False

    def record(self, applicant_id, data_hash, risk_score, risk_category, pod) -> str:
        if self.w3 and self.contract and self.account_address and self.private_key:
            try:
                nonce = self.w3.eth.get_transaction_count(self.account_address)
                for fn_name, args in [
                    ("storeVerificationResult", (applicant_id, data_hash, int(risk_score), risk_category, int(pod*10000))),
                    ("storeVerification",        (applicant_id, int(risk_score), risk_category, data_hash)),
                ]:
                    try:
                        fn  = getattr(self.contract.functions, fn_name)
                        txn = fn(*args).build_transaction({"from":self.account_address,"nonce":nonce,"gas":300000,"gasPrice":self.w3.eth.gas_price})
                        sig = self.w3.eth.account.sign_transaction(txn, self.private_key)
                        tx  = self.w3.eth.send_raw_transaction(sig.rawTransaction)
                        rec = self.w3.eth.wait_for_transaction_receipt(tx)
                        return rec.transactionHash.hex()
                    except Exception: continue
            except Exception: pass
        entry = {"applicant_id":applicant_id,"data_hash":data_hash,"risk_score":int(risk_score),
                 "risk_category":risk_category,"probability_of_default":float(pod),
                 "timestamp":datetime.utcnow().isoformat()}
        ledger = []
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: ledger = json.load(f)
            except Exception: pass
        ledger.append(entry)
        try:
            with open(LEDGER_PATH,"w") as f: json.dump(ledger, f, indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e: return f"ERROR:{e}"

    def get(self, applicant_id) -> Dict:
        if self.w3 and self.contract:
            for fn in ["getVerificationResult","getVerification"]:
                try:
                    res = getattr(self.contract.functions, fn)(applicant_id).call()
                    return {"data_hash":res[0],"risk_score":res[1],"risk_category":res[2]}
                except Exception: continue
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: ledger = json.load(f)
                for e in reversed(ledger):
                    if e.get("applicant_id") == applicant_id: return e
            except Exception: pass
        return {"error":"Not found"}

    def ledger_count(self):
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: return len(json.load(f))
            except: pass
        return 0

@st.cache_resource(show_spinner=False)
def get_bm(): return BlockchainManager()

# ── Session state ──────────────────────────────────────────────────────────────
if "last_result" not in st.session_state: st.session_state.last_result = None
bm = get_bm()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### CreditAI")
    st.caption("Risk Verification System")
    st.markdown("---")

    menu = st.radio(
        "Navigation",
        ["New Verification", "History", "Insights", "Ledger", "Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
    model_ok = models_ready()
    chain_ok = bm.is_connected()
    st.write(f"Model · {'Ready' if model_ok else 'Not found'}")
    st.write(f"Chain · {'Connected' if chain_ok else 'Local ledger'}")
    if SHAP_AVAILABLE: st.write("SHAP · Available")
    st.markdown("---")
    st.caption(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")


# ═══════════════════════════════════════════════════════════════════
#  NEW VERIFICATION
# ═══════════════════════════════════════════════════════════════════
if menu == "New Verification":
    st.markdown('<div class="page-title">New Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter applicant details to generate a risk assessment.</div>', unsafe_allow_html=True)

    if not model_ok:
        st.error("Model files not found. Run `train.py` to generate them.")
        st.code("python train.py", language="bash")
        st.stop()

    with st.form("form"):
        st.markdown('<div class="section-label">Applicant</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_id   = st.text_input("Applicant ID")
            applicant_name = st.text_input("Full Name")
        with c2:
            applicant_email   = st.text_input("Email")
            employment_status = st.selectbox("Employment", ["employed","self-employed","unemployed","student"])
        with c3:
            education_level = st.selectbox("Education", ["High School","Diploma","Bachelor","Master","PhD"])
            age             = st.slider("Age", 18, 90, 32)

        st.markdown('<div class="section-label">Financials</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            annual_income        = st.number_input("Annual Income (USD)", min_value=0, value=60000, step=5000)
            current_credit_score = st.slider("Credit Score", 300, 850, 680)
        with c5:
            credit_history_length  = st.slider("Credit History (years)", 0, 30, 6)
            num_previous_loans     = st.slider("Previous Loans", 0, 20, 2)
        with c6:
            num_defaults           = st.slider("Defaults", 0, 10, 0)
            avg_payment_delay_days = st.slider("Avg Payment Delay (days)", 0, 90, 5)

        st.markdown('<div class="section-label">Loan</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        with c7:
            loan_amount      = st.number_input("Loan Amount (USD)", min_value=1000, value=30000, step=1000)
            loan_term_months = st.slider("Term (months)", 6, 84, 36)
        with c8:
            loan_purpose       = st.selectbox("Purpose", ["Business","Car Loan","Education","Home Loan","Crypto-Backed"])
            collateral_present = st.radio("Collateral", ["Yes","No"], horizontal=True)
        with c9:
            identity_verified_on_chain = st.radio("On-Chain Identity Verified", ["Yes","No"], horizontal=True)
            fraud_alert_flag           = st.radio("Fraud Alert", ["No","Yes"], horizontal=True)

        st.markdown('<div class="section-label">Blockchain Signals</div>', unsafe_allow_html=True)
        c10, c11 = st.columns(2)
        with c10:
            transaction_consistency_score = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.82)
        with c11:
            on_chain_credit_history = st.slider("On-Chain Credit History (years)", 0, 10, 3)

        st.markdown("")
        submitted = st.form_submit_button("Run Verification", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not applicant_id.strip():    errors.append("Applicant ID is required.")
        if not applicant_name.strip():  errors.append("Full Name is required.")
        if not applicant_email.strip(): errors.append("Email is required.")
        for e in errors: st.error(e)

        if not errors:
            with st.spinner("Running analysis..."):
                app_data = {
                    "applicant_id": applicant_id.strip(),
                    "applicant_name": applicant_name.strip(),
                    "applicant_email": applicant_email.strip(),
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
                    "identity_verified_on_chain": 1 if identity_verified_on_chain == "Yes" else 0,
                    "transaction_consistency_score": float(transaction_consistency_score),
                    "fraud_alert_flag": 1 if fraud_alert_flag == "Yes" else 0,
                    "on_chain_credit_history": int(on_chain_credit_history),
                    "submission_timestamp": datetime.utcnow().isoformat()
                }
                data_hash = generate_data_hash(app_data)

                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        INSERT OR REPLACE INTO verification_results
                        (applicant_id,applicant_name,applicant_email,age,data_hash,
                         risk_score,probability_of_default,risk_category,timestamp,tx_hash)
                        VALUES (?,?,?,?,?,NULL,NULL,NULL,?,NULL)
                    """, (app_data['applicant_id'],app_data['applicant_name'],
                          app_data['applicant_email'],app_data['age'],
                          data_hash,app_data['submission_timestamp']))
                    conn.commit()
                finally:
                    conn.close()

                try:
                    proba, score, category, processed = predict(app_data)
                except FileNotFoundError as e:
                    st.error(str(e)); st.stop()
                except Exception as e:
                    st.error(f"Prediction failed: {e}"); st.stop()

                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""UPDATE verification_results
                        SET risk_score=?,probability_of_default=?,risk_category=?
                        WHERE applicant_id=?""",
                        (score, proba, category, app_data['applicant_id']))
                    conn.commit()
                finally:
                    conn.close()

                st.session_state.last_result = {
                    "data": app_data, "hash": data_hash,
                    "proba": proba, "score": score,
                    "category": category, "processed": processed,
                    "timestamp": datetime.utcnow().isoformat(), "tx": None
                }

    # ── Result display ─────────────────────────────────────────────
    r = st.session_state.last_result
    if r:
        proba    = r['proba']
        score    = r['score']
        category = r['category']
        pct      = proba * 100

        st.markdown(f"""
        <div class="result-box">
          <div style="display:flex;align-items:flex-start;gap:2.5rem;flex-wrap:wrap;">
            <div>
              <div class="metric-label">Risk Score</div>
              <div class="score-big">{score}<span style="font-size:1.1rem;color:#94A3B8;font-weight:400;">/1000</span></div>
            </div>
            <div>
              <div class="metric-label">Default Probability</div>
              <div class="score-big" style="font-size:2.2rem;">{pct:.1f}%</div>
            </div>
            <div>
              <div class="metric-label">Category</div>
              <div style="margin-top:0.6rem;">{badge_html(category)}</div>
            </div>
          </div>
          <div class="gauge-bg">
            <div class="gauge-fg" style="width:{min(pct,100):.1f}%;background:{gauge_color(proba)};"></div>
          </div>
          <div class="hash-box">SHA-256: {r['hash']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        btn_cols = st.columns([1, 1, 4])

        with btn_cols[0]:
            if st.button("Store to Ledger", key="store"):
                with st.spinner("Storing..."):
                    tx = bm.record(r['data']['applicant_id'], r['hash'], r['score'], r['category'], r['proba'])
                if tx.startswith("0x") or tx == "LOCAL_LEDGER_OK":
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        conn.execute("UPDATE verification_results SET tx_hash=? WHERE applicant_id=?",
                                     (tx, r['data']['applicant_id']))
                        conn.commit()
                    finally:
                        conn.close()
                    st.session_state.last_result['tx'] = tx
                    st.success("Stored to local ledger." if tx == "LOCAL_LEDGER_OK" else f"TX: {tx[:20]}...")
                else:
                    st.error(f"Failed: {tx}")

        if SHAP_AVAILABLE:
            with btn_cols[1]:
                if st.button("Explain Prediction", key="shap"):
                    with st.spinner("Computing SHAP..."):
                        try:
                            model, _ = load_model()
                            bg = None
                            csv_p = os.path.join(DATA_DIR, "sample_dataset.csv")
                            if os.path.exists(csv_p):
                                try:
                                    bg_raw = pd.read_csv(csv_p)
                                    bg = preprocess(bg_raw) if len(bg_raw) else None
                                except Exception: bg = None
                            sv  = shap_explain(model, r['processed'], bg)
                            fig = plot_shap(sv, r['processed'])
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                        except Exception as e:
                            st.warning(f"SHAP unavailable: {e}")

        if r.get('tx') and r['tx'] != "LOCAL_LEDGER_OK":
            st.markdown(f'<div class="hash-box">TX Hash: {r["tx"]}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HISTORY
# ═══════════════════════════════════════════════════════════════════
elif menu == "History":
    st.markdown('<div class="page-title">Verification History</div>', unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.info("No verifications yet.")
    else:
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')
        df['risk_score']             = pd.to_numeric(df['risk_score'], errors='coerce')

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Total</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Avg Score</div><div class="metric-value">{df["risk_score"].mean():.0f}</div><div class="metric-sub">out of 1000</div></div>', unsafe_allow_html=True)
        high_ct = df['risk_category'].str.contains('High', na=False).sum()
        c3.markdown(f'<div class="metric-card"><div class="metric-label">High Risk</div><div class="metric-value">{high_ct}</div></div>', unsafe_allow_html=True)
        stored = df['tx_hash'].notnull().sum()
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Stored</div><div class="metric-value">{stored}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        for _, row in df.head(15).iterrows():
            cat   = row.get('risk_category') or "Unknown"
            sc    = int(row['risk_score']) if pd.notnull(row.get('risk_score')) else "—"
            p     = row.get('probability_of_default')
            p_str = f"{float(p):.1%}" if pd.notnull(p) else "—"
            name  = row.get('applicant_name') or "Unknown"
            aid   = row.get('applicant_id') or "—"
            ts    = str(row.get('timestamp') or "")[:16]
            stored_tag = '<span style="font-size:0.68rem;color:#22C55E;margin-left:4px;">stored</span>' if row.get('tx_hash') else ''

            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-dot" style="background:{dot_color(cat)};"></div>
              <div>
                <div class="hist-name">{name}</div>
                <div class="hist-id">{aid}{stored_tag}</div>
              </div>
              <div class="hist-right">
                <div class="hist-score">{sc}<span style="font-size:0.8rem;color:#94A3B8;font-weight:400;">/1000</span></div>
                {badge_html(cat)}
                <div class="hist-ts">{ts} &middot; {p_str} PoD</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        display = df.copy()
        display['probability_of_default'] = display['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else "—"
        )
        st.dataframe(
            display[['applicant_id','applicant_name','risk_score','risk_category',
                      'probability_of_default','timestamp','tx_hash']],
            use_container_width=True, hide_index=True
        )
        st.download_button("Export CSV", df.to_csv(index=False).encode(), "history.csv", "text/csv")

        stored_ids = df[df['tx_hash'].notnull()]['applicant_id'].tolist()
        if stored_ids:
            st.markdown("---")
            st.markdown("**Verify Ledger Record**")
            sel = st.selectbox("Applicant ID", stored_ids)
            if st.button("Fetch Record", key="fetch"):
                rec = bm.get(sel)
                if 'error' in rec: st.error("Not found.")
                else: st.json(rec)


# ═══════════════════════════════════════════════════════════════════
#  INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif menu == "Insights":
    st.markdown('<div class="page-title">Insights</div>', unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        db_df = pd.read_sql_query("SELECT * FROM verification_results", conn)
    finally:
        conn.close()

    if not db_df.empty and db_df['risk_score'].notnull().any():
        db_df['risk_score']             = pd.to_numeric(db_df['risk_score'], errors='coerce')
        db_df['probability_of_default'] = pd.to_numeric(db_df['probability_of_default'], errors='coerce')

        st.markdown("**Portfolio**")
        c_map = {"Very Low Risk":"#22C55E","Low Risk":"#4ADE80","Medium Risk":"#EAB308",
                 "High Risk":"#F97316","Very High Risk":"#EF4444"}
        cats   = db_df['risk_category'].value_counts()
        bar_c  = [c_map.get(c,"#94A3B8") for c in cats.index]

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        fig.patch.set_facecolor("#FFFFFF")
        axes[0].set_facecolor("#F8FAFC")
        axes[0].bar(cats.index, cats.values, color=bar_c, edgecolor="#E2E8F0")
        axes[0].set_title("Risk Categories", fontsize=10, color="#475569")
        axes[0].tick_params(colors="#475569", labelsize=7.5)
        for l in axes[0].get_xticklabels(): l.set_rotation(20)
        for sp in axes[0].spines.values(): sp.set_edgecolor("#E2E8F0")
        axes[0].grid(axis="y", color="#E2E8F0", linewidth=0.5)

        axes[1].set_facecolor("#F8FAFC")
        axes[1].hist(db_df['risk_score'].dropna(), bins=15, color="#3B82F6", edgecolor="#E2E8F0", alpha=0.85)
        axes[1].set_title("Score Distribution", fontsize=10, color="#475569")
        axes[1].set_xlabel("Risk Score", fontsize=9, color="#94A3B8")
        axes[1].tick_params(colors="#475569", labelsize=8)
        for sp in axes[1].spines.values(): sp.set_edgecolor("#E2E8F0")
        axes[1].grid(axis="y", color="#E2E8F0", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    csv_p = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(csv_p):
        try:
            sample = pd.read_csv(csv_p)
            st.markdown("---")
            st.markdown("**Training Dataset**")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="metric-label">Records</div><div class="metric-value">{len(sample):,}</div></div>', unsafe_allow_html=True)
            if 'default_flag' in sample.columns:
                m2.markdown(f'<div class="metric-card"><div class="metric-label">Default Rate</div><div class="metric-value">{sample["default_flag"].mean():.1%}</div></div>', unsafe_allow_html=True)
            if 'current_credit_score' in sample.columns:
                m3.markdown(f'<div class="metric-card"><div class="metric-label">Avg Credit Score</div><div class="metric-value">{sample["current_credit_score"].mean():.0f}</div></div>', unsafe_allow_html=True)

            plot_fields = [(c,l,col) for c,l,col in [
                ('current_credit_score','Credit Score','#3B82F6'),
                ('loan_amount','Loan Amount','#8B5CF6'),
                ('annual_income','Annual Income','#10B981'),
            ] if c in sample.columns]

            if plot_fields:
                fig, axes = plt.subplots(1, len(plot_fields), figsize=(5*len(plot_fields), 3.5))
                if len(plot_fields) == 1: axes = [axes]
                fig.patch.set_facecolor("#FFFFFF")
                for ax, (col, label, clr) in zip(axes, plot_fields):
                    ax.set_facecolor("#F8FAFC")
                    ax.hist(sample[col].dropna(), bins=30, color=clr, edgecolor="#E2E8F0", alpha=0.85)
                    ax.set_xlabel(label, fontsize=9, color="#94A3B8")
                    ax.tick_params(colors="#475569", labelsize=8)
                    for sp in ax.spines.values(): sp.set_edgecolor("#E2E8F0")
                    ax.grid(axis="y", color="#E2E8F0", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with st.expander("Data preview"):
                st.dataframe(sample.head(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load sample dataset: {e}")
    else:
        st.info("No sample dataset found at `data/sample_dataset.csv`.")


# ═══════════════════════════════════════════════════════════════════
#  LEDGER
# ═══════════════════════════════════════════════════════════════════
elif menu == "Ledger":
    st.markdown('<div class="page-title">Immutable Ledger</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Blockchain**")
        st.caption(f"Provider: {bm.provider_url}")
        if bm.is_connected():
            st.success("Connected")
            if WEB3_AVAILABLE and bm.w3:
                try: st.caption(f"Chain ID: {bm.w3.eth.chain_id}  |  Block: {bm.w3.eth.block_number}")
                except Exception: pass
        else:
            st.info("Using local JSON ledger (fallback)")

    with col_r:
        st.markdown("**Account**")
        if bm.account_address:
            st.code(bm.account_address, language=None)
        else:
            st.caption("No account configured. Set ACCOUNT_ADDRESS in .env")
        if bm.contract_address:
            st.code(bm.contract_address, language=None)
        else:
            st.caption("No contract configured. Set CONTRACT_ADDRESS in .env")

    st.markdown("---")
    count = bm.ledger_count()
    st.write(f"Local ledger records: **{count}**")

    if count > 0:
        try:
            with open(LEDGER_PATH) as f: ledger_data = json.load(f)
            st.dataframe(pd.DataFrame(ledger_data), use_container_width=True, hide_index=True)
            st.download_button("Export Ledger CSV",
                pd.DataFrame(ledger_data).to_csv(index=False).encode(), "ledger.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read ledger: {e}")


# ═══════════════════════════════════════════════════════════════════
#  MODEL INFO
# ═══════════════════════════════════════════════════════════════════
elif menu == "Model Info":
    st.markdown('<div class="page-title">Model Information</div>', unsafe_allow_html=True)

    model, feat_cols = load_model()

    if model is None:
        st.error("No model found. Run `train.py` to generate model files.")
        st.code("python train.py", language="bash")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Details**")
            st.write(f"Type: `{type(model).__name__}`")
            st.write(f"Features: `{len(feat_cols)}`")
            st.write(f"SHAP: `{'Available' if SHAP_AVAILABLE else 'Not installed'}`")

        with c2:
            st.markdown("**Files**")
            for label, path in [
                ("Calibrated Model", CALIBRATED_MODEL_FILE),
                ("Base Model",       BASE_MODEL_FILE),
                ("Feature Columns",  FEATURE_COLUMNS_FILE),
            ]:
                exists = os.path.exists(path)
                sz     = f" — {os.path.getsize(path)/1024:.0f} KB" if exists else ""
                st.write(f"{'+ ' if exists else '- '}{label}{sz}")

        fig_fi = plot_feature_importance(feat_cols)
        if fig_fi:
            st.markdown("---")
            st.markdown("**Feature Importance (Top 15)**")
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()

        with st.expander(f"All {len(feat_cols)} features"):
            cols = st.columns(3)
            for i, f in enumerate(feat_cols):
                cols[i % 3].caption(f)

        st.markdown("---")
        st.markdown("**Scoring**")
        st.code("Risk Score = round((1 - P(default)) x 1000)", language=None)
        st.caption("Calibrated via CalibratedClassifierCV on a LightGBM base model.")
