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
import matplotlib.patches as mpatches

# ── Optional deps ──────────────────────────────────────────────────────────────
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

# ── MUST be first Streamlit call ───────────────────────────────────────────────
st.set_page_config(
    page_title="CreditAI · Risk Verification",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Dark financial-grade aesthetic
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #E2E8F0;
}
.stApp {
    background: #080C14;
}
section[data-testid="stSidebar"] {
    background: #0D1520;
    border-right: 1px solid #1E2D42;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; }

/* ── Typography ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #1E2D42;
    margin-bottom: 2rem;
}
.top-bar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #38BDF8;
}
.top-bar-logo span { color: #64748B; }
.top-bar-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    background: #1E3A5F;
    color: #38BDF8;
    border: 1px solid #2563EB44;
    border-radius: 4px;
    padding: 3px 10px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1.8rem;
    font-weight: 300;
}

/* ── Cards ── */
.card {
    background: #0D1520;
    border: 1px solid #1E2D42;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #38BDF8;
}

/* ── KPI tiles ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-tile {
    background: #0D1520;
    border: 1px solid #1E2D42;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38BDF8, #6366F1);
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
}
.risk-very-low  { background: #052E16; color: #4ADE80; border: 1px solid #166534; }
.risk-low       { background: #052E16; color: #86EFAC; border: 1px solid #166534; }
.risk-medium    { background: #422006; color: #FCD34D; border: 1px solid #92400E; }
.risk-high      { background: #431407; color: #FB923C; border: 1px solid #9A3412; }
.risk-very-high { background: #450A0A; color: #F87171; border: 1px solid #991B1B; }

/* ── Result panel ── */
.result-panel {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    background: #0D1520;
    border: 1px solid #1E2D42;
    position: relative;
    overflow: hidden;
}
.result-panel::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748B;
    margin-bottom: 0.5rem;
}
.result-score {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-hash {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #334155;
    word-break: break-all;
    background: #060A10;
    border: 1px solid #1E2D42;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 1rem;
}

/* ── Gauge bar ── */
.gauge-wrap { margin: 1rem 0; }
.gauge-track {
    height: 8px;
    border-radius: 99px;
    background: #1E2D42;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* ── History card ── */
.hist-card {
    background: #0D1520;
    border: 1px solid #1E2D42;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.hist-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.hist-name { font-weight: 500; font-size: 0.95rem; }
.hist-id { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #64748B; }
.hist-meta { margin-left: auto; text-align: right; }
.hist-score { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.2rem; }
.hist-ts { font-size: 0.7rem; color: #64748B; font-family: 'DM Mono', monospace; }

/* ── Sidebar nav ── */
.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 2px;
    font-size: 0.9rem;
    transition: background 0.15s;
}
.nav-item:hover { background: #1E2D42; }
.nav-item.active { background: #1E3A5F; color: #38BDF8; }

/* ── Streamlit widget overrides ── */
div[data-testid="stForm"] { background: transparent !important; border: none !important; }
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSlider > div { background: #0D1520 !important; border-color: #1E2D42 !important; color: #E2E8F0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #4F46E5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0EA5E9, #6366F1) !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
}
div[data-testid="metric-container"] {
    background: #0D1520 !important;
    border: 1px solid #1E2D42 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
.stAlert { border-radius: 10px !important; }
.stDataFrame { background: #0D1520 !important; border-radius: 10px !important; }
.stExpander { background: #0D1520 !important; border: 1px solid #1E2D42 !important; border-radius: 10px !important; }
label { color: #94A3B8 !important; font-size: 0.82rem !important; letter-spacing: 0.03em !important; }

/* ── Sidebar brand ── */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #38BDF8;
    padding: 1rem 0 0.5rem;
    letter-spacing: -0.02em;
}
.sidebar-brand span { color: #334155; }
.sidebar-version {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #334155;
    margin-bottom: 1.5rem;
}
.sidebar-section {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #334155;
    margin: 1.2rem 0 0.4rem;
}
.info-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    background: #1E3A5F22;
    color: #38BDF8;
    border: 1px solid #1E3A5F;
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS & DIRS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR      = "data"
MODELS_DIR    = "models"
CONTRACTS_DIR = "contracts"
for d in [DATA_DIR, MODELS_DIR, CONTRACTS_DIR]:
    os.makedirs(d, exist_ok=True)

DB_PATH               = os.path.join(DATA_DIR, "verification_results.db")
LEDGER_PATH           = os.path.join(CONTRACTS_DIR, "ledger.json")
FEATURE_COLUMNS_FILE  = os.path.join(MODELS_DIR, "feature_columns.pkl")
CALIBRATED_MODEL_FILE = os.path.join(MODELS_DIR, "calibration_model.pkl")
BASE_MODEL_FILE       = os.path.join(MODELS_DIR, "trained_lgbm_model.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data_hash(data: Dict[str, Any]) -> str:
    copy = {k: v for k, v in data.items() if k not in ("submission_timestamp", "timestamp")}
    canonical = json.dumps(copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()

def risk_color_class(category: str) -> str:
    c = category.lower()
    if "very low" in c:  return "risk-very-low"
    if "very high" in c: return "risk-very-high"
    if "low" in c:       return "risk-low"
    if "medium" in c:    return "risk-medium"
    if "high" in c:      return "risk-high"
    return "risk-medium"

def risk_dot_color(category: str) -> str:
    c = category.lower()
    if "very low" in c:  return "#4ADE80"
    if "very high" in c: return "#F87171"
    if "low" in c:       return "#86EFAC"
    if "medium" in c:    return "#FCD34D"
    if "high" in c:      return "#FB923C"
    return "#94A3B8"

def gauge_color(proba: float) -> str:
    if proba < 0.1:  return "linear-gradient(90deg,#4ADE80,#22C55E)"
    if proba < 0.2:  return "linear-gradient(90deg,#86EFAC,#4ADE80)"
    if proba < 0.4:  return "linear-gradient(90deg,#FCD34D,#F59E0B)"
    if proba < 0.6:  return "linear-gradient(90deg,#FB923C,#F97316)"
    return "linear-gradient(90deg,#F87171,#EF4444)"

# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
EMPLOYMENT_MAPPING = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
EDUCATION_MAPPING  = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
LOAN_PURPOSE_MAP   = {'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4}

def preprocess_inference_data(input_data) -> pd.DataFrame:
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()

    df['employment_status'] = df.get('employment_status', pd.Series(dtype=str)).map(EMPLOYMENT_MAPPING).fillna(0).astype(int)
    df['education_level']   = df.get('education_level',   pd.Series(dtype=str)).map(EDUCATION_MAPPING).fillna(0).astype(int)
    df['loan_purpose']      = df.get('loan_purpose',      pd.Series(dtype=str)).map(LOAN_PURPOSE_MAP).fillna(0).astype(int)
    df['collateral_present']= df.get('collateral_present',pd.Series(dtype=str)).map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    for col in ['annual_income', 'loan_amount', 'num_previous_loans', 'credit_history_length', 'num_defaults']:
        if col not in df.columns:
            df[col] = 0

    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization']   = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate']         = df['num_defaults'] / (df['num_previous_loans'] + 1)

    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median() if not df[c].isnull().all() else 0)

    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(
            "Feature columns file not found. Run train.py to generate model files."
        )
    feature_columns: List[str] = joblib.load(FEATURE_COLUMNS_FILE)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[Optional[object], Optional[List[str]]]:
    try:
        model    = joblib.load(CALIBRATED_MODEL_FILE)
        feat_col = joblib.load(FEATURE_COLUMNS_FILE)
        return model, feat_col
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def load_base_model():
    try:
        return joblib.load(BASE_MODEL_FILE)
    except Exception:
        return None

def models_available() -> bool:
    return all(os.path.exists(f) for f in [CALIBRATED_MODEL_FILE, FEATURE_COLUMNS_FILE])

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    model, _ = load_models()
    if model is None:
        raise RuntimeError("Model not loaded.")
    processed = preprocess_inference_data(input_dict)
    proba = float(model.predict_proba(processed)[:, 1][0])
    score = int(round((1 - proba) * 1000))
    if proba < 0.10:   category = "Very Low Risk"
    elif proba < 0.20: category = "Low Risk"
    elif proba < 0.40: category = "Medium Risk"
    elif proba < 0.60: category = "High Risk"
    else:              category = "Very High Risk"
    return proba, score, category, processed

# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ═══════════════════════════════════════════════════════════════════════════════
def get_shap_values(model, input_df: pd.DataFrame, bg_df: Optional[pd.DataFrame] = None):
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].base_estimator
    else:
        base = model
    try:
        explainer = shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")
        vals = explainer.shap_values(input_df)
        if isinstance(vals, list): vals = vals[1]
        return explainer, vals
    except Exception:
        if bg_df is not None:
            explainer = shap.KernelExplainer(base.predict_proba, bg_df.sample(min(50, len(bg_df))))
            vals = explainer.shap_values(input_df)
            if isinstance(vals, list): vals = vals[1]
            return explainer, vals
        raise

def plot_shap_bar(shap_values, features: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#0D1520")
    
    sv = shap_values[0]
    cols = features.columns.tolist()
    pairs = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [p[0].replace("_", " ").title() for p in pairs]
    vals  = [p[1] for p in pairs]
    colors = ["#F87171" if v > 0 else "#4ADE80" for v in vals]
    
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("SHAP Value  (→ increases default risk)", color="#64748B", fontsize=9)
    ax.tick_params(colors="#94A3B8", labelsize=8.5)
    ax.spines[:].set_visible(False)
    for spine in ax.spines.values(): spine.set_edgecolor("#1E2D42")
    ax.grid(axis="x", color="#1E2D42", linewidth=0.5)
    
    red_p = mpatches.Patch(color="#F87171", label="↑ Increases risk")
    grn_p = mpatches.Patch(color="#4ADE80", label="↓ Reduces risk")
    ax.legend(handles=[red_p, grn_p], loc="lower right",
              facecolor="#0D1520", edgecolor="#1E2D42", labelcolor="#94A3B8", fontsize=8)
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feat_cols: List[str]) -> Optional[plt.Figure]:
    base = load_base_model()
    if base is None or not hasattr(base, "feature_importances_"):
        return None
    fi = base.feature_importances_
    df = pd.DataFrame({"Feature": feat_cols, "Importance": fi}) \
           .sort_values("Importance", ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#080C14")
    ax.set_facecolor("#0D1520")
    bars = ax.barh(df["Feature"][::-1], df["Importance"][::-1], color="#38BDF8", height=0.6, alpha=0.85)
    ax.tick_params(colors="#94A3B8", labelsize=8.5)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#1E2D42", linewidth=0.5)
    ax.set_xlabel("Feature Importance", color="#64748B", fontsize=9)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  BLOCKCHAIN / LEDGER
# ═══════════════════════════════════════════════════════════════════════════════
class BlockchainManager:
    def __init__(self):
        self.provider_url     = os.getenv("WEB3_PROVIDER_URL", "http://127.0.0.1:8545")
        self.account_address  = os.getenv("ACCOUNT_ADDRESS", "")
        self.private_key      = os.getenv("PRIVATE_KEY", "")
        self.contract_address = os.getenv("CONTRACT_ADDRESS", "")
        self.abi_path         = os.path.join(CONTRACTS_DIR, "VerificationContract.json")
        self.w3               = None
        self.contract         = None

        if WEB3_AVAILABLE:
            try:
                self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
                if self.contract_address and os.path.exists(self.abi_path):
                    with open(self.abi_path) as f:
                        data = json.load(f)
                    self.contract = self.w3.eth.contract(
                        address=self.contract_address, abi=data.get("abi", data))
            except Exception:
                self.w3 = None

    def is_chain_connected(self) -> bool:
        if self.w3:
            try: return self.w3.is_connected()
            except: return False
        return False

    def record_verification(self, applicant_id, data_hash, risk_score, risk_category, pod) -> str:
        # Try on-chain
        if self.w3 and self.contract and self.account_address and self.private_key:
            try:
                prob_int = int(max(0.0, min(1.0, pod)) * 10000)
                nonce    = self.w3.eth.get_transaction_count(self.account_address)
                for fn_name, args in [
                    ("storeVerificationResult", (applicant_id, data_hash, int(risk_score), risk_category, prob_int)),
                    ("storeVerification",        (applicant_id, int(risk_score), risk_category, data_hash)),
                ]:
                    try:
                        fn  = getattr(self.contract.functions, fn_name)
                        txn = fn(*args).build_transaction({
                            "from": self.account_address, "nonce": nonce,
                            "gas": 300000, "gasPrice": self.w3.eth.gas_price
                        })
                        signed  = self.w3.eth.account.sign_transaction(txn, self.private_key)
                        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
                        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                        return receipt.transactionHash.hex()
                    except Exception:
                        continue
            except Exception:
                pass

        # JSON ledger fallback
        entry = {
            "applicant_id": applicant_id, "data_hash": data_hash,
            "risk_score": int(risk_score), "risk_category": risk_category,
            "probability_of_default": float(pod),
            "timestamp": datetime.utcnow().isoformat()
        }
        ledger = []
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: ledger = json.load(f)
            except Exception: pass
        ledger.append(entry)
        try:
            with open(LEDGER_PATH, "w") as f: json.dump(ledger, f, indent=2)
            return "LOCAL_LEDGER_OK"
        except Exception as e:
            return f"ERROR: {e}"

    def get_verification(self, applicant_id: str) -> Dict:
        if self.w3 and self.contract:
            for fn_name in ["getVerificationResult", "getVerification"]:
                try:
                    res = getattr(self.contract.functions, fn_name)(applicant_id).call()
                    return {"data_hash": res[0], "risk_score": res[1], "risk_category": res[2]}
                except Exception:
                    continue
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: ledger = json.load(f)
                for e in reversed(ledger):
                    if e.get("applicant_id") == applicant_id: return e
            except Exception: pass
        return {"error": "Not found"}

    def ledger_count(self) -> int:
        if os.path.exists(LEDGER_PATH):
            try:
                with open(LEDGER_PATH) as f: return len(json.load(f))
            except Exception: pass
        return 0

@st.cache_resource(show_spinner=False)
def get_blockchain_manager():
    return BlockchainManager()

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'bm' not in st.session_state:
    st.session_state.bm = get_blockchain_manager()

bm = st.session_state.bm

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Credit<span>AI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v2.0 · AI Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    menu = st.radio(
        "", ["🔍 New Verification", "History", "Insights", "⛓ Ledger", "Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)

    model_ok = models_available()
    chain_ok = bm.is_chain_connected()

    col_a, col_b = st.columns(2)
    with col_a:
        if model_ok:
            st.success("Model ✓")
        else:
            st.error("Model ✗")
    with col_b:
        if chain_ok:
            st.success("Chain ✓")
        else:
            st.info("Ledger ✓")

    if model_ok:
        _, feat_cols = load_models()
        st.markdown(f'<span class="info-chip">{len(feat_cols) if feat_cols else "?"} features</span>', unsafe_allow_html=True)
    if SHAP_AVAILABLE:
        st.markdown('<span class="info-chip">SHAP ready</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#334155;font-family:\'DM Mono\',monospace;">'
        f'UTC · {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-bar">
  <div class="top-bar-logo">Credit<span>AI</span> · Risk Verification System</div>
  <div class="top-bar-badge">PRODUCTION · SECURE</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: NEW VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
if menu == "🔍 New Verification":
    st.markdown('<div class="section-header">New Credit Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter applicant details below to generate an AI-powered risk assessment.</div>', unsafe_allow_html=True)

    if not model_ok:
        st.error("Model files not found. Run `train.py` first to generate model files in the `models/` directory.")
        st.code("python train.py", language="bash")
        st.stop()

    with st.form("verification_form", clear_on_submit=False):
        # ── Row 1: Identity ──────────────────────────────────────────────────
        st.markdown("#### Applicant Identity")
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_id    = st.text_input("Applicant ID *", placeholder="e.g. APP-2024-001")
            applicant_name  = st.text_input("Full Name *",    placeholder="Jane Smith")
        with c2:
            applicant_email = st.text_input("Email *",        placeholder="jane@example.com")
            age             = st.slider("Age", 18, 90, 32)
        with c3:
            employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed", "student"])
            education_level   = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])

        st.markdown("---")
        # ── Row 2: Financials ────────────────────────────────────────────────
        st.markdown("#### Financial Profile")
        c4, c5, c6 = st.columns(3)
        with c4:
            annual_income          = st.number_input("Annual Income (USD) *", min_value=0, value=60000, step=5000)
            current_credit_score   = st.slider("Credit Score", 300, 850, 680)
        with c5:
            credit_history_length  = st.slider("Credit History (years)", 0, 30, 6)
            num_previous_loans     = st.slider("Previous Loans", 0, 20, 2)
        with c6:
            num_defaults           = st.slider("Number of Defaults", 0, 10, 0)
            avg_payment_delay_days = st.slider("Avg Payment Delay (days)", 0, 90, 5)

        st.markdown("---")
        # ── Row 3: Loan ──────────────────────────────────────────────────────
        st.markdown("#### Loan Details")
        c7, c8, c9 = st.columns(3)
        with c7:
            loan_amount    = st.number_input("Loan Amount (USD) *", min_value=1000, value=30000, step=1000)
            loan_term_months = st.slider("Loan Term (months)", 6, 84, 36)
        with c8:
            loan_purpose   = st.selectbox("Loan Purpose", ["Business", "Car Loan", "Education", "Home Loan", "Crypto-Backed"])
            collateral_present = st.radio("Collateral Present", ["Yes", "No"], horizontal=True)
        with c9:
            identity_verified_on_chain    = st.radio("On-Chain Identity Verified", ["Yes", "No"], horizontal=True)
            fraud_alert_flag              = st.radio("Fraud Alert", ["No", "Yes"], horizontal=True)

        st.markdown("---")
        # ── Row 4: On-chain data ─────────────────────────────────────────────
        st.markdown("#### ⛓ Blockchain Signals")
        c10, c11 = st.columns(2)
        with c10:
            transaction_consistency_score = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.82)
        with c11:
            on_chain_credit_history = st.slider("On-Chain Credit History (years)", 0, 10, 3)

        st.markdown("")
        submitted = st.form_submit_button("⚡ Run AI Verification", type="primary", use_container_width=True)

    # ── PROCESS SUBMISSION ───────────────────────────────────────────────────
    if submitted:
        errors = []
        if not applicant_id.strip():   errors.append("Applicant ID is required.")
        if not applicant_name.strip(): errors.append("Full Name is required.")
        if not applicant_email.strip():errors.append("Email is required.")

        if errors:
            for e in errors: st.error(e)
        else:
            with st.spinner("Running AI risk analysis…"):
                application = {
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
                data_hash = generate_data_hash(application)

                # Save initial DB record
                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        INSERT OR REPLACE INTO verification_results
                        (applicant_id,applicant_name,applicant_email,age,data_hash,
                         risk_score,probability_of_default,risk_category,timestamp,tx_hash)
                        VALUES (?,?,?,?,?,NULL,NULL,NULL,?,NULL)
                    """, (application['applicant_id'], application['applicant_name'],
                          application['applicant_email'], application['age'],
                          data_hash, application['submission_timestamp']))
                    conn.commit()
                finally:
                    conn.close()

                # Predict
                try:
                    proba, score, category, processed = predict_single(application)
                except FileNotFoundError as e:
                    st.error(str(e)); st.stop()
                except Exception as e:
                    st.error(f"Prediction error: {e}"); st.stop()

                # Update DB
                try:
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("""
                        UPDATE verification_results
                        SET risk_score=?, probability_of_default=?, risk_category=?
                        WHERE applicant_id=?
                    """, (score, proba, category, application['applicant_id']))
                    conn.commit()
                finally:
                    conn.close()

                # Session state
                st.session_state.last_result = {
                    "data": application, "hash": data_hash,
                    "proba": proba, "score": score,
                    "category": category, "processed": processed,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # ── RESULTS UI ───────────────────────────────────────────────────
            r = st.session_state.last_result
            pct = r['proba'] * 100
            badge_class = risk_color_class(r['category'])

            st.markdown(f"""
            <div class="result-panel">
              <div class="result-title">Verification Complete · {r['timestamp'][:19]} UTC</div>
              <div style="display:flex;align-items:flex-end;gap:1.5rem;flex-wrap:wrap;margin-top:0.5rem;">
                <div>
                  <div style="font-size:0.75rem;color:#64748B;margin-bottom:0.2rem;">Risk Score</div>
                  <div class="result-score">{r['score']}<span style="font-size:1.5rem;color:#334155">/1000</span></div>
                </div>
                <div>
                  <div style="font-size:0.75rem;color:#64748B;margin-bottom:0.6rem;">Default Probability</div>
                  <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:700;color:#F1F5F9;">{pct:.1f}%</div>
                </div>
                <div>
                  <div style="font-size:0.75rem;color:#64748B;margin-bottom:0.6rem;">Category</div>
                  <span class="risk-badge {badge_class}">{r['category']}</span>
                </div>
              </div>

              <div class="gauge-wrap">
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#64748B;margin-bottom:4px;">
                  <span>Low Risk</span><span>High Risk</span>
                </div>
                <div class="gauge-track">
                  <div class="gauge-fill" style="width:{min(pct,100):.1f}%;background:{gauge_color(r['proba'])};"></div>
                </div>
              </div>

              <div class="result-hash">
                <span style="color:#38BDF8;">SHA-256 DATA HASH</span><br>
                {r['hash']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # KPIs
            st.markdown("")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Applicant", r['data']['applicant_name'])
            k2.metric("Annual Income", f"${r['data']['annual_income']:,.0f}")
            k3.metric("Loan Amount",   f"${r['data']['loan_amount']:,.0f}")
            k4.metric("Credit Score",  str(r['data']['current_credit_score']))

            # ── Store to ledger button ────────────────────────────────────────
            st.markdown("")
            col_btn1, col_btn2, _ = st.columns([1, 1, 3])
            with col_btn1:
                store_btn = st.button("Store to Ledger", key="store_btn")
            with col_btn2:
                shap_btn = st.button("Explain (SHAP)", key="shap_btn") if SHAP_AVAILABLE else None

            if store_btn:
                with st.spinner("Recording verification…"):
                    tx = bm.record_verification(
                        r['data']['applicant_id'], r['hash'],
                        r['score'], r['category'], r['proba']
                    )
                if tx.startswith("0x") or tx == "LOCAL_LEDGER_OK":
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        conn.execute("UPDATE verification_results SET tx_hash=? WHERE applicant_id=?",
                                     (tx, r['data']['applicant_id']))
                        conn.commit()
                    finally:
                        conn.close()
                    st.session_state.last_result['tx'] = tx
                    if tx == "LOCAL_LEDGER_OK":
                        st.success("Stored to local JSON ledger.")
                    else:
                        st.success(f"On-chain TX: `{tx}`")
                else:
                    st.error(f"Storage failed: {tx}")

            if SHAP_AVAILABLE and shap_btn:
                with st.spinner("Computing SHAP explanations…"):
                    try:
                        model, _ = load_models()
                        bg = None
                        csv_p = os.path.join(DATA_DIR, "sample_dataset.csv")
                        if os.path.exists(csv_p):
                            try:
                                bg_raw = pd.read_csv(csv_p)
                                bg = preprocess_inference_data(bg_raw) if len(bg_raw) else None
                            except Exception:
                                bg = None
                        _, sv = get_shap_values(model, r['processed'], bg)
                        fig = plot_shap_bar(sv, r['processed'])
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP unavailable: {e}")

            # Show stored TX if already done
            if st.session_state.last_result and st.session_state.last_result.get('tx'):
                tx_val = st.session_state.last_result['tx']
                if tx_val and tx_val != "LOCAL_LEDGER_OK":
                    st.markdown(f'<div class="result-hash"><span style="color:#38BDF8;">TX HASH</span><br>{tx_val}</div>',
                                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "History":
    st.markdown('<div class="section-header">Verification History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All past credit risk assessments.</div>', unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM verification_results ORDER BY timestamp DESC", conn)
    finally:
        conn.close()

    if df.empty:
        st.markdown('<div class="card">No verifications yet. Run a new verification to get started.</div>', unsafe_allow_html=True)
    else:
        df['probability_of_default'] = pd.to_numeric(df['probability_of_default'], errors='coerce')
        df['risk_score']             = pd.to_numeric(df['risk_score'], errors='coerce')

        # KPI row
        total        = len(df)
        avg_score    = df['risk_score'].mean()
        high_risk_ct = df['risk_category'].str.contains('High', na=False).sum()
        stored_ct    = df['tx_hash'].notnull().sum()

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-tile">
            <div class="kpi-label">Total Verifications</div>
            <div class="kpi-value">{total}</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Avg Risk Score</div>
            <div class="kpi-value">{avg_score:.0f}</div>
            <div class="kpi-sub">out of 1000</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">High Risk Applicants</div>
            <div class="kpi-value">{high_risk_ct}</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Ledger Records</div>
            <div class="kpi-value">{stored_ct}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Cards
        st.markdown("#### Recent Applications")
        for _, row in df.head(15).iterrows():
            cat      = row.get('risk_category') or "Unknown"
            score    = int(row['risk_score']) if pd.notnull(row.get('risk_score')) else "—"
            proba    = row.get('probability_of_default')
            prob_str = f"{float(proba):.1%}" if pd.notnull(proba) else "—"
            dot_col  = risk_dot_color(cat)
            badge    = risk_color_class(cat)
            name     = row.get('applicant_name') or "Unknown"
            aid      = row.get('applicant_id') or "—"
            ts       = str(row.get('timestamp') or "")[:16]
            tx       = row.get('tx_hash') or ""
            chain_tag= '<span class="info-chip">⛓ Stored</span>' if tx else ''

            st.markdown(f"""
            <div class="hist-card">
              <div class="hist-dot" style="background:{dot_col};"></div>
              <div>
                <div class="hist-name">{name}</div>
                <div class="hist-id">{aid}</div>
                {chain_tag}
              </div>
              <div class="hist-meta">
                <div class="hist-score">{score}<span style="font-size:0.75rem;color:#64748B;">/1000</span></div>
                <span class="risk-badge {badge}" style="font-size:0.65rem;">{cat}</span><br>
                <div class="hist-ts">{ts} · {prob_str} PoD</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Full table + export
        st.markdown("#### Full Record Table")
        display = df.copy()
        display['probability_of_default'] = display['probability_of_default'].apply(
            lambda x: f"{float(x):.2%}" if pd.notnull(x) else "—"
        )
        st.dataframe(
            display[['applicant_id','applicant_name','risk_score','risk_category',
                      'probability_of_default','timestamp','tx_hash']],
            use_container_width=True, hide_index=True
        )
        csv_bytes = display.to_csv(index=False).encode()
        st.download_button("⬇ Export CSV", csv_bytes, "verification_history.csv",
                           "text/csv", key="csv-export")

        # Ledger lookup
        stored_ids = df[df['tx_hash'].notnull()]['applicant_id'].tolist()
        if stored_ids:
            st.markdown("#### Verify Ledger Record")
            sel_id = st.selectbox("Select Applicant ID", stored_ids, key="lookup_id")
            if st.button("Fetch Record", key="fetch_ledger"):
                rec = bm.get_verification(sel_id)
                if 'error' in rec:
                    st.error("Record not found in ledger.")
                else:
                    st.success("Record retrieved.")
                    st.json(rec)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "Insights":
    st.markdown('<div class="section-header">Data Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Analytics from the training dataset and verification history.</div>', unsafe_allow_html=True)

    plt.style.use("dark_background")

    # ── From DB ──────────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    try:
        db_df = pd.read_sql_query("SELECT * FROM verification_results", conn)
    finally:
        conn.close()

    if not db_df.empty and db_df['risk_score'].notnull().any():
        db_df['risk_score'] = pd.to_numeric(db_df['risk_score'], errors='coerce')
        db_df['probability_of_default'] = pd.to_numeric(db_df['probability_of_default'], errors='coerce')

        st.markdown("#### Your Verification Portfolio")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#080C14")
        for ax in axes: ax.set_facecolor("#0D1520")

        # Risk category distribution
        cats = db_df['risk_category'].value_counts()
        colors_map = {
            "Very Low Risk": "#4ADE80", "Low Risk": "#86EFAC",
            "Medium Risk": "#FCD34D",   "High Risk": "#FB923C",
            "Very High Risk": "#F87171"
        }
        bar_colors = [colors_map.get(c, "#64748B") for c in cats.index]
        axes[0].bar(cats.index, cats.values, color=bar_colors, edgecolor="#1E2D42")
        axes[0].tick_params(colors="#94A3B8", labelsize=7.5)
        axes[0].set_title("Risk Category Distribution", color="#94A3B8", fontsize=10)
        axes[0].spines[:].set_visible(False)
        axes[0].grid(axis="y", color="#1E2D42", linewidth=0.5)
        for label in axes[0].get_xticklabels(): label.set_rotation(20)

        # Score histogram
        axes[1].hist(db_df['risk_score'].dropna(), bins=15, color="#38BDF8", edgecolor="#1E2D42", alpha=0.85)
        axes[1].tick_params(colors="#94A3B8", labelsize=8)
        axes[1].set_xlabel("Risk Score (0–1000)", color="#64748B", fontsize=9)
        axes[1].set_title("Risk Score Distribution", color="#94A3B8", fontsize=10)
        axes[1].spines[:].set_visible(False)
        axes[1].grid(axis="y", color="#1E2D42", linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── From sample dataset ───────────────────────────────────────────────────
    csv_p = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(csv_p):
        try:
            sample = pd.read_csv(csv_p)
            st.markdown("#### Training Dataset Overview")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", f"{len(sample):,}")
            if 'default_flag' in sample.columns:
                m2.metric("Default Rate", f"{sample['default_flag'].mean():.2%}")
            if 'current_credit_score' in sample.columns:
                m3.metric("Avg Credit Score", f"{sample['current_credit_score'].mean():.0f}")

            if 'current_credit_score' in sample.columns or 'loan_amount' in sample.columns:
                ncols = sum([1 for c in ['current_credit_score','loan_amount','annual_income'] if c in sample.columns])
                fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
                if ncols == 1: axes = [axes]
                fig.patch.set_facecolor("#080C14")
                plot_fields = [(c, l, col) for c, l, col in [
                    ('current_credit_score', 'Credit Score', '#38BDF8'),
                    ('loan_amount',          'Loan Amount ($)', '#818CF8'),
                    ('annual_income',        'Annual Income ($)', '#34D399'),
                ] if c in sample.columns]
                for ax, (col, label, clr) in zip(axes, plot_fields):
                    ax.set_facecolor("#0D1520")
                    ax.hist(sample[col].dropna(), bins=30, color=clr, edgecolor="#1E2D42", alpha=0.85)
                    ax.set_xlabel(label, color="#64748B", fontsize=9)
                    ax.tick_params(colors="#94A3B8", labelsize=8)
                    ax.spines[:].set_visible(False)
                    ax.grid(axis="y", color="#1E2D42", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with st.expander("Preview Data (first 20 rows)"):
                st.dataframe(sample.head(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load sample dataset: {e}")
    else:
        st.info("No sample dataset found at `data/sample_dataset.csv`. Run `train.py` to generate it.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: LEDGER
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "⛓ Ledger":
    st.markdown('<div class="section-header">Immutable Ledger</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Blockchain integration status and verification records.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("**Provider**")
        st.code(bm.provider_url, language=None)
        if bm.is_chain_connected():
            st.success("Blockchain Connected")
            if WEB3_AVAILABLE and bm.w3:
                try:
                    st.write("**Chain ID:**", bm.w3.eth.chain_id)
                    st.write("**Latest Block:**", bm.w3.eth.block_number)
                except Exception: pass
        else:
            st.info("Using Local JSON Ledger (fallback)")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Account**")
        if bm.account_address:
            st.code(bm.account_address, language=None)
            if WEB3_AVAILABLE and bm.w3 and bm.is_chain_connected():
                try:
                    bal = bm.w3.from_wei(bm.w3.eth.get_balance(bm.account_address), "ether")
                    st.metric("Balance", f"{bal:.4f} ETH")
                except Exception: pass
        else:
            st.write("No account configured. Set `ACCOUNT_ADDRESS` in `.env`.")

        if bm.contract_address:
            st.write("**Contract:**")
            st.code(bm.contract_address, language=None)
        else:
            st.write("No contract deployed. Set `CONTRACT_ADDRESS` in `.env`.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Ledger contents
    st.markdown("#### Local Ledger Contents")
    ledger_count = bm.ledger_count()
    st.metric("Total Ledger Records", ledger_count)

    if os.path.exists(LEDGER_PATH) and ledger_count > 0:
        try:
            with open(LEDGER_PATH) as f:
                ledger_data = json.load(f)
            ledger_df = pd.DataFrame(ledger_data)
            st.dataframe(ledger_df, use_container_width=True, hide_index=True)

            csv_bytes = ledger_df.to_csv(index=False).encode()
            st.download_button("⬇ Export Ledger CSV", csv_bytes, "ledger.csv", "text/csv")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Refresh Ledger View", key="refresh_ledger"):
                    st.rerun()
        except Exception as e:
            st.error(f"Could not read ledger: {e}")
    else:
        st.info("No ledger records yet. Store a verification to populate.")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "Model Info":
    st.markdown('<div class="section-header">Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Details about the AI model powering risk predictions.</div>', unsafe_allow_html=True)

    model, feat_cols = load_models()

    if model is None:
        st.error("No model found. Run `train.py` to generate model files.")
        st.markdown("""
        **Required files:**
        - `models/calibration_model.pkl`
        - `models/trained_lgbm_model.pkl`
        - `models/feature_columns.pkl`
        """)
        st.code("python train.py", language="bash")
    else:
        # Status row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
            st.markdown("**Model Type**")
            st.markdown(f"`{type(model).__name__}`")
            st.markdown("**Features**")
            st.markdown(f"`{len(feat_cols)}`")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**File Status**")
            for name, path in [
                ("Calibrated Model", CALIBRATED_MODEL_FILE),
                ("Base Model",       BASE_MODEL_FILE),
                ("Feature Columns",  FEATURE_COLUMNS_FILE),
            ]:
                icon = "" if os.path.exists(path) else ""
                sz   = f" ({os.path.getsize(path)/1024:.1f} KB)" if os.path.exists(path) else ""
                st.write(f"{icon} **{name}**{sz}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Risk Thresholds**")
            thresholds = [
                ("Very Low", "< 10%",   "#4ADE80"),
                ("Low",      "10–20%",  "#86EFAC"),
                ("Medium",   "20–40%",  "#FCD34D"),
                ("High",     "40–60%",  "#FB923C"),
                ("Very High","> 60%",   "#F87171"),
            ]
            for label, rng, col in thresholds:
                st.markdown(f'<span style="color:{col};">●</span> **{label}**: {rng}', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance chart
        fig_fi = plot_feature_importance(model, feat_cols)
        if fig_fi:
            st.markdown("#### Feature Importance (Top 15)")
            st.pyplot(fig_fi)
            plt.close()

        # Feature list
        with st.expander(f"All {len(feat_cols)} Features"):
            cols = st.columns(3)
            for i, f in enumerate(feat_cols):
                cols[i % 3].markdown(f'<span class="info-chip">{f}</span>', unsafe_allow_html=True)

        # Score formula
        st.markdown("#### Scoring Formula")
        st.markdown("""
        ```
        Risk Score  = round((1 - Probability_of_Default) × 1000)
        Score range : 0 (maximum risk)  →  1000 (minimum risk)
        ```
        The model outputs a calibrated probability via `CalibratedClassifierCV`,
        trained on a LightGBM base estimator with engineered features including
        income-to-loan ratio, credit utilization, and historical default rate.
        """)
