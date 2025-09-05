import streamlit as st
import pandas as pd
import os
from datetime import datetime
from utils.hash_utils import generate_data_hash
from inference import predict_single, load_models as load_models_cached
from shap_analysis import explain_prediction_sampled, plot_shap_waterfall
from blockchain.blockchain_manager import BlockchainManager
from utils import DATA_DIR
import joblib

st.set_page_config(page_title="Credit Risk Verification", layout="wide")

@st.cache_resource
def get_blockchain_manager():
    return BlockchainManager()

@st.cache_resource
def get_models():
    return load_models_cached()

# session state
if 'verification_results' not in st.session_state:
    st.session_state.verification_results = {}
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = get_blockchain_manager()

def main():
    st.title("🔒 Advanced Credit Risk Verification")
    menu = st.sidebar.radio("Navigation", ["New Verification","Verification History","Data Insights","Blockchain Status","Model Info"])
    if menu == "New Verification":
        new_verification()
    elif menu == "Verification History":
        verification_history()
    elif menu == "Data Insights":
        data_insights()
    elif menu == "Blockchain Status":
        blockchain_status()
    else:
        model_info()

def new_verification():
    st.header("New Credit Risk Verification")
    with st.form("form"):
        applicant_id = st.text_input("Applicant ID*")
        applicant_name = st.text_input("Full name*")
        applicant_email = st.text_input("Email*")
        age = st.slider("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
        employment_status = st.selectbox("Employment Status", ["employed","self-employed","unemployed","student"])
        education_level = st.selectbox("Education Level", ["High School","Diploma","Bachelor","Master","PhD"])
        credit_history_length = st.slider("Credit history (years)", 0, 30, 5)
        num_previous_loans = st.slider("Previous loans", 0, 20, 2)
        num_defaults = st.slider("Defaults", 0, 10, 0)
        avg_payment_delay_days = st.slider("Avg delay days", 0, 60, 5)
        current_credit_score = st.slider("Credit score", 300, 850, 650)
        loan_amount = st.number_input("Loan amount", min_value=0, step=1000, value=25000)
        loan_term_months = st.slider("Loan term months", 12, 84, 36)
        loan_purpose = st.selectbox("Loan purpose", ["Business","Crypto-Backed","Car Loan","Education","Home Loan"])
        collateral_present = st.radio("Collateral present", ["Yes","No"])
        identity_verified_on_chain = st.radio("Identity verified on chain", [1,0], format_func=lambda x: "Yes" if x==1 else "No")
        fraud_alert_flag = st.radio("Fraud alert", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        transaction_consistency_score = st.slider("Transaction consistency", 0.0, 1.0, 0.8)
        on_chain_credit_history = st.slider("On-chain credit history", 0, 10, 5)
        submitted = st.form_submit_button("Verify")
    if submitted:
        if not all([applicant_id, applicant_name, applicant_email]):
            st.error("Please provide required fields")
            return
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
        data_hash = generate_data_hash(application)
        with st.spinner("Running model..."):
            try:
                proba, score, category, processed = predict_single(application)
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                return
        st.success("Done")
        st.metric("Risk Score", score)
        st.metric("Prob. of Default", f"{proba:.2%}")
        st.markdown(f"**Category:** {category}")
        st.write("Data Hash:", data_hash)
        st.session_state.verification_results[applicant_id] = {
            "data": application,
            "hash": data_hash,
            "probability_of_default": proba,
            "risk_score": score,
            "risk_category": category,
            "timestamp": datetime.utcnow().isoformat()
        }

        st.subheader("SHAP explanation (fast sample)")
        try:
            model, feat_cols = get_models()
            bg = None
            sample_path = os.path.join(DATA_DIR, "sample_dataset.csv")
            if os.path.exists(sample_path):
                bg = pd.read_csv(sample_path)
            explainer, shap_vals = explain_prediction_sampled(model, processed, background_df=bg, nsample=100)
            expected = explainer.expected_value if hasattr(explainer, 'expected_value') else None
            fig = plot_shap_waterfall(explainer, expected, shap_vals, processed, index=0)
            st.pyplot(fig)
        except Exception as e:
            st.warning("SHAP unavailable: " + str(e))

        st.subheader("Blockchain / Ledger")
        bm = st.session_state.blockchain_manager
        if st.button("Store on blockchain / ledger"):
            res = bm.store_verification_result(applicant_id, data_hash, score, category, proba)
            if (isinstance(res, str) and (res.startswith("0x") or res == "LOCAL_LEDGER_OK")):
                st.success("Stored: " + str(res))
                st.session_state.verification_results[applicant_id]['tx_hash'] = res
            else:
                st.error("Store failed: " + str(res))

def verification_history():
    st.header("Verification History")
    hist = st.session_state.verification_results
    if not hist:
        st.info("No verifications yet")
        return
    df = []
    for k,v in hist.items():
        df.append({
            "Applicant ID": k,
            "Name": v['data']['applicant_name'],
            "Risk Score": v['risk_score'],
            "Risk Category": v['risk_category'],
            "Prob Default": f"{v['probability_of_default']:.2%}",
            "Date": v['timestamp'][:10],
            "TX": "Yes" if "tx_hash" in v else "No"
        })
    st.dataframe(pd.DataFrame(df), use_container_width=True)

def data_insights():
    st.header("Data Insights")
    sample_path = os.path.join(DATA_DIR, "sample_dataset.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.metric("Total", len(df))
        st.metric("Default Rate", f"{df['default_flag'].mean():.2%}")
        st.plotly_chart(pd.DataFrame(df['current_credit_score']).hist(), use_container_width=True)
    else:
        st.info("sample dataset missing; run train.py to generate it")

def blockchain_status():
    st.header("Blockchain Status")
    bm = st.session_state.blockchain_manager
    st.write("Provider:", bm.provider_url)
    st.write("Connected:", bm.is_connected())
    if bm.is_connected() and getattr(bm, 'w3', None):
        try:
            st.write("Chain ID:", bm.w3.eth.chain_id)
            if bm.account_address:
                bal = bm.w3.eth.get_balance(bm.account_address)
                st.write("Account balance:", bm.w3.from_wei(bal,'ether'))
        except Exception:
            st.write("Unable to query chain details")

def model_info():
    st.header("Model Information")
    try:
        model, feat_cols = get_models()
        st.write("Model loaded.")
        st.write("Features:", feat_cols)
    except Exception:
        st.info("No trained model found. Run `python train.py` first.")

if __name__ == "__main__":
    main()
