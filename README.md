# Credit Risk Verification (Streamlit + LightGBM + Blockchain)

Industry-ready project that:
- trains a LightGBM credit-risk model with Optuna,
- applies SMOTE, calibration, and SHAP explanations,
- stores/verifies results on-chain (Web3) or a local JSON ledger fallback,
- optimized for CPU (~8 GB RAM).

## Quickstart

1. Create virtualenv & install:
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
