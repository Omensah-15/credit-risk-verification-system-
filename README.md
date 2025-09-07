# AI Credit Risk Verification System with Cryptographic Audit Trail

A machine learning-based credit risk assessment platform featuring robust data integrity protection through cryptographic hashing and local immutable storage, designed with blockchain-like principles for verification and audit capabilities.
Try App online:[CRVS](https://credit-risk-blockchain-he6gnprqlveepzos9ud5pl.streamlit.app/)

## Key Features:
- **ML-Powered Risk Prediction**: Trains a LightGBM credit-risk model with Optuna for hyperparameter optimization.
- **Cryptographic Verification**: SHA-256 hashing for data integrity and tamper-evident records
- **Immutable Local Storage**: JSON-based ledger with blockchain-inspired immutability patterns
- **SHAP Explainability**: Transparent model decision explanations for regulatory compliance(currently in development)
- **Streamlit Web Interface**: User-friendly dashboard for complete risk assessment workflow

## Tech Stack:
- Python, Scikit-Learn, Streamlit, SHAP, Cryptographic Hashing, Local JSON Database

## Architecture Highlights:
- **Data Integrity**: All records cryptographically hashed to prevent tampering
- **Audit Trail**: Complete verification history with timestamps and hashes
- **Blockchain-Ready**: Architecture designed for easy migration to actual blockchain
- **Fallback System**: Robust local storage that maintains verification capabilities

## Use Cases:
- Banking and financial services preliminary credit assessment
- Loan application processing and risk evaluation
- Regulatory compliance with explainable AI requirements
- Educational demonstration of fintech system architecture
