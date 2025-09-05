import joblib
import numpy as np
import pandas as pd
from utils.preprocessing import preprocess_inference_data
from typing import Tuple
from functools import lru_cache

@lru_cache(maxsize=1)
def load_models():
    """
    Load cached models. Expects:
      - models/calibration_model.pkl
      - models/feature_columns.pkl
    """
    model = joblib.load("models/calibration_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns

def predict_single(input_dict: dict) -> Tuple[float, int, str, pd.DataFrame]:
    model, feature_columns = load_models()
    processed = preprocess_inference_data(input_dict)
    # Ensure all columns present
    for c in feature_columns:
        if c not in processed.columns:
            processed[c] = 0
    processed = processed[feature_columns]
    proba = float(model.predict_proba(processed)[:,1][0])
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
    return proba, risk_score, category, processed

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    model, feature_columns = load_models()
    processed = preprocess_inference_data(df)
    for c in feature_columns:
        if c not in processed.columns:
            processed[c] = 0
    processed = processed[feature_columns]
    probs = model.predict_proba(processed)[:,1]
    scores = np.round((1 - probs) * 1000).astype(int)
    def cat(p):
        if p < 0.1: return "Very Low Risk"
        if p < 0.2: return "Low Risk"
        if p < 0.4: return "Medium Risk"
        if p < 0.6: return "High Risk"
        return "Very High Risk"
    cats = [cat(p) for p in probs]
    ret = pd.DataFrame({
        "probability_of_default": probs,
        "risk_score": scores,
        "risk_category": cats
    }, index=df.index)
    return ret
