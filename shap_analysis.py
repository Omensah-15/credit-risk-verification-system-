import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def explain_prediction_sampled(model, input_df: pd.DataFrame, background_df: pd.DataFrame = None, nsample: int = 100):
    """
    Returns (explainer, shap_values) computed using a sampled background to reduce memory.
    """
    # Get base estimator if model is a CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].base_estimator
    else:
        base = model

    if background_df is not None and len(background_df) > 0:
        background = background_df.sample(min(nsample, len(background_df)))
        explainer = shap.TreeExplainer(base, data=background)
    else:
        explainer = shap.TreeExplainer(base)

    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values

def plot_shap_waterfall(explainer, expected_value, shap_values, features: pd.DataFrame, index: int = 0):
    """
    Small wrapper that returns a Matplotlib fig for a single-sample SHAP waterfall/decision plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # decision_plot for single-sample explanation
        shap.decision_plot(expected_value, shap_values, features.iloc[index], show=False)
    except Exception:
        # fallback to summary plot if decision_plot fails
        shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    return fig
