# train.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import optuna
from utils.preprocessing import preprocess_data
from utils import MODELS_DIR, DATA_DIR

# --- Config tuned for CPU/8GB
N_TRIALS = 20
OPTUNA_N_JOBS = 1
RANDOM_STATE = 42
CV_FOLDS = 3

os.makedirs(MODELS_DIR, exist_ok=True)

# --- Create sample dataset if missing
sample_path = os.path.join(DATA_DIR, "sample_dataset.csv")
if not os.path.exists(sample_path):
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    data = {
        'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(25, 70, n_samples),
        'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed', 'student'], n_samples, p=[0.6,0.2,0.1,0.1]),
        'annual_income': np.random.randint(20000,100000,n_samples),
        'education_level': np.random.choice(['High School','Diploma','Bachelor','Master'], n_samples, p=[0.2,0.3,0.4,0.1]),
        'credit_history_length': np.random.randint(1,20,n_samples),
        'num_previous_loans': np.random.randint(0,10,n_samples),
        'num_defaults': np.random.randint(0,3,n_samples),
        'avg_payment_delay_days': np.random.randint(0,15,n_samples),
        'current_credit_score': np.random.randint(500,800,n_samples),
        'loan_amount': np.random.randint(5000,150000,n_samples),
        'loan_term_months': np.random.choice([12,24,36,48,60], n_samples),
        'loan_purpose': np.random.choice(['Business','Crypto-Backed','Car Loan','Education','Home Loan'], n_samples),
        'collateral_present': np.random.choice(['Yes','No'], n_samples, p=[0.7,0.3]),
        'identity_verified_on_chain': np.random.randint(0,2,n_samples),
        'transaction_consistency_score': np.round(np.random.uniform(0.2,1.0,n_samples),2),
        'fraud_alert_flag': np.random.randint(0,2,n_samples, p=[0.9,0.1]),
        'on_chain_credit_history': np.random.randint(0,10,n_samples)
    }
    df = pd.DataFrame(data)
    prob_default = (
        0.3 * (df['num_defaults'] > 0).astype(int) +
        0.2 * (df['employment_status'] == 'unemployed').astype(int) +
        0.1 * (df['current_credit_score'] < 600).astype(int) +
        0.1 * (df['avg_payment_delay_days'] > 7).astype(int) +
        0.1 * (df['loan_amount'] / df['annual_income'] > 0.5).astype(int) +
        0.1 * (df['fraud_alert_flag'] == 1).astype(int) +
        np.random.normal(0, 0.1, len(df))
    )
    df['default_flag'] = (prob_default > 0.5).astype(int)
    df['probability_of_default'] = np.clip(1 / (1 + np.exp(-prob_default)), 0.01, 0.99)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(sample_path, index=False)
    print("Sample dataset created at", sample_path)

# --- Load and preprocess
df = pd.read_csv(sample_path)
df = preprocess_data(df)

features = [
 'age','employment_status','annual_income','education_level',
 'credit_history_length','num_previous_loans','num_defaults',
 'avg_payment_delay_days','current_credit_score','loan_amount',
 'loan_term_months','loan_purpose','collateral_present',
 'identity_verified_on_chain','transaction_consistency_score',
 'fraud_alert_flag','on_chain_credit_history','income_to_loan_ratio',
 'credit_utilization','default_rate'
]

X = df[features]
y = df['default_flag']

joblib.dump(features, os.path.join(MODELS_DIR,'feature_columns.pkl'))

# --- Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=RANDOM_STATE,stratify=y)

# --- Optuna objective
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_jobs': 2
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        sm = SMOTE(random_state=RANDOM_STATE)
        Xt_res, yt_res = sm.fit_resample(Xt, yt)
        model = lgb.LGBMClassifier(**params)
        model.fit(Xt_res, yt_res, eval_set=[(Xv,yv)], eval_metric='auc', early_stopping_rounds=50, verbose=False)
        preds = model.predict_proba(Xv)[:,1]
        scores.append(roc_auc_score(yv,preds))
    return float(np.mean(scores))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, n_jobs=OPTUNA_N_JOBS)

print("Best trial params:", study.best_trial.params)
best = study.best_trial.params
best.update({'objective':'binary','metric':'binary_logloss','verbosity':-1,'boosting_type':'gbdt','n_jobs':2})

# --- Final training with SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)
final_model = lgb.LGBMClassifier(**best)
final_model.fit(X_tr_res, y_tr_res, eval_set=[(X_test,y_test)], eval_metric='auc', early_stopping_rounds=50, verbose=50)

joblib.dump(final_model, os.path.join(MODELS_DIR,'trained_lgbm_model.pkl'))

# --- Calibration
calibrated = CalibratedClassifierCV(final_model, method='isotonic', cv=3)
calibrated.fit(X_tr_res, y_tr_res)
joblib.dump(calibrated, os.path.join(MODELS_DIR,'calibration_model.pkl'))

# --- Evaluation
y_proba = calibrated.predict_proba(X_test)[:,1]
y_pred = (y_proba > 0.5).astype(int)
print("ROC AUC:", roc_auc_score(y_test,y_proba))
print("Accuracy:", accuracy_score(y_test,y_pred))
print("F1:", f1_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
