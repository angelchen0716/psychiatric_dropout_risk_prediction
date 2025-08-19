# app.py — Psychiatric Dropout Risk (Monotone + Calibration + Pre/Post + Ablation + Capacity + Fairness)
# 修正與老師回饋對應（每點一句）：
# 1) Chief problem 納入（住院主因）
# 2) Bipolar current episode (manic/depressive/mixed) 影響風險
# 3) LOS 分佈與上限改貼近 3–4 週，並與診斷連動
# 4) PD/PTSD/SUD 提高 self-harm 機率
# 5) Follow-ups 明確定義為「出院後 30 天內接觸次數」
# 6) Living status / Case manager / Financial strain / Prior dropout readmission 納入
# 7) 修正 np.exp() 對 pandas 物件錯誤：先轉 numpy
# 8) 其餘原功能完整保留

import os, re, math
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO

try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor (Monotone + Calibrated)")

# ==== Math helpers ====
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps)); return np.log(p / (1 - p))
def _logit_vec(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

# ==== Global knobs ====
CAL_LOGIT_SHIFT = float(os.getenv("RISK_CAL_SHIFT", "0.0"))
BORDER_BAND = 7
BLEND_W_DEFAULT = 0.30
SOFT_UPLIFT = {"floor": 0.60, "add": 0.10, "cap": 0.90}

OVERLAY_SCALE = 0.75
DELTA_CLIP   = 1.00
TEMP         = 1.20

# ====== Options & schema ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female","Other/Unknown"]

# 👇 新增：住院主因 / 躁鬱期別 / 居住 / 個管 / 經濟 / 既往因 dropout 再住院
CHIEF_PROBLEMS = [
    "Suicidal/Self-harm risk","Aggression/Impulsivity","Unable self-care",
    "Severe psychosis/diagnostic workup","Family cannot manage","Medication adverse effects","Other/Unknown"
]
BIPOLAR_EPISODES = ["Unknown/NA","Manic","Depressive","Mixed"]
LIVING_STATUS = ["Alone","With family/others"]
TEMPLATE_COLUMNS = [
    "age","length_of_stay","num_previous_admissions",
    "medication_compliance_score","family_support_score","post_discharge_followups",
    "financial_strain_score","prior_dropout_readmit_Yes","prior_dropout_readmit_No",
    "has_case_manager_Yes","has_case_manager_No",
    "living_alone_Yes","living_alone_No",
    "chief_problem_" + CHIEF_PROBLEMS[0],"chief_problem_" + CHIEF_PROBLEMS[1],
    "chief_problem_" + CHIEF_PROBLEMS[2],"chief_problem_" + CHIEF_PROBLEMS[3],
    "chief_problem_" + CHIEF_PROBLEMS[4],"chief_problem_" + CHIEF_PROBLEMS[5],
    "chief_problem_" + CHIEF_PROBLEMS[6],
    "bipolar_episode_Manic","bipolar_episode_Depressive","bipolar_episode_Mixed","bipolar_episode_Unknown/NA",
    "gender_Male","gender_Female","gender_Other/Unknown",
] + [f"diagnosis_{d}" for d in DIAG_LIST] + [
    "has_recent_self_harm_Yes","has_recent_self_harm_No",
    "self_harm_during_admission_Yes","self_harm_during_admission_No",
]

# ====== Null-safe defaults ======
DEFAULTS = {
    "age": 40.0,
    "length_of_stay": 21.0,  # ← 改為約 3 週
    "num_previous_admissions": 0.0,
    "medication_compliance_score": 5.0,
    "family_support_score": 5.0,
    "post_discharge_followups": 0.0,  # 30 days post-discharge
    "financial_strain_score": 5.0,
}
NUMERIC_KEYS = ["age","length_of_stay","num_previous_admissions","medication_compliance_score",
                "family_support_score","post_discharge_followups","financial_strain_score"]

def _num_or_default(x, key):
    try: v = float(x)
    except Exception: v = np.nan
    if pd.isna(v): v = DEFAULTS.get(key, 0.0)
    return v

def fill_defaults_single_row(X1: pd.DataFrame):
    i = X1.index[0]
    for k in NUMERIC_KEYS:
        if k in X1.columns:
            X1.at[i, k] = _num_or_default(X1.at[i, k], k)
    diag_cols = [c for c in X1.columns if c.startswith("diagnosis_")]
    if diag_cols and (X1.loc[i, diag_cols].sum() == 0):
        col = "diagnosis_Other/Unknown"
        if col in X1.columns: X1.at[i, col] = 1
    oh_cols = [c for c in X1.columns if "_" in c and c not in NUMERIC_KEYS]
    if oh_cols: X1.loc[i, oh_cols] = X1.loc[i, oh_cols].fillna(0)

def fill_defaults_batch(df_feat: pd.DataFrame):
    for k in NUMERIC_KEYS:
        if k in df_feat.columns:
            df_feat[k] = pd.to_numeric(df_feat[k], errors="coerce").fillna(DEFAULTS[k])
    oh_cols = [c for c in df_feat.columns if "_" in c and c not in NUMERIC_KEYS]
    if oh_cols: df_feat[oh_cols] = df_feat[oh_cols].fillna(0)
    diag_cols = [c for c in df_feat.columns if c.startswith("diagnosis_")]
    if diag_cols:
        none_diag = (df_feat[diag_cols].sum(axis=1) == 0)
        if none_diag.any() and "diagnosis_Other/Unknown" in df_feat.columns:
            df_feat.loc[none_diag, "diagnosis_Other/Unknown"] = 1

# ====== Overlay policy（log-odds）======
POLICY = {
    "per_prev_admission": 0.18,
    "per_point_low_support": 0.20,
    "per_followup": -0.18,
    "no_followup_extra": 0.30,
    "los_short": 0.40, "los_mid": 0.00, "los_mid_high": 0.15, "los_long": 0.30,
    "age_young": 0.10, "age_old": 0.10,
    "diag": {
        "Personality Disorder":    0.40,   # ↑ 些微
        "Substance Use Disorder":  0.40,
        "Bipolar":                 0.12,
        "PTSD":                    0.12,
        "Schizophrenia":           0.12,
        "Depression":              0.05,
        "Anxiety":                 0.00,
        "OCD":                     0.00,
        "Dementia":                0.00,
        "ADHD":                    0.00,
        "Other/Unknown":           0.00,
    },
    "x_sud_lowcomp": 0.30,
    "x_pd_shortlos": 0.12,
    "per_point_low_compliance": 0.22,
    "per_point_high_compliance_protect": -0.06,

    # 👇 新增：對應老師建議（各用一句）
    "chief": {
        "Suicidal/Self-harm risk": 0.45,
        "Aggression/Impulsivity":  0.25,
        "Unable self-care":        0.15,
        "Severe psychosis/diagnostic workup": 0.20,
        "Family cannot manage":    0.10,
        "Medication adverse effects": 0.05,
        "Other/Unknown":           0.00,
    },
    "bipolar_episode": {"Manic":0.15, "Depressive":0.10, "Mixed":0.20, "Unknown/NA":0.00},
    "living_alone": 0.20,
    "no_case_manager": 0.15,
    "per_point_financial_strain": 0.05,
    "prior_dropout_readmit": 0.35,
}

# ====== UI — Sidebar ======
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 95, 35)
    gender = st.selectbox("Gender", GENDER_LIST, index=0)
    diagnoses = st.multiselect("Diagnoses (multi-select)", DIAG_LIST, default=[])
    # LOS 更貼近臨床：3~60 天，預設 21
    length_of_stay = st.slider("Length of Stay (days)", 3, 120, 21)
    num_adm = st.slider("Previous Admissions (1y)", 0, 15, 1)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0)
    followups = st.slider("Post-discharge Followups (within 30 days)", 0, 10, 2)  # 明確定義 30 天
    financial = st.slider("Financial Strain (0–10)", 0.0, 10.0, 5.0)
    living = st.selectbox("Living status", LIVING_STATUS, index=0)
    has_cm = st.radio("Has case manager?", BIN_YESNO, index=1)
    prior_dr = st.radio("Prior dropout-related readmission?", BIN_YESNO, index=1)
    chief = st.selectbox("Chief problem of this admission", CHIEF_PROBLEMS, index=6)
    bipolar_ep = st.selectbox("Bipolar current episode (if applicable)", BIPOLAR_EPISODES, index=0)
    recent_self_harm = st.radio("Recent Self-harm", BIN_YESNO, index=1)
    selfharm_adm = st.radio("Self-harm During Admission", BIN_YESNO, index=1)

    mode = st.radio("Mode", ["Pre-planning (no followup as feature)", "Post-planning (monitoring)"], index=0)
    use_followups_feature = (mode.startswith("Post"))

    st.markdown("---")
    with st.expander("Advanced (calibration & overlay)", expanded=False):
        cal_shift = st.slider("Global calibration (log-odds shift)", -1.0, 1.0, CAL_LOGIT_SHIFT, 0.05)
        blend_w  = st.slider("Blend weight (Final = (1-BLEND)*Model + BLEND*Overlay)", 0.0, 1.0, BLEND_W_DEFAULT, 0.05)
        overlay_scale = st.slider("Overlay scale", 0.0, 1.0, OVERLAY_SCALE, 0.05)
        delta_clip = st.slider("Overlay delta clip |log-odds|", 0.0, 2.0, DELTA_CLIP, 0.05)
        temp_val = st.slider("Temperature (>1 softer probs)", 0.5, 3.0, TEMP, 0.05)
        st.caption("Follow-ups are defined as contacts **within 30 days post-discharge** (clinic/phone/social work).")

    CAL_LOGIT_SHIFT = cal_shift
    BLEND_W = blend_w
    OVERLAY_SCALE = overlay_scale
    DELTA_CLIP = delta_clip
    TEMP = temp_val

# ====== Helpers ======
def set_onehot_by_prefix(df, prefix, value):
    col = f"{prefix}_{value}"
    if col in df.columns: df.at[0, col] = 1

def set_onehot_by_prefix_multi(df, prefix, values):
    for v in values:
        col = f"{prefix}_{v}"
        if col in df.columns: df.at[0, col] = 1

def flag_yes(row, prefix):
    col = f"{prefix}_Yes"; return (col in row.index) and (row[col] == 1)

def risk_bins(score, mod=20, high=40, band=BORDER_BAND):
    if score >= high + band: return "High"
    if score >= high - band: return "Moderate–High"
    if score >= mod + band:  return "Moderate"
    if score >= mod - band:  return "Low–Moderate"
    return "Low"

def proba_to_percent(p): return float(p) * 100
def proba_to_score(p): return int(round(proba_to_percent(p)))

# ====== Model load / train (monotone) + optional isotonic calibration ======
def get_monotone_constraints(feature_names):
    mono_map = {
        "num_previous_admissions": +1,
        "medication_compliance_score": -1,
        "family_support_score": -1,
        "post_discharge_followups": -1,
        "length_of_stay": +1,
        "financial_strain_score": +1,
    }
    cons = [str(mono_map.get(f, 0)) for f in feature_names]
    return "(" + ",".join(cons) + ")"

def xgb_model_with_monotone(feature_names):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
        monotone_constraints=get_monotone_constraints(feature_names)
    )

def try_load_model(path="dropout_model.pkl"):
    if not os.path.exists(path): return None
    try:
        mdl = joblib.load(path)
        if isinstance(mdl, dict) and "model" in mdl: return mdl["model"]
        return mdl
    except Exception:
        return None

def align_df_to_model(df: pd.DataFrame, m):
    names = None
    try:
        booster = getattr(m, "get_booster", lambda: None)()
        if booster is not None:
            nm = getattr(booster, "feature_names", None)
            if nm: names = list(nm)
    except Exception:
        pass
    if names:
        aligned = pd.DataFrame(0, index=df.index, columns=names, dtype=np.float32)
        inter = [c for c in names if c in df.columns]
        aligned.loc[:, inter] = df[inter].astype(np.float32).values
        return aligned, names
    out = df.astype(np.float32)
    return out, list(out.columns)

def train_demo_model_and_calibrator(columns):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV

    rng = np.random.default_rng(42)
    n = 12000
    X = pd.DataFrame(0, index=range(n), columns=columns, dtype=np.float32)

    # 基本連續特徵（LOS 接近 3–4 週）
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(21.0, 8.0, n).clip(3, 60)  # ← 3~60天
    X["num_previous_admissions"] = rng.poisson(1.0, n).clip(0, 12)
    X["medication_compliance_score"] = rng.normal(6.0, 2.5, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 6, n)
    X["financial_strain_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)

    # one-hots
    idx_gender = rng.integers(0, len(GENDER_LIST), n)
    for i, g in enumerate(GENDER_LIST): X.loc[idx_gender == i, f"gender_{g}"] = 1

    idx_primary = rng.integers(0, len(DIAG_LIST), n)
    for i, d in enumerate(DIAG_LIST): X.loc[idx_primary == i, f"diagnosis_{d}"] = 1

    # 住院主因 / 居住 / 個管 / 期別 / 既往因 dropout 再住院
    idx_chief = rng.integers(0, len(CHIEF_PROBLEMS), n)
    for i, c in enumerate(CHIEF_PROBLEMS): X.loc[idx_chief == i, f"chief_problem_{c}"] = 1
    X["living_alone_Yes"] = (rng.random(n) < 0.35).astype(int)
    X["living_alone_No"] = 1 - X["living_alone_Yes"]
    X["has_case_manager_Yes"] = (rng.random(n) < 0.4).astype(int)
    X["has_case_manager_No"] = 1 - X["has_case_manager_Yes"]
    be_idx = rng.integers(0, len(BIPOLAR_EPISODES), n)
    for i, b in enumerate(BIPOLAR_EPISODES):
        X.loc[be_idx == i, f"bipolar_episode_{b}"] = 1
    X["prior_dropout_readmit_Yes"] = (rng.random(n) < 0.12).astype(int)
    X["prior_dropout_readmit_No"]  = 1 - X["prior_dropout_readmit_Yes"]

    # 自傷旗標：PD/PTSD/SUD 提高
    base_self = rng.random(n)
    pd_mask  = (X.get("diagnosis_Personality Disorder",0)==1)
    ptsd_mask= (X.get("diagnosis_PTD",X.get("diagnosis_PTSD",0))==1)
    sud_mask = (X.get("diagnosis_Substance Use Disorder",0)==1)
    boost = 0.10*pd_mask + 0.08*ptsd_mask + 0.10*sud_mask
    r1 = (base_self < (0.18 + boost)).astype(int)
    r2 = (rng.random(n) < (0.12 + 0.5*boost)).astype(int)
    X["has_recent_self_harm_Yes"] = r1; X["has_recent_self_harm_No"] = 1-r1
    X["self_harm_during_admission_Yes"] = r2; X["self_harm_during_admission_No"] = 1-r2

    # 真值生成（加上新特徵影響）
    beta0 = -0.55
    prev_ge2 = (X["num_previous_admissions"] >= 2).astype(np.float32)
    logit = (beta0
             + 0.85*X["has_recent_self_harm_Yes"]
             + 0.65*X["self_harm_during_admission_Yes"]
             + 0.60*prev_ge2
             - 0.26*X["medication_compliance_score"]
             - 0.20*X["family_support_score"]
             - 0.15*X["post_discharge_followups"]
             + 0.05*X["length_of_stay"]
             + 0.05*X["financial_strain_score"]
             + 0.35*X["prior_dropout_readmit_Yes"]
             + 0.15*X["living_alone_Yes"]
             - 0.12*X["has_case_manager_Yes"])
    # chief
    for k,w in POLICY["chief"].items():
        col=f"chief_problem_{k}"
        if col in X.columns: logit += w * X[col]
    # bipolar episode
    for k,w in POLICY["bipolar_episode"].items():
        col=f"bipolar_episode_{k}"
        if col in X.columns: logit += w * X[col]
    # 診斷
    for d, w in POLICY["diag"].items(): 
        col=f"diagnosis_{d}"
        if col in X.columns: logit += w * X[col]
    # 噪音
    noise = rng.normal(0.0, 0.35, n).astype(np.float32)
    # ✅ 修正：先轉 numpy 再進 np.exp，避免 pandas array_ufunc 錯誤
    logit_np = logit.to_numpy(np.float32)
    p = 1.0 / (1.0 + np.exp(-(logit_np + noise)))
    y = (rng.random(n) < p).astype(np.int32)

    model = xgb_model_with_monotone(list(X.columns))
    model.fit(X, y)

    calibrator = None
    try:
        X_tr, X_ca, y_tr, y_ca = train_test_split(X, y, test_size=0.2, random_state=777)
        calibrator = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        calibrator.fit(X_ca, y_ca)
    except Exception:
        calibrator = None

    return model, calibrator, "demo (monotone + isotonic if available)"

def get_feat_names(m):
    try:
        b = m.get_booster()
        if getattr(b, "feature_names", None): return list(b.feature_names)
    except Exception: pass
    if hasattr(m, "feature_names_in_"): return list(m.feature_names_in_)
    return None

_loaded = try_load_model()
if _loaded is None:
    model, calibrator, model_source = train_demo_model_and_calibrator(TEMPLATE_COLUMNS)
else:
    model, calibrator, model_source = _loaded, None, "loaded from dropout_model.pkl"

def predict_model_proba(df_aligned: pd.DataFrame):
    probs = model.predict_proba(df_aligned, validate_features=False)[:, 1]
    if calibrator is not None:
        try:
            probs = calibrator.predict_proba(df_aligned)[:, 1]
        except Exception:
            pass
    return probs

# ====== Overlay（單例 + 驅動因子）=====
def overlay_single_and_drivers(X1: pd.DataFrame, include_followup_effect: bool = True):
    row = X1.iloc[0]
    drivers = []
    def add(label, val):
        if val != 0: drivers.append((label, float(val)))
        return val
    base_logit = _logit( float(model.predict_proba(X1, validate_features=False)[:,1][0]) )
    lz = base_logit

    adm = _num_or_default(row["num_previous_admissions"], "num_previous_admissions")
    comp = _num_or_default(row["medication_compliance_score"], "medication_compliance_score")
    sup  = _num_or_default(row["family_support_score"], "family_support_score")
    fup  = _num_or_default(row["post_discharge_followups"], "post_discharge_followups")
    los  = _num_or_default(row["length_of_stay"], "length_of_stay")
    agev = _num_or_default(row["age"], "age")
    fin  = _num_or_default(row["financial_strain_score"], "financial_strain_score")

    lz += add("More previous admissions", POLICY["per_prev_admission"] * min(int(adm), 5))
    lz += add("Low family support", POLICY["per_point_low_support"] * max(0.0, 5.0 - sup))
    lz += add("Low medication compliance", POLICY["per_point_low_compliance"] * max(0.0, 5.0 - comp))
    if comp >= 8:
        lz += add("High compliance (protective)", POLICY["per_point_high_compliance_protect"] * (comp - 7.0))

    if include_followup_effect:
        lz += add("More post-discharge followups (protective)", POLICY["per_followup"] * fup)
        if fup == 0: lz += add("No follow-up scheduled", POLICY["no_followup_extra"])

    if los < 3: lz += add("Very short stay (<3d)", POLICY["los_short"])
    elif los <= 21: lz += add("Typical stay (3–21d)", POLICY["los_mid"])
    elif los <= 35: lz += add("Longish stay (22–35d)", POLICY["los_mid_high"])
    else: lz += add("Very long stay (>35d)", POLICY["los_long"])

    if agev < 21: lz += add("Young age (<21)", POLICY["age_young"])
    elif agev >= 75: lz += add("Older age (≥75)", POLICY["age_old"])

    # 新增特徵
    lz += add("Living alone", POLICY["living_alone"] * int(row.get("living_alone_Yes",0)==1))
    lz += add("No case manager", POLICY["no_case_manager"] * int(row.get("has_case_manager_No",0)==1))
    lz += add("Financial strain", POLICY["per_point_financial_strain"] * fin)
    lz += add("Prior dropout readmission", POLICY["prior_dropout_readmit"] * int(row.get("prior_dropout_readmit_Yes",0)==1))

    # chief / bipolar
    for k,w in POLICY["chief"].items():
        col=f"chief_problem_{k}"
        if col in X1.columns and X1.at[0,col]==1: lz += add(f"Chief: {k}", w)
    for k,w in POLICY["bipolar_episode"].items():
        col=f"bipolar_episode_{k}"
        if col in X1.columns and X1.at[0,col]==1: lz += add(f"Bipolar episode: {k}", w)

    # 診斷與交互
    for dx, w in POLICY["diag"].items():
        if X1.at[0, f"diagnosis_{dx}"] == 1:
            lz += add(f"Diagnosis: {dx}", w)
    if (X1.at[0, "diagnosis_Substance Use Disorder"] == 1) and (comp <= 3):
        lz += add("SUD × very low compliance", POLICY["x_sud_lowcomp"])
    if (X1.at[0, "diagnosis_Personality Disorder"] == 1) and (los < 3):
        lz += add("PD × very short stay", POLICY["x_pd_shortlos"])

    delta = np.clip(OVERLAY_SCALE * (lz - base_logit), -DELTA_CLIP, DELTA_CLIP)
    lz2 = base_logit + delta + CAL_LOGIT_SHIFT
    p_overlay = _sigmoid(lz2 / TEMP)
    return float(p_overlay), drivers

# ====== Build single-row DF ======
X_single = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
for k, v in {
    "age": age, "length_of_stay": float(length_of_stay), "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance), "family_support_score": float(support),
    "post_discharge_followups": int(followups), "financial_strain_score": float(financial),
}.items(): X_single.at[0, k] = v
set_onehot_by_prefix(X_single, "gender", gender)
set_onehot_by_prefix_multi(X_single, "diagnosis", diagnoses)
set_onehot_by_prefix(X_single, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_single, "self_harm_during_admission", selfharm_adm)
set_onehot_by_prefix(X_single, "chief_problem", chief)
set_onehot_by_prefix(X_single, "bipolar_episode", bipolar_ep)
set_onehot_by_prefix(X_single, "prior_dropout_readmit", prior_dr)
set_onehot_by_prefix(X_single, "has_case_manager", has_cm)
set_onehot_by_prefix(X_single, "living_alone", "Yes" if living=="Alone" else "No")
fill_defaults_single_row(X_single)

X_used = X_single.copy()
if not use_followups_feature:
    X_used.at[0, "post_discharge_followups"] = 0

X_align, _ = align_df_to_model(X_used, model)
p_model = float(predict_model_proba(X_align)[0])
p_overlay, drivers = overlay_single_and_drivers(X_used, include_followup_effect=use_followups_feature)
p_final = (1.0 - BLEND_W) * p_model + BLEND_W * p_overlay

if flag_yes(X_used.iloc[0], "has_recent_self_harm") or flag_yes(X_used.iloc[0], "self_harm_during_admission"):
    p_final = min(max(p_final, SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"], SOFT_UPLIFT["cap"])

percent_model = proba_to_percent(p_model)
percent_overlay = proba_to_percent(p_overlay)
percent_final = proba_to_percent(p_final)
score = proba_to_score(p_final)
level = risk_bins(score)

# ====== Guards ======
warns = []
if length_of_stay > 90: warns.append("Length of stay > 90d is unusual in acute settings; check data.")
if (num_adm > 10): warns.append("Previous admissions > 10 in 1y is uncommon; confirm definition.")
if warns:
    st.info("ℹ️ Data sanity check:\n- " + "\n- ".join(warns))

# ====== Show results ======
st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Model Probability", f"{percent_model:.1f}%")
with c2: st.metric("Overlay Probability", f"{percent_overlay:.1f}%")
with c3: st.metric("Final Probability", f"{percent_final:.1f}%")
with c4: st.metric("Risk Score (0–100)", f"{score}")

note = "Pre-planning mode: follow-ups feature is ignored." if not use_followups_feature else "Post-planning mode: follow-ups feature is used (within 30 days)."
st.caption(note)

if level == "High":
    st.error("🔴 High Risk")
elif level == "Moderate–High":
    st.warning("🟠 Moderate–High (borderline to High)")
elif level == "Moderate":
    st.warning("🟡 Moderate Risk")
elif level == "Low–Moderate":
    st.info("🔵 Low–Moderate (borderline to Moderate)")
else:
    st.success("🟢 Low Risk")

# ====== SHAP + Policy drivers + 對齊卡 ======
with st.expander("🔍 Explanations — Model SHAP vs Policy drivers", expanded=True):
    import xgboost as xgb
    try:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_align, feature_names=list(X_align.columns))
        contribs = booster.predict(dmat, pred_contribs=True, validate_features=False)
        contrib = np.asarray(contribs)[0]
        base_value = float(contrib[-1])
        sv_map = dict(zip(list(X_align.columns), contrib[:-1]))
    except Exception:
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_align)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
            base_value = base_value[0]
            if isinstance(sv_raw, list): sv_raw = sv_raw[0]
        sv_map = dict(zip(list(X_align.columns), sv_raw[0]))

    feat_rows = []
    def _push_shap(label, key, shown_value):
        if key in sv_map:
            feat_rows.append({"feature": label, "value": shown_value, "model_shap": float(sv_map[key]), "key": key})

    for lab,key in [
        ("Age","age"),("Length of Stay","length_of_stay"),("Previous Admissions","num_previous_admissions"),
        ("Medication Compliance","medication_compliance_score"),("Family Support","family_support_score"),
        ("Followups (30d)","post_discharge_followups"),("Financial strain","financial_strain_score")
    ]:
        _push_shap(lab, key, X_used.at[0,key])

    for dx in diagnoses: _push_shap(f"Diagnosis={dx}", f"diagnosis_{dx}", 1)
    _push_shap(f"Gender={gender}", f"gender_{gender}", 1)
    _push_shap(f"Recent Self-harm={recent_self_harm}", f"has_recent_self_harm_{recent_self_harm}", 1)
    _push_shap(f"Self-harm During Admission={selfharm_adm}", f"self_harm_during_admission_{selfharm_adm}", 1)
    _push_shap(f"Chief={chief}", f"chief_problem_{chief}", 1)
    _push_shap(f"Bipolar episode={bipolar_ep}", f"bipolar_episode_{bipolar_ep}", 1)
    _push_shap(f"Living alone={'Yes' if living=='Alone' else 'No'}", f"living_alone_{'Yes' if living=='Alone' else 'No'}", 1)
    _push_shap(f"Case manager={has_cm}", f"has_case_manager_{has_cm}", 1)
    _push_shap(f"Prior dropout readmit={prior_dr}", f"prior_dropout_readmit_{prior_dr}", 1)

    df_shap = pd.DataFrame(feat_rows)
    if len(df_shap):
        df_top = df_shap.reindex(df_shap["model_shap"].abs().sort_values(ascending=False).index).head(12)
        exp = shap.Explanation(
            values=df_top["model_shap"].to_numpy(float),
            base_values=float(base_value),
            feature_names=df_top["feature"].tolist(),
            data=df_top["value"].to_numpy(float),
        )
        shap.plots.waterfall(exp, show=False, max_display=12)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Model SHAP (top by |value|)")
        st.dataframe(
            df_shap.reindex(df_shap["model_shap"].abs().sort_values(ascending=False).index)[["feature","value","model_shap"]].head(12),
            use_container_width=True
        )
    else:
        st.caption("No SHAP contributions available for the selected case.")

    df_drv = pd.DataFrame(
        [{"driver": k, "policy_log_odds (pre-scale)": round(v, 3)} for k, v in sorted(drivers, key=lambda x: abs(x[1]), reverse=True)]
    )
    st.caption("Policy drivers")
    st.dataframe(df_drv, use_container_width=True)

    st.caption("Alignment check (Model vs Policy) — look for ⚠️ if directions disagree.")
    def _sign(x): return 1 if x>1e-6 else (-1 if x<-1e-6 else 0)
    align_rows = []
    name_map = [
        ("Previous Admissions","num_previous_admissions","More previous admissions"),
        ("Medication Compliance","medication_compliance_score","Low medication compliance"),
        ("Family Support","family_support_score","Low family support"),
        ("Followups (30d)","post_discharge_followups","More post-discharge followups"),
        ("Length of Stay","length_of_stay","Very short stay (<3d) / Long stay"),
        ("Living alone","living_alone_Yes","Living alone"),
        ("Case manager","has_case_manager_No","No case manager"),
        ("Financial strain","financial_strain_score","Financial strain"),
    ]
    sv_map = {r["key"]:r["model_shap"] for _,r in pd.DataFrame(feat_rows).iterrows()} if len(feat_rows) else {}
    for lab, key, dname in name_map:
        shap_v = float(sv_map.get(key, 0.0))
        pol = 0.0
        for nm, v in drivers:
            if dname.split()[0] in nm: pol += v
        ms, ps = _sign(shap_v), _sign(pol)
        align_rows.append({"feature": lab, "model_sign": ms, "policy_sign": ps, "flag": "⚠️" if (ms*ps==-1) else ""})
    st.dataframe(pd.DataFrame(align_rows), use_container_width=True)

# ====== Recommended actions（同前版，小幅調整文案） ======
st.subheader("Recommended Actions")
BASE_ACTIONS = {
    "High": [
        ("Today","Clinician","Crisis/safety planning; 24/7 crisis contacts"),
        ("Today","Clinic scheduler","Return within 7 days (prefer 72h)"),
        ("Today","Care coordinator","Warm handoff to case management"),
        ("48h","Nurse","Outreach call: symptoms/side-effects/barriers"),
        ("7d","Pharmacist/Nurse","Medication review + adherence aids"),
        ("1–2w","Social worker","SDOH screen; transport/financial aid"),
        ("1–4w","Peer support","Enroll in peer/skills group"),
    ],
    "Moderate": [
        ("1–2w","Clinic scheduler","Book within 14 days; SMS reminders"),
        ("1–2w","Nurse","Barrier check & solutions"),
        ("2–4w","Clinician","Brief MI/BA/psychoeducation; 4-week plan"),
    ],
    "Low": [
        ("2–4w","Clinic scheduler","Routine follow-up; confirm reminders"),
        ("2–4w","Nurse","Education/self-management resources"),
    ],
}
def _normalize_action_tuple(a):
    if len(a)==3: return (a[0],a[1],a[2])
    return a

def personalized_actions(row: pd.Series, chosen_dx: list):
    acts = []
    comp = _num_or_default(row["medication_compliance_score"], "medication_compliance_score")
    sup  = _num_or_default(row["family_support_score"], "family_support_score")
    fup  = _num_or_default(row["post_discharge_followups"], "post_discharge_followups")
    los  = _num_or_default(row["length_of_stay"], "length_of_stay")
    agev = _num_or_default(row["age"], "age")
    fin  = _num_or_default(row["financial_strain_score"], "financial_strain_score")
    has_selfharm = flag_yes(row, "has_recent_self_harm") or flag_yes(row, "self_harm_during_admission")
    has_sud = "Substance Use Disorder" in chosen_dx
    has_pd  = "Personality Disorder" in chosen_dx
    has_dep = "Depression" in chosen_dx
    has_scz = "Schizophrenia" in chosen_dx
    living_alone = (row.get("living_alone_Yes",0)==1)
    no_cm = (row.get("has_case_manager_No",0)==1)
    prior_dr = (row.get("prior_dropout_readmit_Yes",0)==1)

    if has_selfharm:
        acts += [("Today","Clinician","C-SSRS; update safety plan; lethal-means counseling"),
                 ("48h","Nurse","Safety check-in call")]
    if has_sud and comp <= 3:
        acts += [("1–7d","Clinician","Brief MI focused on use goals"),
                 ("1–7d","Care coordinator","Refer to SUD program/IOP or CM"),
                 ("Today","Clinician","Overdose prevention education")]
    if has_pd and los < 3:
        acts += [("Today","Care coordinator","Same-day DBT/skills intake"),
                 ("48h","Peer support","Proactive outreach + skills workbook")]
    if comp <= 3:
        acts += [("7d","Pharmacist","Simplify regimen + blister/pillbox + reminders"),
                 ("1–2w","Clinician","Consider LAI if appropriate")]
    if sup <= 2 or living_alone:
        acts += [("1–2w","Clinician","Family meeting / caregiver engagement"),
                 ("1–2w","Social worker","Community supports; transport/financial counseling")]
    if fup == 0:
        acts += [("Today","Clinic scheduler","Book 2 touchpoints in first 14 days (day2/day7)")]
    if fin >= 7:
        acts += [("1–2w","Social worker","Financial aid application; transport vouchers")]
    if no_cm or prior_dr:
        acts += [("Today","Care coordinator","Enroll/confirm case management; relapse prevention")]
    if los < 3:
        acts += [("48h","Nurse","Early call; review meds/barriers")]
    elif los > 35:
        acts += [("1–7d","Care coordinator","Step-down/day program plan + warm handoff")]
    if agev < 21:
        acts += [("1–2w","Clinician","Involve guardians; link school counseling")]
    elif agev >= 75:
        acts += [("1–2w","Nurse/Pharmacist","Med reconciliation; simplify dosing")]
    if has_dep:
        acts += [("1–2w","Clinician","Behavioral activation + activity schedule")]
    if has_scz:
        acts += [("1–4w","Clinician","Relapse plan; early warning signs; caregiver involvement")]
    return acts

bucket = {"High":"High","Moderate–High":"High","Moderate":"Moderate","Low–Moderate":"Low","Low":"Low"}
acts = [ _normalize_action_tuple(a) for a in BASE_ACTIONS[bucket[level]] ]
acts += personalized_actions(X_used.iloc[0], diagnoses)
seen=set(); uniq=[]
for a in acts:
    key=(a[0],a[1],a[2])
    if key not in seen:
        seen.add(key); uniq.append(a)
ORDER={"Today":0,"48h":1,"7d":2,"1–7d":2,"1–2w":3,"2–4w":4,"1–4w":5}
uniq.sort(key=lambda x:(ORDER.get(x[0],99), x[1], x[2]))
st.dataframe(pd.DataFrame(uniq, columns=["Timeline","Owner","Action"]), use_container_width=True)

if level in ["High","Moderate–High"]:
    def make_sop_txt(score, label, actions):
        lines=["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}",""]
        for (tl,ow,ac) in actions: lines.append(f"- {tl} | {ow} | {ac}")
        buf=BytesIO("\n".join(lines).encode("utf-8")); buf.seek(0); return buf
    st.download_button("⬇️ Export SOP (TXT)", make_sop_txt(score, level, uniq),
                       file_name="dropout_risk_SOP.txt", mime="text/plain")
    if HAS_DOCX:
        def make_docx(score,label,actions):
            doc=Document(); doc.add_heading('Psychiatric Dropout Risk – SOP', level=1)
            doc.add_paragraph(f"Risk score: {score}/100 | Risk level: {label}")
            t=doc.add_table(rows=1, cols=3); hdr=t.rows[0].cells
            hdr[0].text, hdr[1].text, hdr[2].text = 'Timeline','Owner','Action'
            for (tl,ow,ac) in actions:
                r=t.add_row().cells; r[0].text, r[1].text, r[2].text = tl,ow,ac
            buf=BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== What-if ======
with st.expander("🧪 What-if: adjust followups/compliance and recompute", expanded=False):
    wf_follow = st.slider("What-if followups (30d)", 0, 10, int(X_single.at[0,"post_discharge_followups"]))
    wf_comp   = st.slider("What-if compliance", 0.0, 10.0, float(X_single.at[0,"medication_compliance_score"]), 0.5)
    X_wf = X_used.copy()
    X_wf.at[0,"post_discharge_followups"] = wf_follow if use_followups_feature else 0
    X_wf.at[0,"medication_compliance_score"] = wf_comp
    X_wf_al, _ = align_df_to_model(X_wf, model)
    p_m_wf = float(predict_model_proba(X_wf_al)[0])
    p_o_wf, _ = overlay_single_and_drivers(X_wf, include_followup_effect=use_followups_feature)
    p_f_wf = (1.0 - BLEND_W) * p_m_wf + BLEND_W * p_o_wf
    st.write(f"Model={p_m_wf*100:.1f}% | Overlay={p_o_wf*100:.1f}% | Final={p_f_wf*100:.1f}%")

# ====== Batch Prediction ======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")
friendly_cols = [
    "Age","Gender","Diagnoses","Length of Stay (days)","Previous Admissions (1y)",
    "Medication Compliance Score (0–10)","Family Support Score (0–10)","Post-discharge Followups (30d)",
    "Financial Strain (0–10)","Living status","Has case manager?","Prior dropout-related readmission?",
    "Chief problem","Bipolar current episode","Recent Self-harm","Self-harm During Admission"
]
tpl = pd.DataFrame(columns=friendly_cols)
buf_tpl = BytesIO(); tpl.to_excel(buf_tpl, index=False); buf_tpl.seek(0)
st.download_button("📥 Download Excel Template", buf_tpl, file_name="batch_template.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])
def parse_multi(cell):
    parts = [p.strip() for p in re.split(r"[;,/|]", str(cell)) if p.strip()]
    return parts if parts else []

if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = pd.DataFrame(0, index=raw.index, columns=TEMPLATE_COLUMNS, dtype=float)

        def safe(col, default=0): return raw[col] if col in raw.columns else default
        mapping = [
            ("Age","age"), ("Length of Stay (days)","length_of_stay"),
            ("Previous Admissions (1y)","num_previous_admissions"),
            ("Medication Compliance Score (0–10)","medication_compliance_score"),
            ("Family Support Score (0–10)","family_support_score"),
            ("Post-discharge Followups (30d)","post_discharge_followups"),
            ("Financial Strain (0–10)","financial_strain_score"),
        ]
        for k_raw, k in mapping: df[k] = safe(k_raw)

        # one-hots
        if "Gender" in raw.columns:
            for i, v in raw["Gender"].astype(str).str.strip().items():
                col=f"gender_{v}"; 
                if col in df.columns: df.at[i,col]=1
        if "Diagnoses" in raw.columns:
            for i, cell in raw["Diagnoses"].items():
                for v in parse_multi(cell):
                    col=f"diagnosis_{v}"
                    if col in df.columns: df.at[i,col]=1

        # 新增欄位
        if "Living status" in raw.columns:
            for i, v in raw["Living status"].astype(str).str.strip().items():
                df.at[i, f"living_alone_{'Yes' if v=='Alone' else 'No'}"] = 1
        if "Has case manager?" in raw.columns:
            for i, v in raw["Has case manager?"].astype(str).str.strip().items():
                df.at[i, f"has_case_manager_{v}"] = 1
        if "Prior dropout-related readmission?" in raw.columns:
            for i, v in raw["Prior dropout-related readmission?"].astype(str).str.strip().items():
                df.at[i, f"prior_dropout_readmit_{v}"] = 1
        if "Chief problem" in raw.columns:
            for i, v in raw["Chief problem"].astype(str).str.strip().items():
                col=f"chief_problem_{v}"
                if col in df.columns: df.at[i,col]=1
        if "Bipolar current episode" in raw.columns:
            for i, v in raw["Bipolar current episode"].astype(str).str.strip().items():
                col=f"bipolar_episode_{v}"
                if col in df.columns: df.at[i,col]=1

        for col_h, pre in [("Recent Self-harm","has_recent_self_harm"), ("Self-harm During Admission","self_harm_during_admission")]:
            if col_h in raw.columns:
                for i, v in raw[col_h].astype(str).str.strip().items():
                    col=f"{pre}_{v}"; 
                    if col in df.columns: df.at[i,col]=1

        fill_defaults_batch(df)

        if not use_followups_feature:
            df["post_discharge_followups"] = 0

        Xb_al, _ = align_df_to_model(df, model)
        base_probs = predict_model_proba(Xb_al)

        def overlay_vec(df_feat: pd.DataFrame, include_followup=True):
            base = _logit_vec(base_probs); lz = base.copy()
            adm = pd.to_numeric(df_feat["num_previous_admissions"], errors="coerce").fillna(DEFAULTS["num_previous_admissions"]).to_numpy()
            comp= pd.to_numeric(df_feat["medication_compliance_score"], errors="coerce").fillna(DEFAULTS["medication_compliance_score"]).to_numpy()
            sup = pd.to_numeric(df_feat["family_support_score"], errors="coerce").fillna(DEFAULTS["family_support_score"]).to_numpy()
            fup = pd.to_numeric(df_feat["post_discharge_followups"], errors="coerce").fillna(DEFAULTS["post_discharge_followups"]).to_numpy()
            los = pd.to_numeric(df_feat["length_of_stay"], errors="coerce").fillna(DEFAULTS["length_of_stay"]).to_numpy()
            agev= pd.to_numeric(df_feat["age"], errors="coerce").fillna(DEFAULTS["age"]).to_numpy()
            fin = pd.to_numeric(df_feat["financial_strain_score"], errors="coerce").fillna(DEFAULTS["financial_strain_score"]).to_numpy()

            lz += POLICY["per_prev_admission"] * np.minimum(adm, 5)
            lz += POLICY["per_point_low_support"] * np.maximum(0.0, 5.0 - sup)
            lz += POLICY["per_point_low_compliance"] * np.maximum(0.0, 5.0 - comp)
            lz += POLICY["per_point_high_compliance_protect"] * np.maximum(0.0, comp - 7.0)
            if include_followup:
                lz += POLICY["per_followup"] * fup
                lz += POLICY["no_followup_extra"] * (fup == 0)
            lz += np.where(los < 3, POLICY["los_short"],
                     np.where(los <= 21, POLICY["los_mid"],
                     np.where(los <= 35, POLICY["los_mid_high"], POLICY["los_long"])))
            lz += POLICY["age_young"] * (agev < 21) + POLICY["age_old"] * (agev >= 75)

            for dx, w in POLICY["diag"].items():
                col=f"diagnosis_{dx}"
                if col in df_feat.columns: lz += w * (df_feat[col].to_numpy() == 1)
            sud = (df_feat.get("diagnosis_Substance Use Disorder",0).to_numpy()==1)
            pdm = (df_feat.get("diagnosis_Personality Disorder",0).to_numpy()==1)
            lz += POLICY["x_sud_lowcomp"] * (sud & (comp <= 3))
            lz += POLICY["x_pd_shortlos"] * (pdm & (los < 3))

            # 新增項
            lz += POLICY["living_alone"] * (df_feat.get("living_alone_Yes",0).to_numpy()==1)
            lz += POLICY["no_case_manager"] * (df_feat.get("has_case_manager_No",0).to_numpy()==1)
            lz += POLICY["per_point_financial_strain"] * fin
            lz += POLICY["prior_dropout_readmit"] * (df_feat.get("prior_dropout_readmit_Yes",0).to_numpy()==1)

            for k,w in POLICY["chief"].items():
                col=f"chief_problem_{k}"
                if col in df_feat.columns: lz += w * (df_feat[col].to_numpy()==1)
            for k,w in POLICY["bipolar_episode"].items():
                col=f"bipolar_episode_{k}"
                if col in df_feat.columns: lz += w * (df_feat[col].to_numpy()==1)

            delta = np.clip(OVERLAY_SCALE * (lz - base), -DELTA_CLIP, DELTA_CLIP)
            lz2 = base + delta + CAL_LOGIT_SHIFT
            return 1.0 / (1.0 + np.exp(-(lz2 / TEMP)))

        p_overlay_b = overlay_vec(df, include_followup=use_followups_feature)
        p_final_b = (1.0 - BLEND_W) * base_probs + BLEND_W * p_overlay_b

        hr = df.get("has_recent_self_harm_Yes", 0); ha = df.get("self_harm_during_admission_Yes", 0)
        mask = ((np.array(hr)==1) | (np.array(ha)==1))
        p_final_b[mask] = np.minimum(np.maximum(p_final_b[mask], SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"], SOFT_UPLIFT["cap"])

        out = raw.copy()
        out["risk_percent"] = (p_final_b*100).round(1)
        out["risk_score_0_100"] = (p_final_b*100).round().astype(int)

        s = out["risk_score_0_100"].to_numpy()
        levels = np.full(s.shape, "Low", dtype=object)
        levels[s >= 20 - BORDER_BAND] = "Low–Moderate"
        levels[s >= 20 + BORDER_BAND] = "Moderate"
        levels[s >= 40 - BORDER_BAND] = "Moderate–High"
        levels[s >= 40 + BORDER_BAND] = "High"
        out["risk_level"] = levels

        st.dataframe(out, use_container_width=True)
        buf = BytesIO(); out.to_csv(buf, index=False); buf.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Batch error: {e}")

# ====== Validation（Ablation + Decision curve + Capacity + Fairness）======
st.markdown("---")
st.header("✅ Validation (synthetic hold-out)")

cA, cB = st.columns([2,1])
with cA:
    n_val = st.slider("Number of synthetic patients", 5000, 60000, 20000, 5000)
    seed_val = st.number_input("Random seed", min_value=1, max_value=10**9, value=2024, step=1)
with cB:
    pt_low = st.slider("Decision threshold (Moderate%):", 5, 60, 20, 1)
    pt_high = st.slider("Decision threshold (High%):", 20, 90, 40, 1)
run_val = st.button("Run validation")

def generate_synth_holdout(n=20000, seed=2024):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(0, index=range(n), columns=TEMPLATE_COLUMNS, dtype=float)
    df["age"] = rng.integers(16, 85, n)
    df["length_of_stay"] = rng.normal(21.0, 8.0, n).clip(3, 60)
    df["num_previous_admissions"] = rng.poisson(1.0, n).clip(0, 12)
    df["medication_compliance_score"] = rng.normal(6.0, 2.5, n).clip(0, 10)
    df["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    df["post_discharge_followups"] = rng.integers(0, 6, n)
    df["financial_strain_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    idx_gender = rng.integers(0, len(GENDER_LIST), n)
    for i, g in enumerate(GENDER_LIST): df.loc[idx_gender == i, f"gender_{g}"] = 1
    idx_primary = rng.integers(0, len(DIAG_LIST), n)
    for i, d in enumerate(DIAG_LIST): df.loc[idx_primary == i, f"diagnosis_{d}"] = 1
    idx_chief = rng.integers(0, len(CHIEF_PROBLEMS), n)
    for i, c in enumerate(CHIEF_PROBLEMS): df.loc[idx_chief == i, f"chief_problem_{c}"] = 1
    be_idx = rng.integers(0, len(BIPOLAR_EPISODES), n)
    for i, b in enumerate(BIPOLAR_EPISODES): df.loc[be_idx == i, f"bipolar_episode_{b}"] = 1
    df["living_alone_Yes"] = (rng.random(n) < 0.35).astype(int); df["living_alone_No"] = 1-df["living_alone_Yes"]
    df["has_case_manager_Yes"] = (rng.random(n) < 0.4).astype(int); df["has_case_manager_No"] = 1-df["has_case_manager_Yes"]
    df["prior_dropout_readmit_Yes"] = (rng.random(n) < 0.12).astype(int); df["prior_dropout_readmit_No"] = 1-df["prior_dropout_readmit_Yes"]

    base_self = rng.random(n)
    pd_mask  = (df.get("diagnosis_Personality Disorder",0)==1)
    ptsd_mask= (df.get("diagnosis_PTSD",0)==1)
    sud_mask = (df.get("diagnosis_Substance Use Disorder",0)==1)
    boost = 0.10*pd_mask + 0.08*ptsd_mask + 0.10*sud_mask
    r1 = (base_self < (0.18 + boost)).astype(int)
    r2 = (rng.random(n) < (0.12 + 0.5*boost)).astype(int)
    df["has_recent_self_harm_Yes"] = r1; df["has_recent_self_harm_No"] = 1-r1
    df["self_harm_during_admission_Yes"] = r2; df["self_harm_during_admission_No"] = 1-r2

    beta0 = -0.55
    prev_ge2 = (pd.to_numeric(df["num_previous_admissions"]) >= 2).astype(np.float32)
    logit = (beta0
             + 0.85*df["has_recent_self_harm_Yes"]
             + 0.65*df["self_harm_during_admission_Yes"]
             + 0.60*prev_ge2
             - 0.26*df["medication_compliance_score"]
             - 0.20*df["family_support_score"]
             - 0.15*df["post_discharge_followups"]
             + 0.05*df["length_of_stay"]
             + 0.05*df["financial_strain_score"]
             + 0.35*df["prior_dropout_readmit_Yes"]
             + 0.15*df["living_alone_Yes"]
             - 0.12*df["has_case_manager_Yes"])
    for k,w in POLICY["chief"].items():
        col=f"chief_problem_{k}"
        if col in df.columns: logit += w * df[col]
    for k,w in POLICY["bipolar_episode"].items():
        col=f"bipolar_episode_{k}"
        if col in df.columns: logit += w * df[col]
    for d,w in POLICY["diag"].items():
        col=f"diagnosis_{d}"
        if col in df.columns: logit += w * df[col]

    noise = np.random.default_rng(seed+1).normal(0.0, 0.35, n).astype(np.float32)
    logit_np = pd.Series(logit).to_numpy(np.float32)
    p_true = 1.0 / (1.0 + np.exp(-(logit_np + noise)))
    y_true = (np.random.default_rng(seed+2).random(n) < p_true).astype(int)

    fill_defaults_batch(df)
    return df, y_true

def plot_roc_pr(y, p_list, labels):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fig1, ax1 = plt.subplots()
    for p,l in zip(p_list, labels):
        fpr, tpr, _ = roc_curve(y, p); roc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{l} AUC={roc:.3f}")
    ax1.plot([0,1],[0,1],"--")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC")
    ax1.legend(loc="lower right"); st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots()
    for p,l in zip(p_list, labels):
        prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
        ax2.plot(rec, prec, label=f"{l} AP={ap:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR")
    ax2.legend(loc="upper right"); st.pyplot(fig2, clear_figure=True)

def ece(y, p, n_bins=10):
    bins = np.linspace(0.0,1.0,n_bins+1)
    idx = np.digitize(p, bins)-1
    err=0.0
    for b in range(n_bins):
        m=(idx==b)
        if m.sum()==0: continue
        fp=p[m].mean(); tp=y[m].mean()
        err += m.mean()*abs(tp-fp)
    return float(err)

def confusion(y, p, thr):
    from sklearn.metrics import confusion_matrix
    yhat = (p>=thr).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,yhat).ravel()
    return tn,fp,fn,tp

def decision_curve(y, p):
    ths = np.linspace(0.05,0.60,56)
    N=len(y)
    nb=[]
    for t in ths:
        yhat = (p>=t).astype(int)
        tp = ((yhat==1)&(y==1)).sum(); fp=((yhat==1)&(y==0)).sum()
        nb.append((tp/N) - (fp/N)*(t/(1-t)))
    return ths, np.array(nb)

if run_val:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    except Exception as e:
        st.error(f"Need scikit-learn: {e}")
    else:
        with st.spinner("Generating data & evaluating..."):
            df_syn, y_true = generate_synth_holdout(n_val, seed_val)
            if not use_followups_feature:
                df_syn["post_discharge_followups"] = 0
            Xa, _ = align_df_to_model(df_syn, model)
            p_model_v = predict_model_proba(Xa)

            def overlay_only_vec(df_feat, base_probs):
                base = _logit_vec(base_probs); lz = base.copy()
                adm = pd.to_numeric(df_feat["num_previous_admissions"], errors="coerce").fillna(DEFAULTS["num_previous_admissions"]).to_numpy()
                comp= pd.to_numeric(df_feat["medication_compliance_score"], errors="coerce").fillna(DEFAULTS["medication_compliance_score"]).to_numpy()
                sup = pd.to_numeric(df_feat["family_support_score"], errors="coerce").fillna(DEFAULTS["family_support_score"]).to_numpy()
                fup = pd.to_numeric(df_feat["post_discharge_followups"], errors="coerce").fillna(DEFAULTS["post_discharge_followups"]).to_numpy()
                los = pd.to_numeric(df_feat["length_of_stay"], errors="coerce").fillna(DEFAULTS["length_of_stay"]).to_numpy()
                agev= pd.to_numeric(df_feat["age"], errors="coerce").fillna(DEFAULTS["age"]).to_numpy()
                fin = pd.to_numeric(df_feat["financial_strain_score"], errors="coerce").fillna(DEFAULTS["financial_strain_score"]).to_numpy()

                lz += POLICY["per_prev_admission"] * np.minimum(adm, 5)
                lz += POLICY["per_point_low_support"] * np.maximum(0.0, 5.0 - sup)
                lz += POLICY["per_point_low_compliance"] * np.maximum(0.0, 5.0 - comp)
                lz += POLICY["per_point_high_compliance_protect"] * np.maximum(0.0, comp - 7.0)
                if use_followups_feature:
                    lz += POLICY["per_followup"] * fup
                    lz += POLICY["no_followup_extra"] * (fup == 0)
                lz += np.where(los < 3, POLICY["los_short"],
                        np.where(los <= 21, POLICY["los_mid"],
                        np.where(los <= 35, POLICY["los_mid_high"], POLICY["los_long"])))
                lz += POLICY["age_young"] * (agev < 21) + POLICY["age_old"]*(agev >= 75)

                for dx,w in POLICY["diag"].items():
                    col=f"diagnosis_{dx}"
                    if col in df_feat.columns: lz += w * (df_feat[col].to_numpy()==1)
                sud = (df_feat.get("diagnosis_Substance Use Disorder",0).to_numpy()==1)
                pdm = (df_feat.get("diagnosis_Personality Disorder",0).to_numpy()==1)
                lz += POLICY["x_sud_lowcomp"] * (sud & (comp <= 3))
                lz += POLICY["x_pd_shortlos"] * (pdm & (los < 3))

                lz += POLICY["living_alone"] * (df_feat.get("living_alone_Yes",0).to_numpy()==1)
                lz += POLICY["no_case_manager"] * (df_feat.get("has_case_manager_No",0).to_numpy()==1)
                lz += POLICY["per_point_financial_strain"] * fin
                lz += POLICY["prior_dropout_readmit"] * (df_feat.get("prior_dropout_readmit_Yes",0).to_numpy()==1)

                for k,w in POLICY["chief"].items():
                    col=f"chief_problem_{k}"
                    if col in df_feat.columns: lz += w * (df_feat[col].to_numpy()==1)
                for k,w in POLICY["bipolar_episode"].items():
                    col=f"bipolar_episode_{k}"
                    if col in df_feat.columns: lz += w * (df_feat[col].to_numpy()==1)

                delta = np.clip(OVERLAY_SCALE * (lz - base), -DELTA_CLIP, DELTA_CLIP)
                lz2 = base + delta + CAL_LOGIT_SHIFT
                return 1.0 / (1.0 + np.exp(-(lz2 / TEMP)))

            p_overlay_v = overlay_only_vec(df_syn, p_model_v)
            p_final_v = (1.0 - BLEND_W) * p_model_v + BLEND_W * p_overlay_v

            auc_m = roc_auc_score(y_true, p_model_v); auc_o = roc_auc_score(y_true, p_overlay_v); auc_f = roc_auc_score(y_true, p_final_v)
            ap_m  = average_precision_score(y_true, p_model_v); ap_o  = average_precision_score(y_true, p_overlay_v); ap_f  = average_precision_score(y_true, p_final_v)
            br_m  = brier_score_loss(y_true, p_model_v); br_o  = brier_score_loss(y_true, p_overlay_v); br_f  = brier_score_loss(y_true, p_final_v)
            ece_m = ece(y_true, p_model_v); ece_o = ece(y_true, p_overlay_v); ece_f = ece(y_true, p_final_v)
            st.subheader("Ablation metrics")
            st.dataframe(pd.DataFrame([
                {"Model":"Model","AUC":auc_m,"PR-AUC":ap_m,"Brier":br_m,"ECE(10bins)":ece_m},
                {"Model":"Overlay-only","AUC":auc_o,"PR-AUC":ap_o,"Brier":br_o,"ECE(10bins)":ece_o},
                {"Model":"Blend (Final)","AUC":auc_f,"PR-AUC":ap_f,"Brier":br_f,"ECE(10bins)":ece_f},
            ]).round(3), use_container_width=True)

            st.subheader("Curves")
            plot_roc_pr(y_true, [p_model_v, p_overlay_v, p_final_v], ["Model","Overlay","Final"])

            ths_m, nb_m = decision_curve(y_true, p_model_v)
            ths_o, nb_o = decision_curve(y_true, p_overlay_v)
            ths_f, nb_f = decision_curve(y_true, p_final_v)
            fig, ax = plt.subplots()
            ax.plot(ths_m, nb_m, label="Model")
            ax.plot(ths_o, nb_o, label="Overlay")
            ax.plot(ths_f, nb_f, label="Final")
            ax.axhline(0, ls="--"); ax.set_xlabel("Threshold"); ax.set_ylabel("Net benefit"); ax.set_title("Decision Curve")
            ax.legend(); st.pyplot(fig, clear_figure=True)

            thr_mod = pt_low/100.0; thr_hi = pt_high/100.0
            st.subheader("Operational (binary @ thresholds)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Moderate ≥{pt_low}%**")
                for name, p in [("Model",p_model_v),("Overlay",p_overlay_v),("Final",p_final_v)]:
                    tn,fp,fn,tp = confusion(y_true, p, thr_mod)
                    st.write(f"{name} — TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
            with c2:
                st.markdown(f"**High ≥{pt_high}%**")
                for name, p in [("Model",p_model_v),("Overlay",p_overlay_v),("Final",p_final_v)]:
                    tn,fp,fn,tp = confusion(y_true, p, thr_hi)
                    st.write(f"{name} — TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

            st.subheader("Capacity / Time load (per 1,000 patients)")
            N = 1000
            def count_at(p, thr): return int(((p>=thr).sum() / len(p)) * N)
            mod_cnt = count_at(p_final_v, thr_mod); high_cnt = count_at(p_final_v, thr_hi)
            st.caption("Assumptions (editable):")
            t_outreach = st.number_input("Nurse outreach (min)", 5, 60, 15, 5)
            t_sched = st.number_input("Scheduler booking (min)", 2, 30, 5, 1)
            t_pharm = st.number_input("Pharmacist review (min)", 5, 60, 20, 5)
            hours = (mod_cnt*(t_outreach+t_sched) + (high_cnt)*(t_pharm)) / 60.0
            st.write(f"Flagged Moderate+ per 1,000: **{mod_cnt}** ; High: **{high_cnt}** → ~ **{hours:.1f} hours** total effort.")

            st.subheader("Fairness (AUC / ECE by subgroup, Final)")
            def subgroup_auc_ece(df_feat, y, p, mask):
                if mask.sum()<100: return np.nan, np.nan
                from sklearn.metrics import roc_auc_score
                return float(roc_auc_score(y[mask], p[mask])), float(ece(y[mask], p[mask], 10))
            agev = pd.to_numeric(df_syn["age"], errors="coerce").fillna(40).to_numpy()
            bands = {"<30": (agev<30), "30–59": ((agev>=30)&(agev<60)), "≥60": (agev>=60)}
            rows=[]
            for g in GENDER_LIST:
                mask = (df_syn.get(f"gender_{g}",0).to_numpy()==1)
                a,e = subgroup_auc_ece(df_syn, y_true, p_final_v, mask)
                rows.append({"group":"Gender", "value":g, "AUC":a, "ECE":e})
            for k,m in bands.items():
                a,e = subgroup_auc_ece(df_syn, y_true, p_final_v, m)
                rows.append({"group":"AgeBand", "value":k, "AUC":a, "ECE":e})
            diag_cols=[f"diagnosis_{d}" for d in DIAG_LIST]
            prim = np.argmax(df_syn[diag_cols].to_numpy(), axis=1)
            for i,d in enumerate(DIAG_LIST):
                mask = (prim==i)
                a,e = subgroup_auc_ece(df_syn, y_true, p_final_v, mask)
                rows.append({"group":"PrimaryDx", "value":d, "AUC":a, "ECE":e})
            st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)

        st.success("Validation finished.")

# ====== Vignettes & Dictionary ======
st.markdown("---")
st.header("🧾 Vignettes template (for expert review)")
def _mk_vignette_row(age, gender, diags, los, prev, comp, rsh, shadm, sup, fup, fin, living, cm, prior, chief, ep):
    return {
        "Age": age, "Gender": gender, "Diagnoses": ", ".join(diags),
        "Length of Stay (days)": los, "Previous Admissions (1y)": prev,
        "Medication Compliance Score (0–10)": comp,
        "Family Support Score (0–10)": sup,
        "Post-discharge Followups (30d)": fup,
        "Financial Strain (0–10)": fin,
        "Living status": living, "Has case manager?": cm, "Prior dropout-related readmission?": prior,
        "Chief problem": chief, "Bipolar current episode": ep,
        "Recent Self-harm": rsh, "Self-harm During Admission": shadm,
        "Expert Risk (Low/Moderate/High or 0–100)": ""
    }
def build_vignettes_df(n=20, seed=77):
    rng = np.random.default_rng(seed); base=[]
    for _ in range(n):
        age = int(np.clip(rng.normal(40,15), 18, 90))
        gender = GENDER_LIST[int(rng.integers(0,len(GENDER_LIST)))]
        k = int(rng.integers(1,3)); diags = list(rng.choice(DIAG_LIST, size=k, replace=False))
        los = int(np.clip(rng.normal(21,8),3,60)); prev = int(np.clip(rng.poisson(1.0),0,8))
        comp = float(np.clip(rng.normal(6,2.5),0,10)); rsh = rng.choice(["Yes","No"]); shadm = rng.choice(["Yes","No"])
        sup = float(np.clip(rng.normal(5,2.5),0,10)); fup = int(np.clip(rng.integers(0,4),0,10))
        fin = float(np.clip(rng.normal(5,2.5),0,10))
        living = rng.choice(["Alone","With family/others"])
        cm = rng.choice(["Yes","No"])
        prior = rng.choice(["Yes","No"], p=[0.12,0.88])
        chief = rng.choice(CHIEF_PROBLEMS)
        ep = rng.choice(BIPOLAR_EPISODES)
        base.append(_mk_vignette_row(age, gender, diags, los, prev, comp, rsh, shadm, sup, fup, fin, living, cm, prior, chief, ep))
    return pd.DataFrame(base)

vdf = build_vignettes_df(20, 77)
buf_v = BytesIO(); vdf.to_excel(buf_v, index=False); buf_v.seek(0)
st.download_button("📥 Download Vignettes (20 cases, Excel)", buf_v,
                   file_name="vignettes_20.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("📚 Data dictionary / Definitions", expanded=False):
    st.markdown("""
- **Medication Compliance (0–10)**：0=幾乎不服藥；10=幾乎完全依從（近 1 個月）
- **Family Support (0–10)**：0=非常不足；10=非常充足
- **Financial Strain (0–10)**：0=無壓力；10=極大壓力
- **Post-discharge Followups (30d)**：出院後 **30 天內** 已安排的接觸次數（門診/電訪/社工）
- **Chief problem**：本次住院主要問題（自傷風險、攻擊/衝動、無法自理、嚴重精神病理/鑑別、家庭難照、藥物不良反應…）
- **Bipolar current episode**：躁期/鬱期/混合期/不適用
- **Living status**：Alone = 獨居；With family/others = 與他人同住
- **Has case manager?**：是否已有個案管理師
- **Prior dropout-related readmission?**：是否曾因未規律追蹤而再住院
- **Pre-planning 模式**：忽略 followups 特徵（避免把「已安排的追蹤」當作預測輸入）
- **Final Probability**：Model 與 Policy Overlay 混合（可調 BLEND），含必要安全 uplift
""")
