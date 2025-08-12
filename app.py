# app.py — Psychiatric Dropout Risk
# 強化：Medication Compliance ≤3 必定明顯拉高（含 piecewise + uplift）
# 功能：multi-dx + null-safe + overlay(scale/clip/temp) + blend + SHAP + Actions + SOP
#       Batch(Drivers/Top3) + Validation + Vignettes

import os
import re
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
st.title("🧠 Psychiatric Dropout Risk Predictor")

# ==== math helpers ====
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps)); return np.log(p / (1 - p))
def _logit_vec(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

# ==== global knobs (more sensitive defaults) ====
CAL_LOGIT_SHIFT = float(os.getenv("RISK_CAL_SHIFT", "0.0"))
BLEND_W = float(os.getenv("RISK_BLEND_W", "0.35"))          # ↑ 混合比重
BORDER_BAND = 7

# Self-harm uplift（較強）
SOFT_UPLIFT = {"floor": 0.60, "add": 0.10, "cap": 0.90}

# NEW: Compliance uplift（輕於自傷，但保證不會過低）
COMPLIANCE_UPLIFT = {"thr": 3.0, "floor": 0.35, "add": 0.08, "cap": 0.75}

# Overlay safety controls（更靈敏）
OVERLAY_SCALE = float(os.getenv("OVERLAY_SCALE", "0.90"))
DELTA_CLIP   = float(os.getenv("OVERLAY_DELTA_CLIP", "1.20"))
TEMP         = float(os.getenv("OVERLAY_TEMP", "1.20"))

# ====== options ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== schema ======
TEMPLATE_COLUMNS = [
    "age","length_of_stay","num_previous_admissions",
    "medication_compliance_score","family_support_score","post_discharge_followups",
    "gender_Male","gender_Female",
] + [f"diagnosis_{d}" for d in DIAG_LIST] + [
    "has_recent_self_harm_Yes","has_recent_self_harm_No",
    "self_harm_during_admission_Yes","self_harm_during_admission_No",
]

# ====== null-safe defaults ======
DEFAULTS = {
    "age": 40.0,
    "length_of_stay": 5.0,
    "num_previous_admissions": 0.0,
    "medication_compliance_score": 5.0,
    "family_support_score": 5.0,
    "post_discharge_followups": 0.0,
}
NUMERIC_KEYS = list(DEFAULTS.keys())

def _num_or_default(x, key):
    try:
        v = float(x)
    except Exception:
        v = np.nan
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

# ====== overlay (log-odds) weights — strengthened compliance effect ======
POLICY = {
    # core
    "per_prev_admission": 0.18,
    "per_point_low_support": 0.20,
    "per_followup": -0.15,
    "no_followup_extra": 0.25,

    # LOS
    "los_short": 0.45, "los_mid": 0.00, "los_mid_high": 0.15, "los_long": 0.35,

    # age
    "age_young": 0.10, "age_old": 0.10,

    # diagnoses
    "diag": {
        "Personality Disorder":    0.35,
        "Substance Use Disorder":  0.35,
        "Bipolar":                 0.10,
        "PTSD":                    0.10,
        "Schizophrenia":           0.10,
        "Depression":              0.05,
        "Anxiety":                 0.00,
        "OCD":                     0.00,
        "Dementia":                0.00,
        "ADHD":                    0.00,
        "Other/Unknown":           0.00,
    },

    # >>> compliance (強化) <<<
    "per_point_low_compliance": 0.22,  # 線性：以 5 為中心
    "low_comp_boost_leq3": 0.75,       # ≤3 額外強 boost
    "low_comp_boost_leq1": 0.45,       # ≤1 再加一層
    "x_sud_lowcomp": 0.45,             # SUD × very low compliance
    "low_comp_no_followup": 0.35,      # very low × 無追蹤
    "per_point_high_compliance_protect": -0.06,  # ≥8 輕度保護/點
    "x_pd_shortlos": 0.10,
}

# ====== model load / demo ======
def try_load_model(path="dropout_model.pkl"):
    if not os.path.exists(path): return None
    try:
        bundle = joblib.load(path)
        return bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    except Exception:
        return None

def xgboost_classifier():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=450, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
    )

def train_demo_model(columns):
    import xgboost as xgb
    rng = np.random.default_rng(42)
    n = 9000
    X = pd.DataFrame(0, index=range(n), columns=columns, dtype=np.float32)
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(5.0, 3.0, n).clip(0, 45)
    X["num_previous_admissions"] = rng.poisson(0.8, n).clip(0, 12)
    X["medication_compliance_score"] = rng.normal(6.0, 2.5, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 6, n)
    # gender
    idx_gender = rng.integers(0, len(GENDER_LIST), n)
    for i, g in enumerate(GENDER_LIST): X.loc[idx_gender == i, f"gender_{g}"] = 1
    # diagnoses
    idx_primary = rng.integers(0, len(DIAG_LIST), n)
    for i, d in enumerate(DIAG_LIST): X.loc[idx_primary == i, f"diagnosis_{d}"] = 1
    extra_probs = {"Substance Use Disorder": 0.20, "Anxiety": 0.20, "Depression": 0.25, "PTSD": 0.10}
    for d, pr in extra_probs.items(): X.loc[rng.random(n) < pr, f"diagnosis_{d}"] = 1
    has_any = X[[f"diagnosis_{d}" for d in DIAG_LIST]].sum(axis=1) > 0
    if not has_any.all():
        fix_idx = has_any[~has_any].index
        rp = rng.integers(0, len(DIAG_LIST), len(fix_idx))
        for j, ridx in enumerate(fix_idx): X.at[ridx, f"diagnosis_{DIAG_LIST[rp[j]]}"] = 1
    # self-harm flags
    i1, i2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    X.loc[i1 == 1, "has_recent_self_harm_Yes"] = 1; X.loc[i1 == 0, "has_recent_self_harm_No"] = 1
    X.loc[i2 == 1, "self_harm_during_admission_Yes"] = 1; X.loc[i2 == 0, "self_harm_during_admission_No"] = 1

    # target gen
    beta0 = -0.60
    beta = {
        "has_recent_self_harm_Yes": 0.80,
        "self_harm_during_admission_Yes": 0.60,
        "prev_adm_ge2": 0.60,
        "medication_compliance_per_point": -0.25,
        "family_support_per_point": -0.20,
        "followups_per_visit": -0.12,
        "length_of_stay_per_day": 0.05,
    }
    beta_diag = {
        "Personality Disorder":    0.35,
        "Substance Use Disorder":  0.35,
        "Bipolar":                 0.10,
        "PTSD":                    0.10,
        "Schizophrenia":           0.10,
        "Depression":              0.05,
        "Anxiety":                 0.00,
        "OCD":                     0.00,
        "Dementia":                0.00,
        "ADHD":                    0.00,
        "Other/Unknown":           0.00,
    }
    prev_ge2 = (X["num_previous_admissions"] >= 2).astype(np.float32)
    logit = (beta0
             + beta["has_recent_self_harm_Yes"]       * X["has_recent_self_harm_Yes"]
             + beta["self_harm_during_admission_Yes"] * X["self_harm_during_admission_Yes"]
             + beta["prev_adm_ge2"]                   * prev_ge2
             + beta["medication_compliance_per_point"]* X["medication_compliance_score"]
             + beta["family_support_per_point"]       * X["family_support_score"]
             + beta["followups_per_visit"]            * X["post_discharge_followups"]
             + beta["length_of_stay_per_day"]         * X["length_of_stay"])
    for d, w in beta_diag.items(): logit = logit + w * X[f"diagnosis_{d}"]
    noise = rng.normal(0.0, 0.35, n).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(logit + noise)))
    y = (rng.random(n) < p).astype(np.int32)

    model = xgboost_classifier()
    model.fit(X, y)
    return model

def get_feat_names(m):
    try:
        b = m.get_booster()
        if getattr(b, "feature_names", None): return list(b.feature_names)
    except Exception: pass
    if hasattr(m, "feature_names_in_"): return list(m.feature_names_in_)
    return None

model = try_load_model()
use_demo = True
if model is not None:
    names = get_feat_names(model)
    if (names is not None) and (abs(len(names) - len(TEMPLATE_COLUMNS)) <= 3):
        use_demo = False
if use_demo:
    model = train_demo_model(TEMPLATE_COLUMNS)
    model_source = "demo (balanced weights, multi-diagnoses)"
else:
    model_source = "loaded from dropout_model.pkl"

# ====== alignment ======
def get_model_feature_order(m):
    order = None; exp_len = None
    try:
        booster = getattr(m, "get_booster", lambda: None)()
        if booster is not None:
            names = getattr(booster, "feature_names", None)
            if names: order, exp_len = list(names), len(names)
    except Exception: pass
    if order is None:
        if hasattr(m, "feature_names_in_"):
            order = list(m.feature_names_in_); exp_len = len(order)
        elif hasattr(m, "n_features_in_"):
            exp_len = int(m.n_features_in_)
    return order, exp_len

def align_df_to_model(df: pd.DataFrame, m):
    names, exp_len = get_model_feature_order(m)
    if names:
        aligned = pd.DataFrame(0, index=df.index, columns=names, dtype=np.float32)
        inter = [c for c in names if c in df.columns]
        aligned.loc[:, inter] = df[inter].astype(np.float32).values
        return aligned, names
    out = df.astype(np.float32)
    if (exp_len is not None) and (out.shape[1] != exp_len):
        if out.shape[1] > exp_len: out = out.iloc[:, :exp_len]
        else:
            add = exp_len - out.shape[1]
            pad = pd.DataFrame(0, index=out.index, columns=[f"_pad_{i}" for i in range(add)], dtype=np.float32)
            out = pd.concat([out, pad], axis=1)
    return out, list(out.columns)

def to_float32_np(df: pd.DataFrame): return df.astype(np.float32).values

# ====== helpers ======
def set_onehot_by_prefix(df, prefix, value):
    col = f"{prefix}_{value}"
    if col in df.columns: df.at[0, col] = 1

def set_onehot_by_prefix_multi(df, prefix, values):
    for v in values:
        col = f"{prefix}_{v}"
        if col in df.columns: df.at[0, col] = 1

def flag_yes(row, prefix):
    col = f"{prefix}_Yes"; return (col in row.index) and (row[col] == 1)

# ====== thresholds ======
MOD_CUT = 20
HIGH_CUT = 40
def proba_to_percent(p): return float(p) * 100
def proba_to_score(p): return int(round(proba_to_percent(p)))
def classify_soft(score, mod=MOD_CUT, high=HIGH_CUT, band=BORDER_BAND):
    if score >= high + band: return "High"
    if score >= high - band: return "Moderate–High"
    if score >= mod + band:  return "Moderate"
    if score >= mod - band:  return "Low–Moderate"
    return "Low"

# ====== sidebar ======
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 90, 35)
    gender = st.selectbox("Gender", GENDER_LIST)
    diagnoses = st.multiselect("Diagnoses (multi-select)", DIAG_LIST, default=[])
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("Previous Admissions (1y)", 0, 15, 1)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0)
    recent_self_harm = st.radio("Recent Self-harm", BIN_YESNO, index=1)
    selfharm_adm = st.radio("Self-harm During Admission", BIN_YESNO, index=1)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0)
    followups = st.slider("Post-discharge Followups", 0, 10, 2)

    with st.expander("Advanced calibration", expanded=False):
        cal_shift = st.slider("Calibration (log-odds, global)", -1.0, 1.0, CAL_LOGIT_SHIFT, 0.05)
        blend_w  = st.slider("Blend weight (Final = (1-BLEND)*Model + BLEND*Overlay)", 0.0, 1.0, BLEND_W, 0.05)
        overlay_scale = st.slider("Overlay scale (shrink policy effect)", 0.0, 1.0, OVERLAY_SCALE, 0.05)
        delta_clip = st.slider("Overlay delta clip (|log-odds| cap)", 0.0, 2.0, DELTA_CLIP, 0.05)
        temp_val = st.slider("Temperature (>1 = softer probs)", 0.5, 3.0, TEMP, 0.05)
    CAL_LOGIT_SHIFT, BLEND_W, OVERLAY_SCALE, DELTA_CLIP, TEMP = cal_shift, blend_w, overlay_scale, delta_clip, temp_val

# ====== single-row DF ======
X_final = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
for k, v in {
    "age": age, "length_of_stay": float(length_of_stay), "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance), "family_support_score": float(support),
    "post_discharge_followups": int(followups),
}.items(): X_final.at[0, k] = v
set_onehot_by_prefix(X_final, "gender", gender)
set_onehot_by_prefix_multi(X_final, "diagnosis", diagnoses)
set_onehot_by_prefix(X_final, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_final, "self_harm_during_admission", selfharm_adm)
fill_defaults_single_row(X_final)

# ====== predict + overlay ======
X_aligned_df, used_names = align_df_to_model(X_final, model)
p_model = float(model.predict_proba(to_float32_np(X_aligned_df), validate_features=False)[:, 1][0])

drivers = []
def add_driver(label, val):
    if val != 0: drivers.append((label, float(val)))
    return val

base_logit = _logit(p_model)
lz_pol = base_logit

row0 = X_final.iloc[0]
adm = _num_or_default(row0["num_previous_admissions"], "num_previous_admissions")
comp = _num_or_default(row0["medication_compliance_score"], "medication_compliance_score")
sup  = _num_or_default(row0["family_support_score"], "family_support_score")
fup  = _num_or_default(row0["post_discharge_followups"], "post_discharge_followups")
los  = _num_or_default(row0["length_of_stay"], "length_of_stay")
agev = _num_or_default(row0["age"], "age")

# admissions
lz_pol += add_driver("More previous admissions", POLICY["per_prev_admission"] * min(int(adm), 5))
# compliance (linear + thresholds + interactions)
lz_pol += add_driver("Low medication compliance (linear)", POLICY["per_point_low_compliance"] * max(0.0, 5.0 - comp))
if comp <= 3:
    lz_pol += add_driver("Very low compliance (≤3)", POLICY["low_comp_boost_leq3"])
    if fup == 0: lz_pol += add_driver("Very low compliance × no follow-up", POLICY["low_comp_no_followup"])
    if X_final.at[0, "diagnosis_Substance Use Disorder"] == 1:
        lz_pol += add_driver("SUD × very low compliance", POLICY["x_sud_lowcomp"])
if comp <= 1:
    lz_pol += add_driver("Extremely low compliance (≤1)", POLICY["low_comp_boost_leq1"])
if comp >= 8:
    lz_pol += add_driver("High compliance (protective)", POLICY["per_point_high_compliance_protect"] * (comp - 7.0))

# family support + followup
lz_pol += add_driver("Low family support", POLICY["per_point_low_support"] * max(0.0, 5.0 - sup))
lz_pol += add_driver("More post-discharge followups (protective)", POLICY["per_followup"] * fup)
if fup == 0: lz_pol += add_driver("No follow-up scheduled", POLICY["no_followup_extra"])

# LOS
if los < 3: lz_pol += add_driver("Very short stay (<3d)", POLICY["los_short"])
elif los <= 14: lz_pol += add_driver("Typical stay (3–14d)", POLICY["los_mid"])
elif los <= 21: lz_pol += add_driver("Longish stay (15–21d)", POLICY["los_mid_high"])
else: lz_pol += add_driver("Very long stay (>21d)", POLICY["los_long"])

# age
if agev < 21: lz_pol += add_driver("Young age (<21)", POLICY["age_young"])
elif agev >= 75: lz_pol += add_driver("Older age (≥75)", POLICY["age_old"])

# diagnoses
for dx, w in POLICY["diag"].items():
    col = f"diagnosis_{dx}"
    if col in X_final.columns and X_final.at[0, col] == 1:
        lz_pol += add_driver(f"Diagnosis: {dx}", w)

# PD × short LOS
if (X_final.at[0, "diagnosis_Personality Disorder"] == 1) and (los < 3):
    lz_pol += add_driver("PD × very short stay", POLICY["x_pd_shortlos"])

# scale/clip/temp + calibration
delta = lz_pol - base_logit
delta = OVERLAY_SCALE * delta
delta = np.clip(delta, -DELTA_CLIP, DELTA_CLIP)
lz = base_logit + delta
lz += CAL_LOGIT_SHIFT
p_overlay = _sigmoid(lz / TEMP)

# blend
p_policy = (1.0 - BLEND_W) * p_model + BLEND_W * p_overlay

# ====== uplifts (priority: self-harm > compliance) ======
soft_reason = None
if flag_yes(X_final.iloc[0], "has_recent_self_harm") or flag_yes(X_final.iloc[0], "self_harm_during_admission"):
    p_final = min(max(p_policy, SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"], SOFT_UPLIFT["cap"])
    soft_reason = "self-harm uplift"
else:
    if comp <= COMPLIANCE_UPLIFT["thr"]:
        p_final = min(max(p_policy, COMPLIANCE_UPLIFT["floor"]) + COMPLIANCE_UPLIFT["add"], COMPLIANCE_UPLIFT["cap"])
        soft_reason = "compliance uplift (≤3)"
    else:
        p_final = p_policy

percent_model = proba_to_percent(p_model)
percent = proba_to_percent(p_final)
score = proba_to_score(p_final)
level = classify_soft(score)

# ====== diagnostics ======
with st.expander("Model diagnostics", expanded=False):
    st.write(f"**Model source:** {model_source}")
    try:
        booster = model.get_booster()
        fmap = booster.get_fscore()
        imp = (pd.Series(fmap, name="split_count")
               .reindex(get_feat_names(model) or [], fill_value=0)
               .sort_values(ascending=False).head(10))
        st.caption("Top-10 features by split count (proxy for importance):")
        st.dataframe(imp.reset_index(names="feature"))
    except Exception as e:
        st.caption(f"Importance not available: {e}")

# ====== result ======
st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Model Probability", f"{percent_model:.1f}%")
with c2: st.metric("Final Probability", f"{percent:.1f}%")
with c3: st.metric("Risk Score (0–100)", f"{score}")

if soft_reason:
    st.warning(f"🟠 Uplift applied ({soft_reason}).")
else:
    if level == "High": st.error("🔴 High Risk")
    elif level == "Moderate–High": st.warning("🟠 Moderate–High (borderline to High)")
    elif level == "Moderate": st.warning("🟡 Moderate Risk")
    elif level == "Low–Moderate": st.info("🔵 Low–Moderate (borderline to Moderate)")
    else: st.success("🟢 Low Risk")

# ====== policy drivers ======
with st.expander("Why did the risk change? (policy drivers)", expanded=False):
    if drivers:
        df_drv = pd.DataFrame(
            [{"driver": k, "log-odds + (pre-scale)": round(v, 3)} for k, v in sorted(drivers, key=lambda x: abs(x[1]), reverse=True)]
        )
        st.dataframe(df_drv, use_container_width=True)
        st.caption(f"Overlay controls — scale={OVERLAY_SCALE}, clip=±{DELTA_CLIP}, temp={TEMP}.")
    else:
        st.caption("No strong drivers; Final ~ Model.")

# ====== SHAP (model-only) ======
with st.expander("SHAP Explanation (model component)", expanded=True):
    st.caption("Positive bars increase risk; negatives decrease it. Only active one-hots are shown.")
    import xgboost as xgb
    try:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_aligned_df, feature_names=list(X_aligned_df.columns))
        contribs = booster.predict(dmat, pred_contribs=True, validate_features=False)
        contrib = np.asarray(contribs)[0]
        base_value = float(contrib[-1]); feat_contrib = contrib[:-1]
        sv_map = dict(zip(list(X_aligned_df.columns), feat_contrib))
    except Exception:
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_aligned_df)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
            base_value = base_value[0]
            if isinstance(sv_raw, list): sv_raw = sv_raw[0]
        sv_map = dict(zip(list(X_aligned_df.columns), sv_raw[0]))

    names, vals, data_vals = [], [], []
    cont_feats = [
        ("Age","age", X_final.at[0,"age"]),
        ("Length of Stay (days)","length_of_stay", X_final.at[0,"length_of_stay"]),
        ("Previous Admissions (1y)","num_previous_admissions", X_final.at[0,"num_previous_admissions"]),
        ("Medication Compliance (0–10)","medication_compliance_score", X_final.at[0,"medication_compliance_score"]),
        ("Family Support (0–10)","family_support_score", X_final.at[0,"family_support_score"]),
        ("Post-discharge Followups","post_discharge_followups", X_final.at[0,"post_discharge_followups"]),
    ]
    for label, key, dv in cont_feats:
        if key in sv_map: names.append(label); vals.append(float(sv_map[key])); data_vals.append(dv)

    def add_onehot(title, prefix, value):
        col = f"{prefix}_{value}"
        if col in sv_map:
            names.append(f"{title}={value}"); vals.append(float(sv_map[col])); data_vals.append(1)

    for dx in diagnoses: add_onehot("Diagnosis","diagnosis", dx)
    add_onehot("Gender","gender", gender)
    add_onehot("Recent Self-harm","has_recent_self_harm", recent_self_harm)
    add_onehot("Self-harm During Admission","self_harm_during_admission", selfharm_adm)

    if len(vals) > 0:
        order = np.argsort(np.abs(np.array(vals)))[::-1][:12]
        exp = shap.Explanation(values=np.array(vals)[order], base_values=base_value,
                               feature_names=[names[i] for i in order],
                               data=np.array(data_vals, dtype=float)[order])
        shap.plots.waterfall(exp, show=False, max_display=12)
        st.pyplot(plt.gcf(), clear_figure=True)
    else:
        st.caption("No SHAP contributions available.")

# ====== actions（baseline + personalized） ======
st.subheader("Recommended Actions")

BASE_ACTIONS = {
    "High": [
        ("Today","Clinician","Crisis/safety planning; 24/7 crisis contacts","High risk"),
        ("Today","Clinic scheduler","Return within 7 days (prefer 72h)","Early follow-up"),
        ("Today","Care coordinator","Warm handoff to case management","Coordination"),
        ("48h","Nurse","Outreach call: symptoms/side-effects/barriers","Early outreach"),
        ("7d","Pharmacist/Nurse","Medication review + adherence aids","Adherence support"),
        ("1–2w","Social worker","SDOH screen; transport/financial aid","Practical barriers"),
        ("1–4w","Peer support","Enroll in peer/skills group","Engagement"),
    ],
    "Moderate": [
        ("1–2w","Clinic scheduler","Book within 14 days; SMS reminders","Timely follow-up"),
        ("1–2w","Nurse","Barrier check & solutions","Remove drivers"),
        ("2–4w","Clinician","Brief MI/BA/psychoeducation; 4-week plan","Motivation/structure"),
    ],
    "Low": [
        ("2–4w","Clinic scheduler","Routine follow-up; confirm reminders","Maintain engagement"),
        ("2–4w","Nurse","Education/self-management resources","Support autonomy"),
    ],
}

def _normalize_action_tuple(a):
    if len(a) == 4: return a
    if len(a) == 3: return (a[0], a[1], a[2], "")
    return None

def add_action(arr, tl, owner, act, why): arr.append((tl, owner, act, why))

def personalized_actions(row: pd.Series, chosen_dx: list, final_level: str):
    acts = []
    agev = _num_or_default(row["age"], "age")
    los  = _num_or_default(row["length_of_stay"], "length_of_stay")
    adm  = _num_or_default(row["num_previous_admissions"], "num_previous_admissions")
    comp = _num_or_default(row["medication_compliance_score"], "medication_compliance_score")
    sup  = _num_or_default(row["family_support_score"], "family_support_score")
    fup  = _num_or_default(row["post_discharge_followups"], "post_discharge_followups")
    has_selfharm = flag_yes(row, "has_recent_self_harm") or flag_yes(row, "self_harm_during_admission")
    has_sud = ("Substance Use Disorder" in chosen_dx)
    has_pd  = ("Personality Disorder" in chosen_dx)
    has_dep = ("Depression" in chosen_dx)
    has_scz = ("Schizophrenia" in chosen_dx)

    if has_selfharm:
        add_action(acts, "Today","Clinician","C-SSRS; update safety plan; lethal-means counseling","Self-harm")
        add_action(acts, "48h","Nurse","Safety check-in call","Post-discharge safety")

    if has_sud and comp <= 3:
        add_action(acts, "1–7d","Clinician","Brief MI focused on use goals","SUD × very low adherence")
        add_action(acts, "1–7d","Care coordinator","Refer to SUD program/IOP or CM","Retention")
        add_action(acts, "Today","Clinician","Overdose prevention education/resources","Harm reduction")

    if has_pd and los < 3:
        add_action(acts, "Today","Care coordinator","Same-day DBT/skills intake","PD × short stay")
        add_action(acts, "48h","Peer support","Proactive outreach + skills workbook","Engagement")

    if comp <= 3:
        add_action(acts, "7d","Pharmacist","Simplify regimen + blister/pillbox + reminders","Very low adherence")
        add_action(acts, "1–2w","Clinician","Consider LAI/long-acting formulations if appropriate","Stabilize adherence")

    if sup <= 2:
        add_action(acts, "1–2w","Clinician","Family meeting / caregiver engagement","Low support")
        add_action(acts, "1–2w","Social worker","Community supports; transport/financial counseling","SDOH")

    if fup == 0:
        add_action(acts, "Today","Clinic scheduler","Book 2 touchpoints in first 14 days (day2/day7)","No follow-up")

    if adm >= 3:
        add_action(acts, "1–2w","Care coordinator","Case management weekly for 1 month","Revolving door")

    if los < 3:
        add_action(acts, "48h","Nurse","Early call; review meds/barriers","Very short LOS")
    elif los > 21:
        add_action(acts, "1–7d","Care coordinator","Step-down/day program plan + warm handoff","Long LOS")

    if agev < 21:
        add_action(acts, "1–2w","Clinician","Involve guardians; link school/university counseling","Young age")
    elif agev >= 75:
        add_action(acts, "1–2w","Nurse/Pharmacist","Med reconciliation; simplify dosing; assess needs","Older age")

    if has_dep:
        add_action(acts, "1–2w","Clinician","Behavioral activation + activity schedule","Depression")
    if has_scz:
        add_action(acts, "1–4w","Clinician","Relapse plan; early warning signs; caregiver involvement","Schizophrenia")

    return acts

bucket = {"High":"High","Moderate–High":"High","Moderate":"Moderate","Low–Moderate":"Low","Low":"Low"}
actions = []
for a in BASE_ACTIONS[bucket[level]]:
    na = _normalize_action_tuple(a)
    if na is not None: actions.append(na)

pers = personalized_actions(X_final.iloc[0], diagnoses, level)
for a in pers:
    na = _normalize_action_tuple(a)
    if na is not None: actions.append(na)

seen = set(); uniq = []
for tl, ow, ac, why in actions:
    key = (tl, ow, ac)
    if key not in seen:
        seen.add(key); uniq.append((tl, ow, ac, why))

ORDER = {"Today":0,"48h":1,"7d":2,"1–7d":2,"1–2w":3,"2–4w":4,"1–4w":5}
uniq.sort(key=lambda x: (ORDER.get(x[0], 99), x[1], x[2]))
st.dataframe(pd.DataFrame(uniq, columns=["Timeline","Owner","Action","Why"]), use_container_width=True)

# SOP export
if level in ["High","Moderate–High"]:
    def make_sop_txt(score: int, label: str, actions: list) -> BytesIO:
        lines = ["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}", ""]
        for (tl, ow, ac, why) in actions:
            lines.append(f"- {tl} | {ow} | {ac}" + (f" (Why: {why})" if why else ""))
        buf = BytesIO("\n".join(lines).encode("utf-8")); buf.seek(0); return buf
    st.download_button("⬇️ Export SOP (TXT)", make_sop_txt(score, level, uniq),
                       file_name="dropout_risk_SOP.txt", mime="text/plain")
    if HAS_DOCX:
        def make_sop_docx(score: int, label: str, actions: list) -> BytesIO:
            doc = Document()
            doc.add_heading('Psychiatric Dropout Risk – SOP', level=1)
            doc.add_paragraph(f"Risk score: {score}/100 | Risk level: {label}")
            t = doc.add_table(rows=1, cols=4); hdr = t.rows[0].cells
            hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = 'Timeline','Owner','Action','Why'
            for (tl, ow, ac, why) in actions:
                r = t.add_row().cells; r[0].text, r[1].text, r[2].text, r[3].text = tl, ow, ac, (why or "")
            buf = BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== shared vectorized overlay (Batch & Validation) ======
def overlay_blend_vectorized(df_feat: pd.DataFrame, base_probs: np.ndarray):
    base = _logit_vec(base_probs); lz_pol = base.copy()

    adm = pd.to_numeric(df_feat["num_previous_admissions"], errors="coerce").fillna(DEFAULTS["num_previous_admissions"]).to_numpy()
    comp = pd.to_numeric(df_feat["medication_compliance_score"], errors="coerce").fillna(DEFAULTS["medication_compliance_score"]).to_numpy()
    sup  = pd.to_numeric(df_feat["family_support_score"], errors="coerce").fillna(DEFAULTS["family_support_score"]).to_numpy()
    fup  = pd.to_numeric(df_feat["post_discharge_followups"], errors="coerce").fillna(DEFAULTS["post_discharge_followups"]).to_numpy()
    los  = pd.to_numeric(df_feat["length_of_stay"], errors="coerce").fillna(DEFAULTS["length_of_stay"]).to_numpy()
    agev = pd.to_numeric(df_feat["age"], errors="coerce").fillna(DEFAULTS["age"]).to_numpy()

    lz_pol += POLICY["per_prev_admission"] * np.minimum(adm, 5)

    # compliance
    lz_pol += POLICY["per_point_low_compliance"] * np.maximum(0.0, 5.0 - comp)
    low3 = (comp <= 3); low1 = (comp <= 1)
    lz_pol += POLICY["low_comp_boost_leq3"] * low3
    lz_pol += POLICY["low_comp_boost_leq1"] * low1
    # interactions
    sud = (df_feat.get("diagnosis_Substance Use Disorder", 0).to_numpy() == 1)
    lz_pol += POLICY["x_sud_lowcomp"] * (sud & low3)
    lz_pol += POLICY["low_comp_no_followup"] * (low3 & (fup == 0))
    # high compliance protection
    lz_pol += POLICY["per_point_high_compliance_protect"] * np.maximum(0.0, comp - 7.0)

    # support + followups
    lz_pol += POLICY["per_point_low_support"] * np.maximum(0.0, 5.0 - sup)
    lz_pol += POLICY["per_followup"] * fup
    lz_pol += POLICY["no_followup_extra"] * (fup == 0)

    # LOS
    lz_pol += np.where(los < 3, POLICY["los_short"],
                np.where(los <= 14, POLICY["los_mid"],
                np.where(los <= 21, POLICY["los_mid_high"], POLICY["los_long"])))

    # age
    lz_pol += POLICY["age_young"] * (agev < 21)
    lz_pol += POLICY["age_old"]   * (agev >= 75)

    # diagnoses
    diag_term = np.zeros(len(df_feat), dtype=float)
    for dx, w in POLICY["diag"].items():
        col = f"diagnosis_{dx}"
        if col in df_feat.columns:
            diag_term += w * (df_feat[col].to_numpy() == 1)
    lz_pol += diag_term

    # PD × short LOS
    pd_mask = (df_feat.get("diagnosis_Personality Disorder", 0).to_numpy() == 1)
    lz_pol += POLICY["x_pd_shortlos"] * (pd_mask & (los < 3))

    # scale/clip/temp + calibration
    delta = np.clip(OVERLAY_SCALE * (lz_pol - base), -DELTA_CLIP, DELTA_CLIP)
    lz = base + delta + CAL_LOGIT_SHIFT
    p_overlay = 1.0 / (1.0 + np.exp(-(lz / TEMP)))

    # blend
    p_policy = (1.0 - BLEND_W) * base_probs + BLEND_W * p_overlay

    # uplifts：先 self-harm，後 compliance（避免覆蓋更嚴重事件）
    hrsh = df_feat.get("has_recent_self_harm_Yes", 0)
    shadm = df_feat.get("self_harm_during_admission_Yes", 0)
    mask_harm = ((np.array(hrsh) == 1) | (np.array(shadm) == 1))
    out = p_policy.copy()
    out[mask_harm] = np.minimum(np.maximum(out[mask_harm], SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"], SOFT_UPLIFT["cap"])

    mask_comp = (comp <= COMPLIANCE_UPLIFT["thr"]) & (~mask_harm)
    out[mask_comp] = np.minimum(np.maximum(out[mask_comp], COMPLIANCE_UPLIFT["floor"]) + COMPLIANCE_UPLIFT["add"], COMPLIANCE_UPLIFT["cap"])
    return out

# ====== Batch ======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

friendly_cols = [
    "Age","Gender","Diagnoses",
    "Length of Stay (days)","Previous Admissions (1y)",
    "Medication Compliance Score (0–10)",
    "Recent Self-harm","Self-harm During Admission",
    "Family Support Score (0–10)","Post-discharge Followups"
]
tpl_df = pd.DataFrame(columns=friendly_cols)
buf_tpl = BytesIO(); tpl_df.to_excel(buf_tpl, index=False); buf_tpl.seek(0)
st.download_button("📥 Download Excel Template", buf_tpl, file_name="batch_template.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])
if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = pd.DataFrame(0, index=raw.index, columns=TEMPLATE_COLUMNS, dtype=float)

        def safe_get(col, default=0): return raw[col] if col in raw.columns else default
        df["age"] = safe_get("Age")
        df["length_of_stay"] = safe_get("Length of Stay (days)")
        df["num_previous_admissions"] = safe_get("Previous Admissions (1y)")
        df["medication_compliance_score"] = safe_get("Medication Compliance Score (0–10)")
        df["family_support_score"] = safe_get("Family Support Score (0–10)")
        df["post_discharge_followups"] = safe_get("Post-discharge Followups")

        def apply_onehot_prefix_multi(human_col, prefix, options):
            if human_col not in raw.columns: return
            for i, cell in raw[human_col].astype(str).fillna("").items():
                parts = [p.strip() for p in re.split(r"[;,/|]", cell) if p.strip()]
                if not parts and cell.strip(): parts = [cell.strip()]
                for v in parts:
                    col = f"{prefix}_{v}"
                    if v in options and col in df.columns: df.at[i, col] = 1

        def apply_onehot_prefix(human_col, prefix, options):
            if human_col not in raw.columns: return
            for i, v in raw[human_col].astype(str).str.strip().items():
                col = f"{prefix}_{v}"
                if v in options and col in df.columns: df.at[i, col] = 1

        apply_onehot_prefix("Gender","gender", GENDER_LIST)
        if "Diagnoses" in raw.columns: apply_onehot_prefix_multi("Diagnoses","diagnosis", DIAG_LIST)
        elif "Diagnosis" in raw.columns: apply_onehot_prefix_multi("Diagnosis","diagnosis", DIAG_LIST)
        apply_onehot_prefix("Recent Self-harm","has_recent_self_harm", BIN_YESNO)
        apply_onehot_prefix("Self-harm During Admission","self_harm_during_admission", BIN_YESNO)

        fill_defaults_batch(df)

        Xb_aligned, _ = align_df_to_model(df, model)
        base_probs = model.predict_proba(to_float32_np(Xb_aligned), validate_features=False)[:, 1]
        adj_probs = overlay_blend_vectorized(df, base_probs)

        out = raw.copy()
        out["risk_percent"] = (adj_probs * 100).round(1)
        out["risk_score_0_100"] = (adj_probs * 100).round().astype(int)

        s = out["risk_score_0_100"].to_numpy()
        levels = np.full(s.shape, "Low", dtype=object)
        levels[s >= MOD_CUT - BORDER_BAND] = "Low–Moderate"
        levels[s >= MOD_CUT + BORDER_BAND] = "Moderate"
        levels[s >= HIGH_CUT - BORDER_BAND] = "Moderate–High"
        levels[s >= HIGH_CUT + BORDER_BAND] = "High"
        out["risk_level"] = levels

        # drivers top3
        top3_list = []
        for i in range(len(df)):
            contribs = []
            def push(name, val):
                if val != 0: contribs.append((name, float(val)))
            prev_i = float(df.iloc[i]["num_previous_admissions"]); push("More previous admissions", POLICY["per_prev_admission"]*min(prev_i,5))
            comp_i = float(df.iloc[i]["medication_compliance_score"])
            push("Low medication compliance (linear)", POLICY["per_point_low_compliance"] * max(0.0, 5.0 - comp_i))
            if comp_i <= 3:
                push("Very low compliance (≤3)", POLICY["low_comp_boost_leq3"])
                if float(df.iloc[i]["post_discharge_followups"]) == 0:
                    push("Very low compliance × no follow-up", POLICY["low_comp_no_followup"])
            if comp_i <= 1: push("Extremely low compliance (≤1)", POLICY["low_comp_boost_leq1"])
            if comp_i >= 8: push("High compliance (protective)", POLICY["per_point_high_compliance_protect"] * (comp_i - 7.0))
            sup_i = float(df.iloc[i]["family_support_score"]); push("Low family support", POLICY["per_point_low_support"]*max(0.0,5.0-sup_i))
            fu_i = float(df.iloc[i]["post_discharge_followups"]); push("More post-discharge followups (protective)", POLICY["per_followup"]*fu_i)
            if fu_i == 0: push("No follow-up scheduled", POLICY["no_followup_extra"])
            los_i = float(df.iloc[i]["length_of_stay"])
            if los_i < 3: push("Very short stay (<3d)", POLICY["los_short"])
            elif los_i <= 14: push("Typical stay (3–14d)", POLICY["los_mid"])
            elif los_i <= 21: push("Longish stay (15–21d)", POLICY["los_mid_high"])
            else: push("Very long stay (>21d)", POLICY["los_long"])
            age_i = float(df.iloc[i]["age"])
            if age_i < 21: push("Young age (<21)", POLICY["age_young"])
            elif age_i >= 75: push("Older age (≥75)", POLICY["age_old"])
            for dx, w in POLICY["diag"].items():
                if f"diagnosis_{dx}" in df.columns and df.iloc[i][f"diagnosis_{dx}"] == 1: push(f"Diagnosis: {dx}", w)
            if (df.iloc[i].get("diagnosis_Substance Use Disorder",0)==1) and (comp_i<=3): push("SUD × very low compliance", POLICY["x_sud_lowcomp"])
            if (df.iloc[i].get("diagnosis_Personality Disorder",0)==1) and (los_i<3): push("PD × very short stay", POLICY["x_pd_shortlos"])
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            top3_list.append(" | ".join([f"{n} ({v:+.2f})" for n,v in contribs[:3]]) if contribs else "")
        out["policy_drivers_top3"] = top3_list

        # Top-3 actions
        def chosen_dx_for_row(i, df_feat):
            return [d for d in DIAG_LIST if f"diagnosis_{d}" in df_feat.columns and df_feat.at[i, f"diagnosis_{d}"] == 1]
        def top3_actions_for_row(i, out_df, features_df):
            level_i = out_df.loc[i, "risk_level"]
            base_map = {"High":"High","Moderate–High":"High","Moderate":"Moderate","Low–Moderate":"Low","Low":"Low"}
            acts = [(_normalize_action_tuple(a)) for a in BASE_ACTIONS[base_map.get(level_i,"Low")]]
            acts = [a for a in acts if a is not None]
            row_series = features_df.iloc[i]; dxs = chosen_dx_for_row(i, features_df)
            acts += [a for a in personalized_actions(row_series, dxs, level_i) if a is not None]
            seen=set(); uniq_local=[]
            for a in acts:
                tl,ow,ac,why=a; key=(tl,ow,ac)
                if key not in seen: seen.add(key); uniq_local.append(a)
            ORDER = {"Today":0,"48h":1,"7d":2,"1–7d":2,"1–2w":3,"2–4w":4,"1–4w":5}
            uniq_local.sort(key=lambda x: (ORDER.get(x[0], 99), x[1], x[2]))
            return " || ".join([f"{tl} | {ow} | {ac}" + (f" (Why: {why})" if why else "") for (tl,ow,ac,why) in uniq_local[:3]])
        out["recommended_actions_top3"] = [ top3_actions_for_row(i, out, df) for i in range(len(out)) ]

        st.dataframe(out, use_container_width=True)
        buf_out = BytesIO(); out.to_csv(buf_out, index=False); buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

# ====== Validation (synthetic) ======
st.markdown("---")
st.header("✅ Validation (synthetic hold-out)")

def generate_synth_holdout(n=20000, seed=2024):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(0, index=range(n), columns=TEMPLATE_COLUMNS, dtype=float)
    df["age"] = rng.integers(16, 85, n)
    df["length_of_stay"] = rng.normal(5.0, 3.0, n).clip(0, 45)
    df["num_previous_admissions"] = rng.poisson(0.8, n).clip(0, 12)
    df["medication_compliance_score"] = rng.normal(6.0, 2.5, n).clip(0, 10)
    df["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    df["post_discharge_followups"] = rng.integers(0, 6, n)
    idx_gender = rng.integers(0, len(GENDER_LIST), n)
    for i, g in enumerate(GENDER_LIST): df.loc[idx_gender == i, f"gender_{g}"] = 1
    idx_primary = rng.integers(0, len(DIAG_LIST), n)
    for i, d in enumerate(DIAG_LIST): df.loc[idx_primary == i, f"diagnosis_{d}"] = 1
    extra_probs = {"Substance Use Disorder": 0.20, "Anxiety": 0.20, "Depression": 0.25, "PTSD": 0.10}
    for d, pr in extra_probs.items(): df.loc[rng.random(n) < pr, f"diagnosis_{d}"] = 1
    # ensure ≥1 dx
    diag_cols = [f"diagnosis_{d}" for d in DIAG_LIST]
    mask_none = (df[diag_cols].sum(axis=1) == 0)
    if mask_none.any():
        rp = rng.integers(0, len(DIAG_LIST), mask_none.sum())
        df.loc[mask_none, [f"diagnosis_{DIAG_LIST[i]}" for i in rp]] = 1
    # self-harm flags
    r1, r2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    df.loc[r1 == 1, "has_recent_self_harm_Yes"] = 1; df.loc[r1 == 0, "has_recent_self_harm_No"] = 1
    df.loc[r2 == 1, "self_harm_during_admission_Yes"] = 1; df.loc[r2 == 0, "self_harm_during_admission_No"] = 1

    # true labels generator (same as training)
    beta0 = -0.60
    beta = {
        "has_recent_self_harm_Yes": 0.80,
        "self_harm_during_admission_Yes": 0.60,
        "prev_adm_ge2": 0.60,
        "medication_compliance_per_point": -0.25,
        "family_support_per_point": -0.20,
        "followups_per_visit": -0.12,
        "length_of_stay_per_day": 0.05,
    }
    beta_diag = {
        "Personality Disorder":    0.35,
        "Substance Use Disorder":  0.35,
        "Bipolar":                 0.10,
        "PTSD":                    0.10,
        "Schizophrenia":           0.10,
        "Depression":              0.05,
        "Anxiety":                 0.00,
        "OCD":                     0.00,
        "Dementia":                0.00,
        "ADHD":                    0.00,
        "Other/Unknown":           0.00,
    }
    prev_ge2 = (df["num_previous_admissions"] >= 2).astype(np.float32)
    logit = (beta0
             + beta["has_recent_self_harm_Yes"]       * df["has_recent_self_harm_Yes"]
             + beta["self_harm_during_admission_Yes"] * df["self_harm_during_admission_Yes"]
             + beta["prev_adm_ge2"]                   * prev_ge2
             + beta["medication_compliance_per_point"]* df["medication_compliance_score"]
             + beta["family_support_per_point"]       * df["family_support_score"]
             + beta["followups_per_visit"]            * df["post_discharge_followups"]
             + beta["length_of_stay_per_day"]         * df["length_of_stay"])
    for d, w in beta_diag.items(): logit = logit + w * df[f"diagnosis_{d}"]
    rng2 = np.random.default_rng(seed+1)
    p_true = 1.0 / (1.0 + np.exp(-(logit + rng2.normal(0.0, 0.35, n).astype(np.float32))))
    y_true = (np.random.default_rng(seed+2).random(n) < p_true).astype(int)

    fill_defaults_batch(df)
    return df, y_true, p_true

# metrics utils
def _calibration_bins(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(prob, bins) - 1
    frac_pos, mean_pred, weights = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if m.sum() == 0:
            frac_pos.append(np.nan); mean_pred.append(np.nan); weights.append(0.0)
        else:
            frac_pos.append(y_true[m].mean()); mean_pred.append(prob[m].mean()); weights.append(m.mean())
    return np.array(mean_pred), np.array(frac_pos), np.array(weights)

def _ece(y_true, prob, n_bins=10):
    mp, fp, w = _calibration_bins(y_true, prob, n_bins)
    mask = ~np.isnan(mp) & ~np.isnan(fp)
    if mask.sum() == 0: return np.nan
    return float(np.sum(w[mask] * np.abs(fp[mask] - mp[mask])))

def _plot_roc_pr(y_true, p1, p2, label1="Model", label2="Final"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr1, tpr1, _ = roc_curve(y_true, p1); roc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(y_true, p2); roc2 = auc(fpr2, tpr2)
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr1, tpr1, label=f"{label1} AUC={roc1:.3f}")
    ax1.plot(fpr2, tpr2, label=f"{label2} AUC={roc2:.3f}")
    ax1.plot([0,1],[0,1], linestyle="--")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate"); ax1.set_title("ROC")
    ax1.legend(loc="lower right"); st.pyplot(fig1, clear_figure=True)

    p1_prec, p1_rec, _ = precision_recall_curve(y_true, p1); ap1 = average_precision_score(y_true, p1)
    p2_prec, p2_rec, _ = precision_recall_curve(y_true, p2); ap2 = average_precision_score(y_true, p2)
    fig2, ax2 = plt.subplots()
    ax2.plot(p1_rec, p1_prec, label=f"{label1} AP={ap1:.3f}")
    ax2.plot(p2_rec, p2_prec, label=f"{label2} AP={ap2:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision–Recall")
    ax2.legend(loc="upper right"); st.pyplot(fig2, clear_figure=True)

def _plot_calibration(y_true, p1, p2, n_bins=10, label1="Model", label2="Final"):
    mp1, fp1, _ = _calibration_bins(y_true, p1, n_bins)
    mp2, fp2, _ = _calibration_bins(y_true, p2, n_bins)
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--", label="Perfect")
    ax.plot(mp1, fp1, marker="o", label=label1)
    ax.plot(mp2, fp2, marker="o", label=label2)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Fraction of positives (bin)")
    ax.set_title("Calibration (Reliability) Diagram")
    ax.legend(loc="upper left"); st.pyplot(fig, clear_figure=True)

def _binary_confusion(y_true, prob, thr):
    from sklearn.metrics import confusion_matrix
    y_pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

cA, cB = st.columns([2,1])
with cA:
    n_val = st.slider("Number of synthetic patients", 5000, 50000, 20000, 5000)
    seed_val = st.number_input("Random seed", min_value=1, max_value=10**9, value=2024, step=1)
with cB:
    run_val = st.button("Run validation")

if run_val:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    except Exception as e:
        st.error(f"scikit-learn not available: {e}. Please `pip install scikit-learn`.")
    else:
        with st.spinner("Generating synthetic hold-out and evaluating..."):
            df_syn, y_true, _ = generate_synth_holdout(n=n_val, seed=seed_val)
            Xb_aligned, _ = align_df_to_model(df_syn, model)
            p_model_v = model.predict_proba(to_float32_np(Xb_aligned), validate_features=False)[:, 1]
            p_final_v = overlay_blend_vectorized(df_syn, p_model_v)

            auc_m = roc_auc_score(y_true, p_model_v); auc_f = roc_auc_score(y_true, p_final_v)
            ap_m = average_precision_score(y_true, p_model_v); ap_f = average_precision_score(y_true, p_final_v)
            brier_m = brier_score_loss(y_true, p_model_v); brier_f = brier_score_loss(y_true, p_final_v)
            ece_m = _ece(y_true, p_model_v, 10); ece_f = _ece(y_true, p_final_v, 10)

            st.subheader("Metrics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("AUC (Model)", f"{auc_m:.3f}")
                st.metric("AUC (Final)", f"{auc_f:.3f}")
            with c2:
                st.metric("PR-AUC (Model)", f"{ap_m:.3f}")
                st.metric("PR-AUC (Final)", f"{ap_f:.3f}")
            with c3:
                st.metric("Brier (Model)", f"{brier_m:.3f}")
                st.metric("Brier (Final)", f"{brier_f:.3f}")
                st.caption(f"ECE (Model/Final, 10 bins): {ece_m:.3f} / {ece_f:.3f}")

            st.subheader("Curves")
            _plot_roc_pr(y_true, p_model_v, p_final_v, "Model", "Final")
            _plot_calibration(y_true, p_model_v, p_final_v, n_bins=10, label1="Model", label2="Final")

            st.subheader("Operational confusion (binary)")
            thr_mod = MOD_CUT / 100.0; thr_high = HIGH_CUT / 100.0
            tn1, fp1, fn1, tp1 = _binary_confusion(y_true, p_model_v, thr_mod)
            tn2, fp2, fn2, tp2 = _binary_confusion(y_true, p_final_v, thr_mod)
            tn3, fp3, fn3, tp3 = _binary_confusion(y_true, p_model_v, thr_high)
            tn4, fp4, fn4, tp4 = _binary_confusion(y_true, p_final_v, thr_high)
            ccm1, ccm2 = st.columns(2)
            with ccm1:
                st.markdown(f"**Threshold: Moderate (≥{MOD_CUT}%)**")
                st.write(f"Model — TN:{tn1} FP:{fp1} FN:{fn1} TP:{tp1}")
                st.write(f"Final — TN:{tn2} FP:{fp2} FN:{fn2} TP:{tp2}")
            with ccm2:
                st.markdown(f"**Threshold: High (≥{HIGH_CUT}%)**")
                st.write(f"Model — TN:{tn3} FP:{fp3} FN:{fn3} TP:{tp3}")
                st.write(f"Final — TN:{tn4} FP:{fp4} FN:{fn4} TP:{tp4}")
        st.success("Validation finished.")

# ====== Vignettes ======
st.markdown("---")
st.header("🧾 Vignettes template (for expert review)")

def _mk_vignette_row(age, gender, diags, los, prev, comp, rsh, shadm, sup, fup):
    return {
        "Age": age, "Gender": gender, "Diagnoses": ", ".join(diags),
        "Length of Stay (days)": los, "Previous Admissions (1y)": prev,
        "Medication Compliance Score (0–10)": comp,
        "Recent Self-harm": rsh, "Self-harm During Admission": shadm,
        "Family Support Score (0–10)": sup, "Post-discharge Followups": fup,
        "Expert Risk (Low/Moderate/High or 0–100)": ""
    }

def build_vignettes_df(n=20, seed=77):
    rng = np.random.default_rng(seed); base = []
    protos = [
        (19,"Female",["Depression"],2,0,3,"No","No",2,0),
        (28,"Male",["Substance Use Disorder"],1,3,2,"No","No",3,0),
        (35,"Male",["Bipolar"],6,1,4,"No","No",5,1),
        (42,"Female",["Personality Disorder"],2,2,3,"No","No",4,0),
        (55,"Male",["Schizophrenia"],10,4,5,"No","No",5,2),
        (63,"Female",["PTSD"],4,1,6,"No","No",6,1),
        (72,"Male",["Depression","Anxiety"],5,0,7,"No","No",7,2),
        (23,"Female",["OCD"],3,0,8,"No","No",8,2),
        (31,"Male",["Substance Use Disorder","Depression"],7,2,3,"No","No",3,0),
        (47,"Female",["Personality Disorder","PTSD"],2,1,4,"No","No",4,0),
        (38,"Male",["ADHD"],4,0,6,"No","No",6,1),
        (26,"Female",["Anxiety"],1,0,5,"No","No",5,1),
        (60,"Male",["Dementia"],12,1,6,"No","No",6,2),
        (45,"Female",["Schizophrenia","Substance Use Disorder"],9,3,2,"No","No",3,0),
        (52,"Male",["Bipolar","Personality Disorder"],2,2,3,"No","No",4,0),
        (33,"Female",["Depression"],3,0,8,"No","No",8,3),
        (29,"Male",["Substance Use Disorder"],5,2,2,"No","No",4,0),
        (70,"Female",["Depression","PTSD"],8,1,5,"No","No",6,2),
        (41,"Male",["Personality Disorder","Substance Use Disorder"],2,3,3,"No","No",3,0),
        (36,"Female",["Other/Unknown"],4,0,5,"No","No",5,1),
    ]
    for i in range(min(n, len(protos))): base.append(_mk_vignette_row(*protos[i]))
    for _ in range(len(base), n):
        age = int(np.clip(rng.normal(40,15), 18, 90))
        gender = GENDER_LIST[int(rng.integers(0,len(GENDER_LIST)))]
        k = int(rng.integers(1,3)); diags = list(rng.choice(DIAG_LIST, size=k, replace=False))
        los = int(np.clip(rng.normal(5,3),0,45)); prev = int(np.clip(rng.poisson(1.0),0,8))
        comp = float(np.clip(rng.normal(6,2.5),0,10)); rsh = rng.choice(["Yes","No"]); shadm = rng.choice(["Yes","No"])
        sup = float(np.clip(rng.normal(5,2.5),0,10)); fup = int(np.clip(rng.integers(0,4),0,10))
        base.append(_mk_vignette_row(age, gender, diags, los, prev, comp, rsh, shadm, sup, fup))
    return pd.DataFrame(base)

vdf = build_vignettes_df(20, 77)
buf_v = BytesIO(); vdf.to_excel(buf_v, index=False); buf_v.seek(0)
st.download_button("📥 Download Vignettes (20 cases, Excel)", buf_v,
                   file_name="vignettes_20.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("說明：請 3–5 位老師獨立評每題風險（Low/Moderate/High 或 0–100）；回收後用 Validation 分頁比對。")
