# app.py — Psychiatric Dropout Risk
# (multi-diagnoses + literature-inspired policy overlay + global calibration + smooth blend
#  + border bands + SHAP + Actions + Batch + policy drivers)
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

# ==== Sigmoid / logit helpers ====
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps)); return np.log(p / (1 - p))
def _logit_vec(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

# === Global calibration + smoothing ===
CAL_LOGIT_SHIFT = float(os.getenv("RISK_CAL_SHIFT", "0.40"))   # 全域校正
SOFT_UPLIFT = {"floor": 0.65, "add": 0.20, "cap": 0.95}        # 自傷 uplift
BLEND_W = 0.50                                                 # Final = (1-BLEND)*Model + BLEND*Overlay
BORDER_BAND = 7                                                # 邊帶寬度（score 0–100）

# ====== Unified options ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== Feature template（無社工變數）======
TEMPLATE_COLUMNS = [
    "age","length_of_stay","num_previous_admissions",
    "medication_compliance_score","family_support_score","post_discharge_followups",
    "gender_Male","gender_Female",
] + [f"diagnosis_{d}" for d in DIAG_LIST] + [
    "has_recent_self_harm_Yes","has_recent_self_harm_No",
    "self_harm_during_admission_Yes","self_harm_during_admission_No",
]

# ====== Literature-inspired policy overlay weights (log-odds) ======
POLICY = {
    "per_prev_admission": 0.18,
    "per_point_low_compliance": 0.24,   # (5 - compliance)
    "per_point_low_support": 0.20,      # (5 - family support)
    "per_followup": -0.15,
    "los_short": 0.45,                  # <3d
    "los_mid": 0.00,                    # 3–14d
    "los_mid_high": 0.15,               # 15–21d
    "los_long": 0.35,                   # >21d
    "age_young": 0.10,                  # <21
    "age_old": 0.10,                    # ≥75
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
    "no_followup_extra": 0.20,          # 0 次追蹤的額外加罰
    "x_sud_lowcomp": 0.15,              # SUD × compliance≤3
    "x_pd_shortlos": 0.10,              # PD × LOS<3
}

# ====== Load or train (auto-fallback to demo) ======
def try_load_model(path="dropout_model.pkl"):
    if not os.path.exists(path): return None
    try:
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "model" in bundle: return bundle["model"]
        return bundle
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

    # Marginals
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(5.0, 3.0, n).clip(0, 45)
    X["num_previous_admissions"] = rng.poisson(0.8, n).clip(0, 12)
    X["medication_compliance_score"] = rng.normal(6.0, 2.5, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 6, n)

    # 性別 one-hot
    idx_gender = rng.integers(0, len(GENDER_LIST), n)
    for i, g in enumerate(GENDER_LIST):
        X.loc[idx_gender == i, f"gender_{g}"] = 1

    # 主診斷 + 常見共病
    idx_primary = rng.integers(0, len(DIAG_LIST), n)
    for i, d in enumerate(DIAG_LIST):
        X.loc[idx_primary == i, f"diagnosis_{d}"] = 1
    extra_probs = {"Substance Use Disorder": 0.20, "Anxiety": 0.20, "Depression": 0.25, "PTSD": 0.10}
    for d, pr in extra_probs.items():
        mask = (rng.random(n) < pr); X.loc[mask, f"diagnosis_{d}"] = 1
    has_any = X[[f"diagnosis_{d}" for d in DIAG_LIST]].sum(axis=1) > 0
    if not has_any.all():
        fix_idx = has_any[~has_any].index
        rp = rng.integers(0, len(DIAG_LIST), len(fix_idx))
        for j, ridx in enumerate(fix_idx):
            X.at[ridx, f"diagnosis_{DIAG_LIST[rp[j]]}"] = 1

    # 自傷標記
    idx_rsh = rng.integers(0, 2, n)
    idx_shadm = rng.integers(0, 2, n)
    X.loc[idx_rsh == 1, "has_recent_self_harm_Yes"] = 1
    X.loc[idx_rsh == 0, "has_recent_self_harm_No"] = 1
    X.loc[idx_shadm == 1, "self_harm_during_admission_Yes"] = 1
    X.loc[idx_shadm == 0, "self_harm_during_admission_No"] = 1

    # Balanced literature-inspired logits（demo 目標）
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
    logit = (
        beta0
        + beta["has_recent_self_harm_Yes"]        * X["has_recent_self_harm_Yes"]
        + beta["self_harm_during_admission_Yes"]  * X["self_harm_during_admission_Yes"]
        + beta["prev_adm_ge2"]                    * prev_ge2
        + beta["medication_compliance_per_point"] * X["medication_compliance_score"]
        + beta["family_support_per_point"]        * X["family_support_score"]
        + beta["followups_per_visit"]             * X["post_discharge_followups"]
        + beta["length_of_stay_per_day"]          * X["length_of_stay"]
    )
    for d, w in beta_diag.items():
        logit = logit + w * X[f"diagnosis_{d}"]

    noise = rng.normal(0.0, 0.35, n).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(logit + noise)))
    y = (rng.random(n) < p).astype(np.int32)

    model = xgboost_classifier()
    model.fit(X, y)   # 用 DataFrame 保留 feature_names
    return model

def get_feat_names(m):
    try:
        b = m.get_booster()
        if getattr(b, "feature_names", None): return list(b.feature_names)
    except Exception:
        pass
    if hasattr(m, "feature_names_in_"): return list(m.feature_names_in_)
    return None

model = try_load_model()
loaded = model is not None
use_demo = False
if model is not None:
    names = get_feat_names(model)
    if (names is None) or (abs(len(names) - len(TEMPLATE_COLUMNS)) > 3):
        use_demo = True

if (not loaded) or use_demo:
    model = train_demo_model(TEMPLATE_COLUMNS)
    model_source = "demo (balanced weights, multi-diagnoses, higher baseline)"
else:
    model_source = "loaded from dropout_model.pkl"

# ====== Alignment helpers ======
def get_model_feature_order(m):
    order = None; exp_len = None
    try:
        booster = getattr(m, "get_booster", lambda: None)()
        if booster is not None:
            names = getattr(booster, "feature_names", None)
            if names: order, exp_len = list(names), len(names)
    except Exception:
        pass
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

# ====== Small helpers ======
def set_onehot_by_prefix(df, prefix, value):
    col = f"{prefix}_{value}"
    if col in df.columns: df.at[0, col] = 1

def set_onehot_by_prefix_multi(df, prefix, values):
    for v in values:
        col = f"{prefix}_{v}"
        if col in df.columns: df.at[0, col] = 1

def flag_yes(row, prefix):
    col = f"{prefix}_Yes"; return (col in row.index) and (row[col] == 1)

# ====== Thresholds + soft classification ======
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

# ====== Sidebar ======
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
    CAL_LOGIT_SHIFT = cal_shift

# ====== Build single-row DF ======
X_final = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
for k, v in {
    "age": age,
    "length_of_stay": float(length_of_stay),
    "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance),
    "family_support_score": float(support),
    "post_discharge_followups": int(followups),
}.items(): X_final.at[0, k] = v
set_onehot_by_prefix(X_final, "gender", gender)
set_onehot_by_prefix_multi(X_final, "diagnosis", diagnoses)
set_onehot_by_prefix(X_final, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_final, "self_harm_during_admission", selfharm_adm)

# ====== Predict (align + overlay + calibration + blending) ======
X_aligned_df, used_names = align_df_to_model(X_final, model)
X_np = to_float32_np(X_aligned_df)
p_model = float(model.predict_proba(X_np, validate_features=False)[:, 1][0])

# ---- Policy overlay on logit space（收集驅動因子）----
drivers = []  # list of (label, contribution)
def add_driver(label, val):
    if val != 0:
        drivers.append((label, float(val)))
    return val

lz = _logit(p_model)

# 住院史
lz += add_driver("More previous admissions",
                 POLICY["per_prev_admission"] * min(int(X_final.at[0, "num_previous_admissions"]), 5))

# 順從、家支（以 5 為中心）
lz += add_driver("Low medication compliance",
                 POLICY["per_point_low_compliance"] * max(0.0, 5.0 - float(X_final.at[0, "medication_compliance_score"])))
lz += add_driver("Low family support",
                 POLICY["per_point_low_support"] * max(0.0, 5.0 - float(X_final.at[0, "family_support_score"])))

# 追蹤
lz += add_driver("More post-discharge followups (protective)",
                 POLICY["per_followup"] * float(X_final.at[0, "post_discharge_followups"]))
if float(X_final.at[0, "post_discharge_followups"]) == 0:
    lz += add_driver("No follow-up scheduled", POLICY["no_followup_extra"])

# 住院日數
los = float(X_final.at[0, "length_of_stay"])
if los < 3:
    lz += add_driver("Very short stay (<3d)", POLICY["los_short"])
elif los <= 14:
    lz += add_driver("Typical stay (3–14d)", POLICY["los_mid"])
elif los <= 21:
    lz += add_driver("Longish stay (15–21d)", POLICY["los_mid_high"])
else:
    lz += add_driver("Very long stay (>21d)", POLICY["los_long"])

# 年齡（極端）
age_val = float(X_final.at[0, "age"])
if age_val < 21:
    lz += add_driver("Young age (<21)", POLICY["age_young"])
elif age_val >= 75:
    lz += add_driver("Older age (≥75)", POLICY["age_old"])

# 診斷（可複選相加）
for dx, w in POLICY["diag"].items():
    col = f"diagnosis_{dx}"
    if col in X_final.columns and X_final.at[0, col] == 1:
        lz += add_driver(f"Diagnosis: {dx}", w)

# 交互效應（修正：用單列 row0 取得純量，避免 Series 布林歧義）
row0 = X_final.iloc[0]
has_sud = bool(row0.get("diagnosis_Substance Use Disorder", 0) == 1)
if has_sud and float(row0["medication_compliance_score"]) <= 3:
    lz += add_driver("SUD × very low compliance", POLICY["x_sud_lowcomp"])
has_pd = bool(row0.get("diagnosis_Personality Disorder", 0) == 1)
if has_pd and los < 3:
    lz += add_driver("PD × very short stay", POLICY["x_pd_shortlos"])

# 全域校正 → overlay 機率
lz += CAL_LOGIT_SHIFT
p_overlay = _sigmoid(lz)

# 平滑混合
p_policy = (1.0 - BLEND_W) * p_model + BLEND_W * p_overlay

# ---- Soft safety uplift（不鎖死，只提升）----
soft_reason = None
if flag_yes(X_final.iloc[0], "has_recent_self_harm") or flag_yes(X_final.iloc[0], "self_harm_during_admission"):
    p_final = min(max(p_policy, SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"], SOFT_UPLIFT["cap"])
    soft_reason = "self-harm uplift"
else:
    p_final = p_policy

percent_model = proba_to_percent(p_model)
percent = proba_to_percent(p_final)
score = proba_to_score(p_final)
level = classify_soft(score)

# ====== Model diagnostics ======
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

# ====== Show result ======
st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Model Probability", f"{percent_model:.1f}%")
with c2: st.metric("Final Probability", f"{percent:.1f}%")
with c3: st.metric("Risk Score (0–100)", f"{score}")

if soft_reason:
    st.warning(f"🟠 Soft safety uplift applied ({soft_reason}).")
else:
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

# ====== Why did the risk go up? (policy drivers) ======
with st.expander("Why did the risk go up? (policy drivers)", expanded=False):
    if drivers:
        df_drv = pd.DataFrame(
            [{"driver": k, "log-odds +": round(v, 3)} for k, v in sorted(drivers, key=lambda x: abs(x[1]), reverse=True)]
        )
        st.dataframe(df_drv, use_container_width=True)
    else:
        st.caption("No positive policy drivers; Final is close to the model output and protective factors.")

# ====== SHAP (model-only explanation) ======
with st.expander("SHAP Explanation (model component)", expanded=True):
    st.caption("Positive bars push toward higher dropout risk; negative bars lower it. Only the selected category for each one-hot feature is shown.")
    import xgboost as xgb
    try:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X_aligned_df, feature_names=list(X_aligned_df.columns))
        contribs = booster.predict(dmat, pred_contribs=True, validate_features=False)
        contrib = np.asarray(contribs)[0]
        base_value = float(contrib[-1])
        feat_contrib = contrib[:-1]
        sv_map = dict(zip(list(X_aligned_df.columns), feat_contrib))
    except Exception:
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(X_aligned_df)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
            base_value = base_value[0]
            if isinstance(sv_raw, list): sv_raw = sv_raw[0]
        sv_raw = sv_raw[0]; sv_map = dict(zip(list(X_aligned_df.columns), sv_raw))

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

    if len(vals) == 0:
        st.caption("No SHAP contributions available.")
    else:
        order = np.argsort(np.abs(np.array(vals)))[::-1][:12]
        exp = shap.Explanation(
            values=np.array(vals, dtype=float)[order],
            base_values=base_value,
            feature_names=[names[i] for i in order],
            data=np.array(data_vals, dtype=float)[order],
        )
        shap.plots.waterfall(exp, show=False, max_display=12)
        st.pyplot(plt.gcf(), clear_figure=True)

# ====== Recommended Actions ======
st.subheader("Recommended Actions")
BASE_ACTIONS = {
    "High": [
        ("Today","Clinic scheduler","Book return within 7 days."),
        ("Today","Social worker","Enroll in case management."),
        ("Today","Clinician","Safety plan + crisis hotline."),
    ],
    "Moderate": [
        ("1–2 weeks","Clinic scheduler","Schedule return."),
        ("1–2 weeks","Nurse","Check adherence barriers.")
    ],
    "Low": [
        ("2–4 weeks","Clinic scheduler","Routine follow-up."),
        ("2–4 weeks","Nurse","Provide education materials.")
    ],
}
def personalized_actions(row: pd.Series):
    acts = []
    if flag_yes(row, "has_recent_self_harm"):
        acts += [("Today","Clinician","C-SSRS assessment; update safety plan.")]
    if flag_yes(row, "self_harm_during_admission"):
        acts += [("Today","Clinician","Immediate psychiatric evaluation.")]
    return acts

bucket = {"High":"High","Moderate–High":"Moderate","Moderate":"Moderate","Low–Moderate":"Low","Low":"Low"}
rows = BASE_ACTIONS[bucket[level]] + personalized_actions(X_final.iloc[0])
seen, uniq = set(), []
for r in rows:
    if r not in seen: seen.add(r); uniq.append(r)

c_timeline, c_owner, c_action = st.columns([1,1,3])
with c_timeline: st.markdown("**Timeline**");       [st.write(tl) for tl,_,_ in uniq]
with c_owner:    st.markdown("**Owner**");          [st.write(ow) for _,ow,_ in uniq]
with c_action:   st.markdown("**Action**");         [st.write(ac) for _,_,ac in uniq]

# ====== SOP export (High only) ======
if level == "High":
    def make_sop_txt(score: int, label: str, actions: list) -> BytesIO:
        lines = ["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}", ""]
        for (tl, ow, ac) in actions: lines.append(f"- {tl} | {ow} | {ac}")
        buf = BytesIO("\n".join(lines).encode("utf-8")); buf.seek(0); return buf
    st.download_button("⬇️ Export SOP (TXT)", make_sop_txt(score, level, uniq),
                       file_name="dropout_risk_SOP.txt", mime="text/plain")

    if HAS_DOCX:
        def make_sop_docx(score: int, label: str, actions: list) -> BytesIO:
            doc = Document()
            doc.add_heading('Psychiatric Dropout Risk – SOP', level=1)
            doc.add_paragraph(f"Risk score: {score}/100 | Risk level: {label}")
            t = doc.add_table(rows=1, cols=3)
            hdr = t.rows[0].cells
            hdr[0].text = 'Timeline'; hdr[1].text = 'Owner'; hdr[2].text = 'Action'
            for (tl, ow, ac) in actions:
                r = t.add_row().cells; r[0].text = tl; r[1].text = ow; r[2].text = ac
            buf = BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== Batch Prediction (Excel) ======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

friendly_cols = [
    "Age","Gender","Diagnoses",  # 多診斷（逗號/分號/斜線/| 分隔），相容舊欄位 Diagnosis
    "Length of Stay (days)","Previous Admissions (1y)",
    "Medication Compliance Score (0–10)",
    "Recent Self-harm","Self-harm During Admission",
    "Family Support Score (0–10)","Post-discharge Followups"
]
tpl_df = pd.DataFrame(columns=friendly_cols)
tpl_buf = BytesIO(); tpl_df.to_excel(tpl_buf, index=False); tpl_buf.seek(0)
st.download_button("📥 Download Excel Template", tpl_buf, file_name="batch_template.xlsx",
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
                    if v in options:
                        col = f"{prefix}_{v}"
                        if col in df.columns: df.at[i, col] = 1

        def apply_onehot_prefix(human_col, prefix, options):
            if human_col not in raw.columns: return
            for i, v in raw[human_col].astype(str).str.strip().items():
                if v in options:
                    col = f"{prefix}_{v}"
                    if col in df.columns: df.at[i, col] = 1

        apply_onehot_prefix("Gender","gender", GENDER_LIST)
        if "Diagnoses" in raw.columns:
            apply_onehot_prefix_multi("Diagnoses","diagnosis", DIAG_LIST)
        elif "Diagnosis" in raw.columns:
            apply_onehot_prefix_multi("Diagnosis","diagnosis", DIAG_LIST)
        apply_onehot_prefix("Recent Self-harm","has_recent_self_harm", BIN_YESNO)
        apply_onehot_prefix("Self-harm During Admission","self_harm_during_admission", BIN_YESNO)

        Xb_aligned, _ = align_df_to_model(df, model)
        Xb_np = to_float32_np(Xb_aligned)
        base_probs = model.predict_proba(Xb_np, validate_features=False)[:, 1]

        # ---- Vectorized policy overlay for batch（含年齡、零追蹤加罰、交互效應）----
        lz = _logit_vec(base_probs)

        # 住院史
        lz += POLICY["per_prev_admission"] * np.minimum(df["num_previous_admissions"].astype(float).to_numpy(), 5)

        # 順從、家支
        lz += POLICY["per_point_low_compliance"] * np.maximum(0.0, 5.0 - df["medication_compliance_score"].astype(float).to_numpy())
        lz += POLICY["per_point_low_support"] * np.maximum(0.0, 5.0 - df["family_support_score"].astype(float).to_numpy())

        # 追蹤 + 零追蹤加罰
        fup = df["post_discharge_followups"].astype(float).to_numpy()
        lz += POLICY["per_followup"] * fup
        lz += POLICY["no_followup_extra"] * (fup == 0)

        # LOS
        los_arr = df["length_of_stay"].astype(float).to_numpy()
        lz += np.where(los_arr < 3, POLICY["los_short"],
                np.where(los_arr <= 14, POLICY["los_mid"],
                np.where(los_arr <= 21, POLICY["los_mid_high"], POLICY["los_long"])))

        # 年齡極端
        age_arr = df["age"].astype(float).to_numpy()
        lz += POLICY["age_young"] * (age_arr < 21)
        lz += POLICY["age_old"]   * (age_arr >= 75)

        # 診斷
        diag_term = np.zeros(len(df), dtype=float)
        for dx, w in POLICY["diag"].items():
            col = f"diagnosis_{dx}"
            if col in df.columns:
                diag_term += w * (df[col].to_numpy() == 1)
        lz += diag_term

        # 交互效應
        sud = (df.get("diagnosis_Substance Use Disorder", 0).to_numpy() == 1)
        very_low_comp = (df["medication_compliance_score"].astype(float).to_numpy() <= 3)
        lz += POLICY["x_sud_lowcomp"] * (sud & very_low_comp)

        pd_mask = (df.get("diagnosis_Personality Disorder", 0).to_numpy() == 1)
        lz += POLICY["x_pd_shortlos"] * (pd_mask & (los_arr < 3))

        # 校正 + 混合
        lz += CAL_LOGIT_SHIFT
        p_overlay = 1.0 / (1.0 + np.exp(-lz))
        p_policy = (1.0 - BLEND_W) * base_probs + BLEND_W * p_overlay

        # self-harm uplift
        hrsh = df.get("has_recent_self_harm_Yes", 0)
        shadm = df.get("self_harm_during_admission_Yes", 0)
        soft_mask = ((np.array(hrsh) == 1) | (np.array(shadm) == 1))
        adj_probs = p_policy.copy()
        adj_probs[soft_mask] = np.minimum(
            np.maximum(adj_probs[soft_mask], SOFT_UPLIFT["floor"]) + SOFT_UPLIFT["add"],
            SOFT_UPLIFT["cap"]
        )

        out = raw.copy()
        out["risk_percent"] = (adj_probs * 100).round(1)
        out["risk_score_0_100"] = (adj_probs * 100).round().astype(int)

        # 邊帶分級（向量化）
        s = out["risk_score_0_100"].to_numpy()
        levels = np.full(s.shape, "Low", dtype=object)
        levels[s >= MOD_CUT - BORDER_BAND] = "Low–Moderate"
        levels[s >= MOD_CUT + BORDER_BAND] = "Moderate"
        levels[s >= HIGH_CUT - BORDER_BAND] = "Moderate–High"
        levels[s >= HIGH_CUT + BORDER_BAND] = "High"
        out["risk_level"] = levels

        # Top-3 policy drivers（字串）
        top3_list = []
        for i in range(len(df)):
            contribs = []
            def push(name, val):
                if val != 0: contribs.append((name, float(val)))
            # 住院史
            push("More previous admissions", POLICY["per_prev_admission"] * min(float(df.iloc[i]["num_previous_admissions"]), 5))
            # 順從/家支
            push("Low medication compliance", POLICY["per_point_low_compliance"] * max(0.0, 5.0 - float(df.iloc[i]["medication_compliance_score"])))
            push("Low family support", POLICY["per_point_low_support"] * max(0.0, 5.0 - float(df.iloc[i]["family_support_score"])))
            # 追蹤
            fu_i = float(df.iloc[i]["post_discharge_followups"])
            push("More post-discharge followups (protective)", POLICY["per_followup"] * fu_i)
            if fu_i == 0: push("No follow-up scheduled", POLICY["no_followup_extra"])
            # LOS
            los_i = float(df.iloc[i]["length_of_stay"])
            if los_i < 3: push("Very short stay (<3d)", POLICY["los_short"])
            elif los_i <= 14: push("Typical stay (3–14d)", POLICY["los_mid"])
            elif los_i <= 21: push("Longish stay (15–21d)", POLICY["los_mid_high"])
            else: push("Very long stay (>21d)", POLICY["los_long"])
            # 年齡
            age_i = float(df.iloc[i]["age"])
            if age_i < 21: push("Young age (<21)", POLICY["age_young"])
            elif age_i >= 75: push("Older age (≥75)", POLICY["age_old"])
            # 診斷
            for dx, w in POLICY["diag"].items():
                if f"diagnosis_{dx}" in df.columns and df.iloc[i][f"diagnosis_{dx}"] == 1:
                    push(f"Diagnosis: {dx}", w)
            # 交互
            sud_i = (df.iloc[i].get("diagnosis_Substance Use Disorder", 0) == 1)
            if sud_i and float(df.iloc[i]["medication_compliance_score"]) <= 3:
                push("SUD × very low compliance", POLICY["x_sud_lowcomp"])
            pd_i = (df.iloc[i].get("diagnosis_Personality Disorder", 0) == 1)
            if pd_i and los_i < 3:
                push("PD × very short stay", POLICY["x_pd_shortlos"])

            if len(contribs) == 0:
                top3_list.append("")
            else:
                contribs.sort(key=lambda x: abs(x[1]), reverse=True)
                top3 = [f"{n} ({v:+.2f})" for n, v in contribs[:3]]
                top3_list.append(" | ".join(top3))
        out["policy_drivers_top3"] = top3_list

        st.dataframe(out, use_container_width=True)
        buf_out = BytesIO(); out.to_csv(buf_out, index=False); buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
