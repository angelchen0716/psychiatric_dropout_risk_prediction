# app.py — Psychiatric Dropout Risk (literature-weighted synthetic training + SHAP + Actions + Batch + robust alignment)
import os
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

# ====== Unified options (match sidebar/Excel/SHAP) ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== Feature template (semantic match with UI) ======
TEMPLATE_COLUMNS = [
    "age","length_of_stay","num_previous_admissions",
    "medication_compliance_score","family_support_score","post_discharge_followups",
    "gender_Male","gender_Female",
] + [f"diagnosis_{d}" for d in DIAG_LIST] + [
    "has_social_worker_Yes","has_social_worker_No",
    "has_recent_self_harm_Yes","has_recent_self_harm_No",
    "self_harm_during_admission_Yes","self_harm_during_admission_No",
]

# ====== Load model or train from literature-weighted synthetic data ======
def try_load_model(path="dropout_model.pkl"):
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"]
    return bundle

model = try_load_model()
if model is None:
    import xgboost as xgb
    st.warning("⚠️ No model found. Training a demo model from literature-weighted synthetic data.")
    rng = np.random.default_rng(42)
    n = 8000
    X = pd.DataFrame(0, index=range(n), columns=TEMPLATE_COLUMNS, dtype=np.float32)

    # Marginals (adjustable)
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(5.0, 3.0, n).clip(0, 45)
    X["num_previous_admissions"] = rng.poisson(0.6, n).clip(0, 10)
    X["medication_compliance_score"] = rng.normal(6.0, 2.2, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.5, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 6, n)

    def pick_one(prefix, options):
        idx = rng.integers(0, len(options), n)
        for i, opt in enumerate(options):
            X.loc[idx == i, f"{prefix}_{opt}"] = 1

    pick_one("gender", GENDER_LIST)
    pick_one("diagnosis", DIAG_LIST)
    pick_one("has_social_worker", BIN_YESNO)
    pick_one("has_recent_self_harm", BIN_YESNO)
    pick_one("self_harm_during_admission", BIN_YESNO)

    # Literature-inspired logit weights
    beta0 = -1.20
    beta = {
        "has_recent_self_harm_Yes": 1.50,
        "self_harm_during_admission_Yes": 1.20,
        "prev_adm_ge2": 0.50,
        "medication_compliance_per_point": -0.12,
        "family_support_per_point": -0.10,
        "followups_per_visit": -0.08,
        "length_of_stay_per_day": 0.02,
        "has_social_worker_Yes": -0.25
    }
    beta_diag = {
        "Schizophrenia": 0.40,"Bipolar": 0.35,"Depression": 0.25,"Personality Disorder": 0.30,
        "Substance Use Disorder": 0.35,"Dementia": 0.15,"Anxiety": 0.15,"PTSD": 0.20,
        "OCD": 0.10,"ADHD": 0.10,"Other/Unknown": 0.10,
    }

    prev_ge2 = (X["num_previous_admissions"] >= 2).astype(np.float32)
    logit = (
        beta0
        + beta["has_recent_self_harm_Yes"]       * X["has_recent_self_harm_Yes"]
        + beta["self_harm_during_admission_Yes"] * X["self_harm_during_admission_Yes"]
        + beta["prev_adm_ge2"]                   * prev_ge2
        + beta["medication_compliance_per_point"]* X["medication_compliance_score"]
        + beta["family_support_per_point"]       * X["family_support_score"]
        + beta["followups_per_visit"]            * X["post_discharge_followups"]
        + beta["length_of_stay_per_day"]         * X["length_of_stay"]
        + beta["has_social_worker_Yes"]          * X["has_social_worker_Yes"]
    )
    for d, w in beta_diag.items():
        col = f"diagnosis_{d}"
        if col in X.columns:
            logit = logit + w * X[col]

    noise = rng.normal(0.0, 0.35, n).astype(np.float32)
    logit = logit + noise
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.int32)
    st.caption(f"Simulated prevalence ≈ {p.mean():.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
    )
    model.fit(X.values.astype(np.float32), y)

# ====== Model-feature alignment helpers (avoid XGBoost columnar pitfalls) ======
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
        if out.shape[1] > exp_len:
            out = out.iloc[:, :exp_len]
        else:
            add = exp_len - out.shape[1]
            pad = pd.DataFrame(0, index=out.index, columns=[f"_pad_{i}" for i in range(add)], dtype=np.float32)
            out = pd.concat([out, pad], axis=1)
    return out, list(out.columns)

def to_float32_np(df: pd.DataFrame):
    return df.astype(np.float32).values

# ====== Small helpers ======
def set_onehot_by_prefix(df, prefix, value):
    col = f"{prefix}_{value}"
    if col in df.columns:
        df.at[0, col] = 1

def flag_yes(row, prefix):
    col = f"{prefix}_Yes"
    return (col in row.index) and (row[col] == 1)

MOD_CUT = 30
HIGH_CUT = 50
def proba_to_percent(p): return float(p) * 100
def proba_to_score(p): return int(round(proba_to_percent(p)))
def classify(score):
    if score >= HIGH_CUT: return "High"
    if score >= MOD_CUT:  return "Moderate"
    return "Low"

# ====== Sidebar (single-patient input) ======
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 90, 35)
    gender = st.selectbox("Gender", GENDER_LIST)
    diagnosis = st.selectbox("Diagnosis", DIAG_LIST)
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("Previous Admissions (1y)", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", BIN_YESNO, index=1)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0)
    recent_self_harm = st.radio("Recent Self-harm", BIN_YESNO, index=1)
    selfharm_adm = st.radio("Self-harm During Admission", BIN_YESNO, index=1)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0)
    followups = st.slider("Post-discharge Followups", 0, 10, 2)

# ====== Build X (single) ======
X_final = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
for k, v in {
    "age": age,
    "length_of_stay": float(length_of_stay),
    "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance),
    "family_support_score": float(support),
    "post_discharge_followups": int(followups),
}.items():
    X_final.at[0, k] = v
set_onehot_by_prefix(X_final, "gender", gender)
set_onehot_by_prefix(X_final, "diagnosis", diagnosis)
set_onehot_by_prefix(X_final, "has_social_worker", social_worker)
set_onehot_by_prefix(X_final, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_final, "self_harm_during_admission", selfharm_adm)

# ====== Predict (align features + float32 + validate_features=False) ======
X_aligned_df, used_names = align_df_to_model(X_final, model)
X_np = to_float32_np(X_aligned_df)
base_prob = model.predict_proba(X_np, validate_features=False)[:, 1][0]

percent, score = proba_to_percent(base_prob), proba_to_score(base_prob)
level = classify(score)
override_reason = None
if flag_yes(X_final.iloc[0], "has_recent_self_harm"):
    percent, score, level = 70.0, 70, "High"; override_reason = "recent self-harm"
elif flag_yes(X_final.iloc[0], "self_harm_during_admission"):
    percent, score, level = 70.0, 70, "High"; override_reason = "in-hospital self-harm"

# ====== Show result ======
st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2 = st.columns(2)
with c1: st.metric("Probability", f"{percent:.1f}%")
with c2: st.metric("Risk Score (0–100)", f"{score}")
if override_reason:
    st.error(f"🔴 High Risk (safety override: {override_reason})")
    st.caption("ℹ️ Determined by a clinical safety override, not purely by the model output.")
elif level == "High":
    st.error("🔴 High Risk")
elif level == "Moderate":
    st.warning("🟡 Moderate Risk")
else:
    st.success("🟢 Low Risk")

# ====== SHAP (grouped display, English captions) ======
with st.expander("SHAP Explanation", expanded=True):
    st.caption("How to read: positive bars push toward higher dropout risk; negative bars lower it. Only the selected category for each one-hot feature is shown.")
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_np)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv_raw, list): sv_raw = sv_raw[0]
    sv_raw = sv_raw[0]
    sv_map = dict(zip(used_names, sv_raw))

    cont_feats = [
        ("Age","age", X_final.at[0,"age"]),
        ("Length of Stay (days)","length_of_stay", X_final.at[0,"length_of_stay"]),
        ("Previous Admissions (1y)","num_previous_admissions", X_final.at[0,"num_previous_admissions"]),
        ("Medication Compliance (0–10)","medication_compliance_score", X_final.at[0,"medication_compliance_score"]),
        ("Family Support (0–10)","family_support_score", X_final.at[0,"family_support_score"]),
        ("Post-discharge Followups","post_discharge_followups", X_final.at[0,"post_discharge_followups"]),
    ]
    names, vals, data_vals = [], [], []
    for label, key, dv in cont_feats:
        if key in sv_map:
            names.append(label); vals.append(sv_map[key]); data_vals.append(dv)

    def add_onehot(title, prefix, value):
        col = f"{prefix}_{value}"
        if col in sv_map:
            names.append(f"{title}={value}"); vals.append(sv_map[col]); data_vals.append(1)

    add_onehot("Gender","gender", gender)
    add_onehot("Diagnosis","diagnosis", diagnosis)
    add_onehot("Has Social Worker","has_social_worker", social_worker)
    add_onehot("Recent Self-harm","has_recent_self_harm", recent_self_harm)
    add_onehot("Self-harm During Admission","self_harm_during_admission", selfharm_adm)

    order = np.argsort(np.abs(np.array(vals)))[::-1][:12]
    exp = shap.Explanation(
        values=np.array(vals)[order],
        base_values=base_value,
        feature_names=[names[i] for i in order],
        data=np.array(data_vals)[order],
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

rows = BASE_ACTIONS[level] + personalized_actions(X_final.iloc[0])
seen, uniq = set(), []
for r in rows:
    if r not in seen:
        seen.add(r); uniq.append(r)

c_timeline, c_owner, c_action = st.columns([1,1,3])
with c_timeline:
    st.markdown("**Timeline**");       [st.write(tl) for tl,_,_ in uniq]
with c_owner:
    st.markdown("**Owner**");          [st.write(ow) for _,ow,_ in uniq]
with c_action:
    st.markdown("**Action**");         [st.write(ac) for _,_,ac in uniq]

# ====== SOP export (only when High risk) ======
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
    "Age","Gender","Diagnosis","Length of Stay (days)","Previous Admissions (1y)",
    "Has Social Worker","Medication Compliance Score (0–10)",
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

        def safe_get(col, default=0):
            return raw[col] if col in raw.columns else default
        df["age"] = safe_get("Age")
        df["length_of_stay"] = safe_get("Length of Stay (days)")
        df["num_previous_admissions"] = safe_get("Previous Admissions (1y)")
        df["medication_compliance_score"] = safe_get("Medication Compliance Score (0–10)")
        df["family_support_score"] = safe_get("Family Support Score (0–10)")
        df["post_discharge_followups"] = safe_get("Post-discharge Followups")

        def apply_onehot_prefix(human_col, prefix, options):
            if human_col not in raw.columns: return
            for i, v in raw[human_col].astype(str).str.strip().items():
                if v not in options: continue
                col = f"{prefix}_{v}"
                if col in df.columns: df.at[i, col] = 1

        apply_onehot_prefix("Gender","gender", GENDER_LIST)
        apply_onehot_prefix("Diagnosis","diagnosis", DIAG_LIST)
        apply_onehot_prefix("Has Social Worker","has_social_worker", BIN_YESNO)
        apply_onehot_prefix("Recent Self-harm","has_recent_self_harm", BIN_YESNO)
        apply_onehot_prefix("Self-harm During Admission","self_harm_during_admission", BIN_YESNO)

        Xb_aligned, used_names_b = align_df_to_model(df, model)
        Xb_np = to_float32_np(Xb_aligned)
        base_probs = model.predict_proba(Xb_np, validate_features=False)[:, 1]

        adj_probs = base_probs.copy()
        yes_recent = (df["has_recent_self_harm_Yes"] == 1)
        yes_adm = (df["self_harm_during_admission_Yes"] == 1)
        adj_probs[yes_recent | yes_adm] = 0.70

        out = raw.copy()
        out["risk_percent"] = (adj_probs * 100).round(1)
        out["risk_score_0_100"] = (adj_probs * 100).round().astype(int)
        out["risk_level"] = out["risk_score_0_100"].apply(
            lambda s: "High" if s >= HIGH_CUT else ("Moderate" if s >= MOD_CUT else "Low")
        )
        st.dataframe(out)

        buf_out = BytesIO(); out.to_csv(buf_out, index=False); buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
