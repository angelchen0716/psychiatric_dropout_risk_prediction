# app.py — Psychiatric Dropout Risk (full version)
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Optional Word export
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")

# ====== UI choice lists ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== Load model or build synthetic ======
def try_load_model(path="dropout_model.pkl"):
    if not os.path.exists(path):
        return None, None
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle["model"], bundle.get("columns")
    return bundle, None

model, pretrained_cols = try_load_model()

def resolve_template_columns(model, pretrained_cols):
    if isinstance(pretrained_cols, (list, tuple)) and len(pretrained_cols) > 0:
        return list(pretrained_cols)
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback from UI
    cols = [
        "age","length_of_stay","num_previous_admissions",
        "medication_compliance_score","family_support_score","post_discharge_followups",
    ]
    cols += [f"gender_{g}" for g in GENDER_LIST]
    cols += [f"diagnosis_{d}" for d in DIAG_LIST]
    cols += [f"has_social_worker_{v}" for v in BIN_YESNO]
    cols += [f"has_recent_self_harm_{v}" for v in BIN_YESNO]
    cols += [f"self_harm_during_admission_{v}" for v in BIN_YESNO]
    return cols

TEMPLATE_COLUMNS = resolve_template_columns(model, pretrained_cols)

if model is None:
    import xgboost as xgb
    st.warning("⚠️ dropout_model.pkl not found — building synthetic demo model.")
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame(0, index=range(n), columns=TEMPLATE_COLUMNS, dtype=float)
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(3.5, 2.0, n).clip(0, 30)
    X["num_previous_admissions"] = rng.poisson(0.3, n).clip(0, 6)
    X["medication_compliance_score"] = rng.normal(6.5, 2.0, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.0, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 4, n)
    def pick_one(prefix, options):
        idx = rng.integers(0, len(options), n)
        for i, opt in enumerate(options):
            X.loc[idx == i, f"{prefix}_{opt}"] = 1
    pick_one("gender", GENDER_LIST)
    pick_one("diagnosis", DIAG_LIST)
    pick_one("has_social_worker", BIN_YESNO)
    pick_one("has_recent_self_harm", BIN_YESNO)
    pick_one("self_harm_during_admission", BIN_YESNO)
    logit = -2 + 1.3*X["has_recent_self_harm_Yes"] + 0.9*X["self_harm_during_admission_Yes"] \
            - 0.3*X["medication_compliance_score"] + 0.4*(X["num_previous_admissions"]>=1)
    p = 1/(1+np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, subsample=0.9, colsample_bytree=0.8,
        learning_rate=0.06, eval_metric="logloss", random_state=42
    )
    model.fit(X, y)

# ====== Risk scoring ======
MOD_CUT = 30
HIGH_CUT = 50
def proba_to_percent(p: float) -> float: return float(p) * 100.0
def proba_to_score(p: float) -> int:     return int(round(proba_to_percent(p)))
def classify(score: int) -> str:
    if score >= HIGH_CUT: return "High"
    if score >= MOD_CUT:  return "Moderate"
    return "Low"
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(p):   return np.log(np.clip(p,1e-8,1-1e-8)/np.clip(1-p,1e-8,1-1e-8))
def recalibrate_probability(row: pd.Series, base_prob: float) -> float:
    z = _logit(base_prob)
    if int(row.get('has_recent_self_harm_Yes', 0)) == 1: z += 1.5
    if int(row.get('self_harm_during_admission_Yes', 0)) == 1: z += 1.0
    return float(_sigmoid(z))

# ====== Sidebar: patient info ======
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 90, 35)
    gender = st.selectbox("Gender", GENDER_LIST)
    diagnosis = st.selectbox("Diagnosis", DIAG_LIST)
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("Previous Admissions (1y)", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", BIN_YESNO, horizontal=True)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0, 0.1)
    recent_self_harm = st.radio("Recent Self-harm", BIN_YESNO, horizontal=True)
    selfharm_adm = st.radio("Self-harm During Admission", BIN_YESNO, horizontal=True)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0, 0.1)
    followups = st.slider("Post-discharge Followups (booked)", 0, 10, 2)

# ====== Build X_final ======
X_final = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
assign_map = {
    "age": age,
    "length_of_stay": float(length_of_stay),
    "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance),
    "family_support_score": float(support),
    "post_discharge_followups": int(followups),
}
for k, v in assign_map.items():
    if k in X_final.columns: X_final.at[0, k] = v
for col in [
    f"gender_{gender}",
    f"diagnosis_{diagnosis}",
    f"has_social_worker_{social_worker}",
    f"has_recent_self_harm_{recent_self_harm}",
    f"self_harm_during_admission_{selfharm_adm}",
]:
    if col in X_final.columns: X_final.at[0, col] = 1

# ====== Predict & classify ======
base_prob = model.predict_proba(X_final, validate_features=False)[0][1]
adj_prob  = recalibrate_probability(X_final.iloc[0], base_prob)
percent, score = proba_to_percent(adj_prob), proba_to_score(adj_prob)
level = classify(score)

st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2 = st.columns(2)
with c1: st.metric("Probability", f"{percent:.1f}%")
with c2: st.metric("Risk Score (0–100)", f"{score}")
if level == "High":      st.error("🔴 High Risk")
elif level == "Moderate":st.warning("🟡 Moderate Risk")
else:                    st.success("🟢 Low Risk")
st.markdown("---")

# ====== SHAP waterfall ======
with st.expander("SHAP Explanation", expanded=True):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_final)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv, list):
            sv = sv[0]
    exp = shap.Explanation(
        values=sv[0],
        base_values=base_value,
        feature_names=list(X_final.columns),
        data=X_final.iloc[0].values
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# ====== Actions ======
st.subheader("Recommended Actions")
BASE_ACTIONS = {
    "High": [
        ("Today","Clinic scheduler","Book return within 7 days (before discharge)."),
        ("Today","Social worker","Enroll in CM/peer support; create plan."),
        ("Today","Clinician","Safety plan + crisis hotline."),
    ],
    "Moderate": [("1–2 weeks","Clinic scheduler","Schedule return.")],
    "Low": [("2–4 weeks","Clinic scheduler","Routine follow-up.")],
}
def personalized_actions(row: pd.Series):
    acts = []
    if int(row.get('has_recent_self_harm_Yes',0))==1:
        acts += [("Today","Clinician","C-SSRS assessment; update safety plan.")]
    return acts
rows = BASE_ACTIONS[level] + personalized_actions(X_final.iloc[0])
seen, uniq = set(), []
for r in rows:
    if r not in seen:
        seen.add(r); uniq.append(r)
for tl, ow, ac in uniq:
    st.markdown(f"**{tl}** | {ow} | {ac}")

# ====== SOP export if High ======
if level == "High":
    def make_sop_txt(score: int, label: str, actions: list) -> BytesIO:
        lines = ["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}", ""]
        for (tl, ow, ac) in actions:
            lines.append(f"- {tl} | {ow} | {ac}")
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
            hdr[0].text='Timeline'; hdr[1].text='Owner'; hdr[2].text='Action'
            for (tl, ow, ac) in actions:
                r = t.add_row().cells
                r[0].text=tl; r[1].text=ow; r[2].text=ac
            buf = BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== Batch prediction ======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")
tpl_buf = BytesIO()
pd.DataFrame(columns=TEMPLATE_COLUMNS).to_excel(tpl_buf, index=False)
tpl_buf.seek(0)
st.download_button("📥 Download Excel Template", tpl_buf, file_name="template_columns.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])
if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = raw.copy()
        for c in TEMPLATE_COLUMNS:
            if c not in df.columns: df[c] = 0
        for colmap, prefix in {
            "Gender": "gender",
            "Diagnosis": "diagnosis",
            "Has Social Worker": "has_social_worker",
            "Recent Self-harm": "has_recent_self_harm",
            "Self-harm During Admission": "self_harm_during_admission"
        }.items():
            if colmap in df.columns:
                for i, v in df[colmap].astype(str).items():
                    onehot = f"{prefix}_{v}"
                    if onehot in df.columns: df.at[i, onehot] = 1
        Xb = df[TEMPLATE_COLUMNS]
        base_probs = model.predict_proba(Xb, validate_features=False)[:, 1]
        adj_probs = [recalibrate_probability(Xb.iloc[i], base_probs[i]) for i in range(len(Xb))]
        out = raw.copy()
        out["risk_percent"] = (np.array(adj_probs)*100).round(1)
        out["risk_score_0_100"] = (np.array(adj_probs)*100).round().astype(int)
        out["risk_level"] = [classify(s) for s in out["risk_score_0_100"]]
        st.dataframe(out)
        buf_out = BytesIO(); out.to_csv(buf_out, index=False); buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
