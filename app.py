# app.py — Psychiatric Dropout Risk with fuzzy one-hot + Safety Override + SHAP + Actions + Batch
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

# ====== UI 選項 ======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== 載入模型或建立示範模型 ======
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
        if booster and getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    except Exception:
        pass
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
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
    st.warning("⚠️ 沒找到模型，建立合成示範模型")
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
    logit = -2 + 1.3*X["has_recent_self_harm_Yes"] + 0.9*X["self_harm_during_admission_Yes"]
    p = 1/(1+np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.06)
    model.fit(X, y)

# ====== 工具函數 ======
def set_onehot_by_prefix(df, prefix, value):
    """模糊匹配設定 one-hot 欄位值"""
    target = f"{prefix}_{value}".lower()
    for col in df.columns:
        if col.lower() == target:
            df.at[0, col] = 1
            return

MOD_CUT = 30
HIGH_CUT = 50
def proba_to_percent(p): return float(p) * 100
def proba_to_score(p): return int(round(proba_to_percent(p)))
def classify(score):
    if score >= HIGH_CUT: return "High"
    if score >= MOD_CUT:  return "Moderate"
    return "Low"

def has_flag(row, keyword):
    return any((row[col] == 1) for col in row.index if keyword in col)

# ====== Sidebar 輸入 ======
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 90, 35)
    gender = st.selectbox("Gender", GENDER_LIST)
    diagnosis = st.selectbox("Diagnosis", DIAG_LIST)
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("Previous Admissions (1y)", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", BIN_YESNO)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0)
    recent_self_harm = st.radio("Recent Self-harm", BIN_YESNO)
    selfharm_adm = st.radio("Self-harm During Admission", BIN_YESNO)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0)
    followups = st.slider("Post-discharge Followups", 0, 10, 2)

# ====== 建立 X_final 並更新 ======
X_final = pd.DataFrame(0, index=[0], columns=TEMPLATE_COLUMNS, dtype=float)
continuous_map = {
    "age": age,
    "length_of_stay": float(length_of_stay),
    "num_previous_admissions": int(num_adm),
    "medication_compliance_score": float(compliance),
    "family_support_score": float(support),
    "post_discharge_followups": int(followups),
}
for k, v in continuous_map.items():
    if k in X_final.columns:
        X_final.at[0, k] = v

set_onehot_by_prefix(X_final, "gender", gender)
set_onehot_by_prefix(X_final, "diagnosis", diagnosis)
set_onehot_by_prefix(X_final, "has_social_worker", social_worker)
set_onehot_by_prefix(X_final, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_final, "self_harm_during_admission", selfharm_adm)

# ====== 預測 + Safety Override ======
base_prob = model.predict_proba(X_final, validate_features=False)[0][1]
percent, score = proba_to_percent(base_prob), proba_to_score(base_prob)
level = classify(score)
override_reason = None
if has_flag(X_final.iloc[0], 'has_recent_self_harm'):
    percent, score, level = 70.0, 70, "High"
    override_reason = "recent self-harm"
elif has_flag(X_final.iloc[0], 'self_harm_during_admission'):
    percent, score, level = 70.0, 70, "High"
    override_reason = "in-hospital self-harm"

# ====== 顯示結果 ======
st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2 = st.columns(2)
with c1: st.metric("Probability", f"{percent:.1f}%")
with c2: st.metric("Risk Score (0–100)", f"{score}")
if override_reason:
    st.error(f"🔴 High Risk (safety override: {override_reason})")
    st.caption("ℹ️ This risk level is determined by a **clinical safety override** rule, not purely by the model's probability output.")
elif level == "High":
    st.error("🔴 High Risk")
elif level == "Moderate":
    st.warning("🟡 Moderate Risk")
else:
    st.success("🟢 Low Risk")

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
    if has_flag(row, 'has_recent_self_harm'):
        acts += [("Today","Clinician","C-SSRS assessment; update safety plan.")]
    if has_flag(row, 'self_harm_during_admission'):
        acts += [("Today","Clinician","Immediate psychiatric evaluation.")]
    return acts

rows = BASE_ACTIONS[level] + personalized_actions(X_final.iloc[0])
seen, uniq = set(), []
for r in rows:
    if r not in seen:
        seen.add(r)
        uniq.append(r)

# 三欄卡片顯示
c_timeline, c_owner, c_action = st.columns([1,1,3])
with c_timeline:
    st.markdown("**Timeline**")
    for tl, _, _ in uniq: st.write(tl)
with c_owner:
    st.markdown("**Owner**")
    for _, ow, _ in uniq: st.write(ow)
with c_action:
    st.markdown("**Action**")
    for _, _, ac in uniq: st.write(ac)

# ====== SOP export ======
if level == "High":
    def make_sop_txt(score: int, label: str, actions: list) -> BytesIO:
        lines = ["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}", ""]
        for (tl, ow, ac) in actions:
            lines.append(f"- {tl} | {ow} | {ac}")
        buf = BytesIO("\n".join(lines).encode("utf-8"))
        buf.seek(0)
        return buf

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
                r = t.add_row().cells
                r[0].text = tl; r[1].text = ow; r[2].text = ac
            buf = BytesIO()
            doc.save(buf)
            buf.seek(0)
            return buf

        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== Batch prediction ======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

# 提供模板下載
tpl_buf = BytesIO()
pd.DataFrame(columns=TEMPLATE_COLUMNS).to_excel(tpl_buf, index=False)
tpl_buf.seek(0)
st.download_button("📥 Download Excel Template", tpl_buf, file_name="template_columns.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 上傳批次檔
uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])
if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        df = raw.copy()

        # 確保模板欄位齊全
        for c in TEMPLATE_COLUMNS:
            if c not in df.columns:
                df[c] = 0

        # 人類可讀欄位轉 one-hot
        def apply_onehot_prefix(colname, prefix):
            if colname in df.columns:
                for i, v in df[colname].astype(str).items():
                    onehot = f"{prefix}_{v}"
                    for col in df.columns:
                        if col.lower() == onehot.lower():
                            df.at[i, col] = 1
        apply_onehot_prefix("Gender", "gender")
        apply_onehot_prefix("Diagnosis", "diagnosis")
        apply_onehot_prefix("Has Social Worker", "has_social_worker")
        apply_onehot_prefix("Recent Self-harm", "has_recent_self_harm")
        apply_onehot_prefix("Self-harm During Admission", "self_harm_during_admission")

        # 預測 + 安全覆蓋
        Xb = df[TEMPLATE_COLUMNS]
        base_probs = model.predict_proba(Xb, validate_features=False)[:, 1]
        adj_probs = []
        for i in range(len(Xb)):
            bp = base_probs[i]
            if has_flag(Xb.iloc[i], 'has_recent_self_harm') or has_flag(Xb.iloc[i], 'self_harm_during_admission'):
                bp = 0.70
            adj_probs.append(bp)

        out = raw.copy()
        out["risk_percent"] = (np.array(adj_probs) * 100).round(1)
        out["risk_score_0_100"] = (np.array(adj_probs) * 100).round().astype(int)
        out["risk_level"] = [classify(s) for s in out["risk_score_0_100"]]

        st.dataframe(out)

        buf_out = BytesIO()
        out.to_csv(buf_out, index=False)
        buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")

