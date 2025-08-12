# app.py — Psychiatric Dropout Risk (fixed overrides + unified features + grouped SHAP)
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

# ====== 統一選項（左側/Excel/SHAP 同一套）======
DIAG_LIST = [
    "Schizophrenia","Bipolar","Depression","Personality Disorder",
    "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
]
BIN_YESNO = ["Yes","No"]
GENDER_LIST = ["Male","Female"]

# ====== 統一的特徵欄位模板（**固定，與左側語意一致**）======
TEMPLATE_COLUMNS = [
    "age","length_of_stay","num_previous_admissions",
    "medication_compliance_score","family_support_score","post_discharge_followups",
    "gender_Male","gender_Female",
] + [f"diagnosis_{d}" for d in DIAG_LIST] + [
    "has_social_worker_Yes","has_social_worker_No",
    "has_recent_self_harm_Yes","has_recent_self_harm_No",
    "self_harm_during_admission_Yes","self_harm_during_admission_No",
]

# ====== 載入模型（若無則以相同欄位訓練示範模型）======
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
    st.warning("⚠️ 沒找到模型，建立合成示範模型（與本頁欄位 100% 對齊）")
    rng = np.random.default_rng(42)
    n = 4000
    X = pd.DataFrame(0, index=range(n), columns=TEMPLATE_COLUMNS, dtype=float)
    # 連續型
    X["age"] = rng.integers(16, 85, n)
    X["length_of_stay"] = rng.normal(3.5, 2.0, n).clip(0, 30)
    X["num_previous_admissions"] = rng.poisson(0.4, n).clip(0, 8)
    X["medication_compliance_score"] = rng.normal(6.5, 2.0, n).clip(0, 10)
    X["family_support_score"] = rng.normal(5.0, 2.0, n).clip(0, 10)
    X["post_discharge_followups"] = rng.integers(0, 6, n)
    # one-hot 幫手
    def pick_one(prefix, options):
        idx = rng.integers(0, len(options), n)
        for i, opt in enumerate(options):
            X.loc[idx == i, f"{prefix}_{opt}"] = 1
    pick_one("gender", GENDER_LIST)
    pick_one("diagnosis", DIAG_LIST)
    pick_one("has_social_worker", BIN_YESNO)
    pick_one("has_recent_self_harm", BIN_YESNO)
    pick_one("self_harm_during_admission", BIN_YESNO)

    # 生成 y（放大近期/住院期間自傷的影響）
    logit = (
        -2.2
        + 1.6*X["has_recent_self_harm_Yes"]
        + 1.2*X["self_harm_during_admission_Yes"]
        + 0.35*(X["num_previous_admissions"]>=2)
        - 0.12*X["medication_compliance_score"]
        - 0.10*X["family_support_score"]
        - 0.08*X["post_discharge_followups"]
        + 0.02*(X["length_of_stay"])
    )
    p = 1/(1+np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    model = xgboost = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0
    )
    model.fit(X, y)

# ====== 工具函數（精準覆蓋）======
def set_onehot_by_prefix(df, prefix, value):
    """只把對應 prefix_value 設為 1，其他維持 0"""
    target = f"{prefix}_{value}"
    for col in df.columns:
        if col == target:
            df.at[0, col] = 1

def flag_yes(row, prefix):
    """只在 <prefix>_Yes == 1 時回傳 True（避免過去 'substring' 誤判）"""
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

# ====== Sidebar 輸入（Excel/SHAP 使用同語意）======
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

# ====== 建立 X_final 並更新（**與模板完全一致**）======
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
    X_final.at[0, k] = v

set_onehot_by_prefix(X_final, "gender", gender)
set_onehot_by_prefix(X_final, "diagnosis", diagnosis)
set_onehot_by_prefix(X_final, "has_social_worker", social_worker)
set_onehot_by_prefix(X_final, "has_recent_self_harm", recent_self_harm)
set_onehot_by_prefix(X_final, "self_harm_during_admission", selfharm_adm)

# ====== 預測 + Safety Override（**只看 _Yes 欄位**）======
base_prob = model.predict_proba(X_final)[:, 1][0]
percent, score = proba_to_percent(base_prob), proba_to_score(base_prob)
level = classify(score)
override_reason = None

if flag_yes(X_final.iloc[0], "has_recent_self_harm"):
    percent, score, level = 70.0, 70, "High"
    override_reason = "recent self-harm"
elif flag_yes(X_final.iloc[0], "self_harm_during_admission"):
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

# ====== SHAP（分組聚合：只顯示左側同語意標籤）======
with st.expander("SHAP Explanation", expanded=True):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_final)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv, list): sv = sv[0]
    sv = sv[0]  # (n_features,)

    # 連續特徵直接取值
    cont_feats = [
        ("Age","age", X_final.at[0,"age"]),
        ("Length of Stay (days)","length_of_stay", X_final.at[0,"length_of_stay"]),
        ("Previous Admissions (1y)","num_previous_admissions", X_final.at[0,"num_previous_admissions"]),
        ("Medication Compliance (0–10)","medication_compliance_score", X_final.at[0,"medication_compliance_score"]),
        ("Family Support (0–10)","family_support_score", X_final.at[0,"family_support_score"]),
        ("Post-discharge Followups","post_discharge_followups", X_final.at[0,"post_discharge_followups"]),
    ]
    names, vals, data_vals = [], [], []

    # 幫手：抓單一 one-hot 的 shap 值
    def add_onehot_group(title, prefix, value):
        col = f"{prefix}_{value}"
        if col in X_final.columns:
            idx = list(X_final.columns).index(col)
            names.append(f"{title}={value}")
            vals.append(sv[idx])
            data_vals.append(1)

    # 先放連續型
    for label, key, dv in cont_feats:
        idx = list(X_final.columns).index(key)
        names.append(label)
        vals.append(sv[idx])
        data_vals.append(dv)

    # 再放類別型（只顯示被選到的）
    add_onehot_group("Gender","gender", gender)
    add_onehot_group("Diagnosis","diagnosis", diagnosis)
    add_onehot_group("Has Social Worker","has_social_worker", social_worker)
    add_onehot_group("Recent Self-harm","has_recent_self_harm", recent_self_harm)
    add_onehot_group("Self-harm During Admission","self_harm_during_admission", selfharm_adm)

    # 依絕對值排序，最多顯示 12
    order = np.argsort(np.abs(np.array(vals)))[::-1][:12]
    exp = shap.Explanation(
        values=np.array(vals)[order],
        base_values=base_value,
        feature_names=[names[i] for i in order],
        data=np.array(data_vals)[order]
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
    st.markdown("**Timeline**")
    for tl, _, _ in uniq: st.write(tl)
with c_owner:
    st.markdown("**Owner**")
    for _, ow, _ in uniq: st.write(ow)
with c_action:
    st.markdown("**Action**")
    for _, _, ac in uniq: st.write(ac)

# ====== SOP export（High risk 才開放）======
if level == "High":
    def make_sop_txt(score: int, label: str, actions: list) -> BytesIO:
        lines = ["Psychiatric Dropout Risk – SOP", f"Risk score: {score}/100 | Risk level: {label}", ""]
        for (tl, ow, ac) in actions:
            lines.append(f"- {tl} | {ow} | {ac}")
        buf = BytesIO("\n".join(lines).encode("utf-8"))
        buf.seek(0); return buf
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
            buf = BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(score, level, uniq),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# ====== Batch prediction（**Excel 與左側完全一致的語意欄位**）======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

# 範本：提供「人類可讀欄位」供填寫
friendly_cols = [
    "Age","Gender","Diagnosis","Length of Stay (days)","Previous Admissions (1y)",
    "Has Social Worker","Medication Compliance Score (0–10)",
    "Recent Self-harm","Self-harm During Admission",
    "Family Support Score (0–10)","Post-discharge Followups"
]
tpl_df = pd.DataFrame(columns=friendly_cols)
tpl_buf = BytesIO()
tpl_df.to_excel(tpl_buf, index=False); tpl_buf.seek(0)
st.download_button("📥 Download Excel Template", tpl_buf, file_name="batch_template.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("📂 Upload Excel", type=["xlsx"])
if uploaded is not None:
    try:
        raw = pd.read_excel(uploaded)
        # 轉換到同一 TEMPLATE_COLUMNS
        df = pd.DataFrame(0, index=raw.index, columns=TEMPLATE_COLUMNS, dtype=float)

        # 連續與計數
        def safe_get(col, default=0):
            return raw[col] if col in raw.columns else default
        df["age"] = safe_get("Age")
        df["length_of_stay"] = safe_get("Length of Stay (days)")
        df["num_previous_admissions"] = safe_get("Previous Admissions (1y)")
        df["medication_compliance_score"] = safe_get("Medication Compliance Score (0–10)")
        df["family_support_score"] = safe_get("Family Support Score (0–10)")
        df["post_discharge_followups"] = safe_get("Post-discharge Followups")

        # one-hot 映射（大小寫/空白寬鬆）
        def apply_onehot_prefix(human_col, prefix, options):
            if human_col not in raw.columns: return
            for i, v in raw[human_col].astype(str).str.strip().items():
                # 容錯：不在 options 的值，跳過
                if v not in options: continue
                col = f"{prefix}_{v}"
                if col in df.columns: df.at[i, col] = 1

        apply_onehot_prefix("Gender","gender", GENDER_LIST)
        apply_onehot_prefix("Diagnosis","diagnosis", DIAG_LIST)
        apply_onehot_prefix("Has Social Worker","has_social_worker", BIN_YESNO)
        apply_onehot_prefix("Recent Self-harm","has_recent_self_harm", BIN_YESNO)
        apply_onehot_prefix("Self-harm During Admission","self_harm_during_admission", BIN_YESNO)

        # 預測 + 覆蓋（**只看 _Yes 欄位**）
        base_probs = model.predict_proba(df)[:, 1]
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
