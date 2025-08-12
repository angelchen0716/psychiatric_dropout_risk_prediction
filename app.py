# app.py — Psychiatric Dropout Risk (robust feature alignment + float32 + grouped SHAP)
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

# ====== 統一的特徵欄位模板（與左側語意一致）======
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

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0
    )
    model.fit(X, y)

# ====== 關鍵：模型特徵對齊 + float32 轉換 =======
def get_model_feature_order(m):
    """回傳模型訓練時的特徵名稱序（若不可得，則回傳 None 與期望長度）"""
    order = None
    exp_len = None
    try:
        booster = getattr(m, "get_booster", lambda: None)()
        if booster is not None:
            names = getattr(booster, "feature_names", None)
            if names:
                order = list(names)
                exp_len = len(order)
    except Exception:
        pass
    if order is None:
        # 某些情況只有 feature_names_in_ 或 n_features_in_
        if hasattr(m, "feature_names_in_"):
            order = list(m.feature_names_in_)
            exp_len = len(order)
        elif hasattr(m, "n_features_in_"):
            exp_len = int(m.n_features_in_)
    return order, exp_len

def align_df_to_model(df: pd.DataFrame, m):
    """將 df 對齊到模型需要的欄位序與長度；缺的補 0，多的丟掉；回傳 (aligned_df, used_feature_names)"""
    names, exp_len = get_model_feature_order(m)
    if names:
        # 以訓練特徵名為準
        aligned = pd.DataFrame(0, index=df.index, columns=names, dtype=np.float32)
        inter = [c for c in names if c in df.columns]
        aligned.loc[:, inter] = df[inter].astype(np.float32).values
        return aligned, names
    else:
        # 沒有名稱資訊但知道需要的長度 -> 若長度相符就原樣轉 float32；否則以現在 df 為準（可能仍會報錯，建議重訓）
        out = df.astype(np.float32)
        if (exp_len is not None) and (out.shape[1] != exp_len):
            # 嘗試截斷或補零到期望長度
            if out.shape[1] > exp_len:
                out = out.iloc[:, :exp_len]
            else:
                # 補零欄
                add = exp_len - out.shape[1]
                pad = pd.DataFrame(0, index=out.index, columns=[f"_pad_{i}" for i in range(add)], dtype=np.float32)
                out = pd.concat([out, pad], axis=1)
        return out, list(out.columns)

def to_float32_np(df: pd.DataFrame):
    return df.astype(np.float32).values

# ====== 工具函數（精準覆蓋）======
def set_onehot_by_prefix(df, prefix, value):
    target = f"{prefix}_{value}"
    if target in df.columns:
        df.at[0, target] = 1

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

# ====== 建立 X_final 並更新（與模板完全一致）======
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

# ====== 對齊到模型特徵並預測（validate_features=False + float32）======
X_aligned_df, used_names = align_df_to_model(X_final, model)
X_np = to_float32_np(X_aligned_df)
base_prob = model.predict_proba(X_np, validate_features=False)[:, 1][0]

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

# ====== SHAP（用對齊後矩陣計算，再映射回左側語意標籤）======
with st.expander("SHAP Explanation", expanded=True):
    # 以對齊後的欄位順序建立 explainer
    explainer = shap.TreeExplainer(model)
    # 注意：這裡用 numpy float32 輸入，避免 columnar 介面問題
    sv_raw = explainer.shap_values(X_np)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv_raw, list): sv_raw = sv_raw[0]
    sv_raw = sv_raw[0]  # (n_features_aligned,)

    # 把 shap 值對應回我們的 TEMPLATE_COLUMNS 名稱（若模型順序不同，以 used_names 映射）
    sv_map = dict(zip(used_names, sv_raw))
    # 連續特徵
    cont_list = [
        ("Age","age", X_final.at[0,"age"]),
        ("Length of Stay (days)","length_of_stay", X_final.at[0,"length_of_stay"]),
        ("Previous Admissions (1y)","num_previous_admissions", X_final.at[0,"num_previous_admissions"]),
        ("Medication Compliance (0–10)","medication_compliance_score", X_final.at[0,"medication_compliance_score"]),
        ("Family Support (0–10)","family_support_score", X_final.at[0,"family_support_score"]),
        ("Post-discharge Followups","post_discharge_followups", X_final.at[0,"post_discharge_followups"]),
    ]
    names, vals, data_vals = [], [], []
    for label, key, dv in cont_list:
        if key in sv_map:
            names.append(label); vals.append(sv_map[key]); data_vals.append(dv)

    # one-hot：只顯示被選中的那一個
    def add_onehot(title, prefix, value):
        col = f"{prefix}_{value}"
        if col in sv_map:
            names.append(f"{title}={value}")
            vals.append(sv_map[col])
            data_vals.append(1)
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

# ====== Batch prediction（Excel 與左側一致的語意欄位）======
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

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

        # 對齊到模型特徵並預測
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
            lambda s: "High" if s >= 50 else ("Moderate" if s >= 30 else "Low")
        )
        st.dataframe(out)

        buf_out = BytesIO(); out.to_csv(buf_out, index=False); buf_out.seek(0)
        st.download_button("⬇️ Download Results (CSV)", buf_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
