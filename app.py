# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")

# ======================
# 模型載入或建立
# ======================
@st.cache_resource
def load_or_build_model():
    try:
        bundle = joblib.load('dropout_model.pkl')
        return bundle['model'], bundle['columns']
    except FileNotFoundError:
        st.warning("⚠️ 未找到模型檔，已自動建立『合成示範模型』供展示使用。")
        rng = np.random.default_rng(42)
        columns = [
            'age','sex_male','recent_ed_visits_90d','inpatient_admits_1y',
            'length_of_stay_last_admit','missed_appointment_ratio_6m',
            'dx_depression','dx_bipolar','dx_substance_use','self_harm_history',
            'assault_injury_history','tobacco_dependence','alcohol_positive_test',
            'med_statins','med_antihypertensives','thyroid_replacement',
            'screening_mammography_recent','psa_recent','insurance_medicaid'
        ]
        n = 2000
        X = pd.DataFrame(rng.integers(0, 2, size=(n, len(columns))), columns=columns)
        X['age'] = rng.integers(16, 80, n)
        X['recent_ed_visits_90d'] = rng.poisson(0.5, n)
        X['inpatient_admits_1y'] = rng.poisson(0.2, n)
        X['length_of_stay_last_admit'] = rng.normal(3, 2, n).clip(0)
        X['missed_appointment_ratio_6m'] = rng.random(n)
        # y 與特徵有關聯
        logit = (
            -2 + 0.8 * X['missed_appointment_ratio_6m'] +
            0.4 * X['dx_substance_use'] +
            0.5 * (X['recent_ed_visits_90d'] > 0) +
            0.6 * X['self_harm_history']
        )
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        return model, columns

model, columns = load_or_build_model()

# ======================
# 處置規則
# ======================
ACTION_RULES = {
    "High": ["7 天內回診", "個案管理/同儕支持", "多渠道提醒", "處理物質使用問題", "安全計畫", "交通支援"],
    "Moderate": ["1–2 週內回診", "啟用提醒服務", "檢視就醫阻礙"],
    "Low": ["2–4 週內回診", "教育資料", "例行提醒"]
}

FEATURE_ACTIONS = {
    'self_harm_history': {1: ["安全計畫", "危機資源提供"]},
    'dx_substance_use': {1: ["成癮治療轉介"]},
    'missed_appointment_ratio_6m': lambda v: ["啟用提醒服務", "交通支援"] if v >= 0.3 else []
}

# ======================
# 頁面標題與輸入方式
# ======================
st.title("精神科中斷治療風險評估系統")
threshold = st.sidebar.slider("高風險門檻", 0.05, 0.95, 0.5, 0.01)
show_shap = st.sidebar.checkbox("顯示 SHAP 解釋", True)

uploaded_file = st.file_uploader("📂 上傳 Excel 檔（欄位需與模型一致）", type=["xlsx"])

# 手動輸入
with st.expander("📝 單筆資料輸入"):
    vals = {}
    for col in columns:
        if col in ['age','recent_ed_visits_90d','inpatient_admits_1y']:
            vals[col] = st.number_input(col, value=0, step=1)
        elif col in ['length_of_stay_last_admit','missed_appointment_ratio_6m']:
            vals[col] = st.number_input(col, value=0.0)
        else:
            vals[col] = st.selectbox(col, [0,1], index=0)
    single_df = pd.DataFrame([vals])

# ======================
# 預測函式
# ======================
def predict(df):
    proba = model.predict_proba(df[columns])[:,1]
    risk_level = []
    for p in proba:
        if p >= threshold:
            risk_level.append("High")
        elif p >= 0.3:
            risk_level.append("Moderate")
        else:
            risk_level.append("Low")
    return proba, risk_level

# ======================
# 批次處理
# ======================
if uploaded_file:
    batch_df = pd.read_excel(uploaded_file)
    proba, risk_levels = predict(batch_df)
    batch_df["風險分數"] = proba
    batch_df["風險等級"] = risk_levels
    st.subheader("📊 批次結果")
    st.dataframe(batch_df)
    st.download_button("下載結果 CSV", batch_df.to_csv(index=False).encode("utf-8"), "predictions.csv")

# ======================
# 單筆結果
# ======================
st.subheader("單筆預測結果")
proba_single, level_single = predict(single_df)
st.metric("風險分數 (0–1)", f"{proba_single[0]:.3f}")
st.write("風險等級：", level_single[0])

# 處置建議
actions = ACTION_RULES[level_single[0]]
for feat, rule in FEATURE_ACTIONS.items():
    if callable(rule):
        actions += rule(single_df.iloc[0][feat])
    else:
        actions += rule.get(int(single_df.iloc[0][feat]), [])
actions = list(dict.fromkeys(actions))
st.write("建議處置：")
st.markdown("\n".join([f"- {a}" for a in actions]))

# SHAP
if show_shap:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(single_df[columns])
    st.write("### SHAP 個案解釋")
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        feature_names=columns,
        data=single_df.iloc[0]
    ))
