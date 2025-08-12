import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from io import BytesIO
from docx import Document

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")

# ============= 模型載入或建立（無資料則用合成示範） =============
@st.cache_resource
def load_or_build_model():
    try:
        bundle = joblib.load('dropout_model.pkl')
        return bundle['model'], bundle['columns']
    except FileNotFoundError:
        st.warning("⚠️ 未找到模型檔，使用合成示範模型")
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

# ============= 二元欄位映射與說明 =============
BINARY_FIELDS = {
    'sex_male': {'label': 'Sex', 'choices': {'Male (1)': 1, 'Female (0)': 0}, 'help': 'Male=1, Female=0'},
    'dx_depression': {'label': 'Depression Dx', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'dx_bipolar': {'label': 'Bipolar Dx', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'dx_substance_use': {'label': 'Substance Use Dx', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'self_harm_history': {'label': 'Self-harm History', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'assault_injury_history': {'label': 'Assault/Injury History', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'tobacco_dependence': {'label': 'Tobacco Dependence', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'alcohol_positive_test': {'label': 'Alcohol Positive Test', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'med_statins': {'label': 'On Statins', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'med_antihypertensives': {'label': 'On Antihypertensives', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'thyroid_replacement': {'label': 'Thyroid Replacement', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'screening_mammography_recent': {'label': 'Recent Mammography', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'psa_recent': {'label': 'Recent PSA Test', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
    'insurance_medicaid': {'label': 'Medicaid Insurance', 'choices': {'Yes (1)': 1, 'No (0)': 0}},
}

# ============= 風險分數與分級 =============
def proba_to_score(p): return int(round(float(p) * 100))
mod_cut = 30
high_cut = 50
def classify_risk(score):
    if score >= high_cut: return "High"
    elif score >= mod_cut: return "Moderate"
    else: return "Low"

# ============= 三欄卡片 CSS =============
CARD_CSS = """
<style>
.card-grid {display:grid; grid-template-columns: 160px 160px 1fr; gap:12px;}
.card {background:#ffffff; border:1px solid #e6e6e6; border-radius:14px; padding:10px;}
.badge {padding:2px 8px; border-radius:999px; background:#f1f5f9; font-size:12px; margin-bottom:6px;}
.item {padding:6px; border-radius:10px; background:#f9fafb; border:1px dashed #eaeaea;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

def render_action_cards(rows):
    for r in rows:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-grid">
                <div><div class="badge">⏱ 時程</div><div class="item">{r['timeframe']}</div></div>
                <div><div class="badge">👤 負責角色</div><div class="item">{r['owner']}</div></div>
                <div><div class="badge">🛠 動作</div><div class="item">{r['action']}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

# ============= 處置規則（範例） =============
BASE_ACTIONS = {
    "High": [
        {"timeframe":"Today","owner":"Clinic scheduler","action":"預約 7 天內回診"},
        {"timeframe":"Today","owner":"Social worker","action":"納入個案管理"},
        {"timeframe":"Today","owner":"Clinician","action":"安全計畫與危機專線"}
    ],
    "Moderate": [
        {"timeframe":"1–2 weeks","owner":"Clinic scheduler","action":"預約回診"},
        {"timeframe":"T-2d","owner":"System","action":"啟用提醒服務"}
    ],
    "Low": [
        {"timeframe":"2–4 weeks","owner":"Clinic scheduler","action":"例行回診"}
    ]
}

def feature_actions_struct(row):
    acts = []
    if int(row.get('self_harm_history',0)) == 1:
        acts.append({"timeframe":"Today","owner":"Clinician","action":"C-SSRS 評估與安全計畫"})
    return acts

# ============= SOP 匯出（Word） =============
def build_sop_docx(patient_row, score, level, action_rows):
    doc = Document()
    doc.add_heading('Psychiatric Dropout Risk – SOP', level=1)
    doc.add_paragraph(f"Risk score: {score} | Risk level: {level}")
    t = doc.add_table(rows=1, cols=3)
    hdr = t.rows[0].cells
    hdr[0].text = 'Timeline'; hdr[1].text = 'Owner'; hdr[2].text = 'Action'
    for r in action_rows:
        row_cells = t.add_row().cells
        row_cells[0].text = r['timeframe']
        row_cells[1].text = r['owner']
        row_cells[2].text = r['action']
    buf = BytesIO(); doc.save(buf); buf.seek(0)
    return buf

# ============= 頁面 =============
st.title("精神科中斷治療風險評估系統")
uploaded_file = st.file_uploader("📂 上傳 Excel（欄位需與模型一致）", type=["xlsx"])
show_shap = st.sidebar.checkbox("顯示 SHAP 解釋", True)

# 手動輸入
with st.expander("📝 單筆輸入", expanded=True):
    vals = {}
    vals['age'] = st.number_input('Age', value=40, step=1)
    vals['recent_ed_visits_90d'] = st.number_input('ED visits (90d)', value=0, step=1)
    vals['inpatient_admits_1y'] = st.number_input('Inpatient admits (1y)', value=0, step=1)
    vals['length_of_stay_last_admit'] = st.number_input('LOS last admit (days)', value=0.0)
    vals['missed_appointment_ratio_6m'] = st.number_input('Missed appt ratio (0–1)', value=0.0, min_value=0.0, max_value=1.0)
    for col in [c for c in columns if c in BINARY_FIELDS]:
        meta = BINARY_FIELDS[col]
        choice = st.selectbox(meta['label'], list(meta['choices'].keys()), help=meta.get('help',''))
        vals[col] = meta['choices'][choice]
    single_df = pd.DataFrame([vals])

# 預測
def predict(df):
    proba = model.predict_proba(df[columns])[:,1]
    scores = np.array([proba_to_score(p) for p in proba])
    levels = [classify_risk(s) for s in scores]
    return scores, levels, proba

# 批次
if uploaded_file:
    batch_df = pd.read_excel(uploaded_file)
    scores, levels, probas = predict(batch_df)
    batch_df["風險分數(0-100)"] = scores
    batch_df["風險等級"] = levels
    st.dataframe(batch_df)

# 單筆
scores_single, level_single, proba_single = predict(single_df)
st.subheader("單筆結果")
st.metric("風險分數", scores_single[0])
st.metric("風險等級", level_single[0])

# 處置卡片
actions = BASE_ACTIONS[level_single[0]] + feature_actions_struct(single_df.iloc[0].to_dict())
st.write("**建議處置**")
render_action_cards(actions)

# High → SOP 匯出
if level_single[0] == "High":
    sop_buf = build_sop_docx(single_df.iloc[0], scores_single[0], level_single[0], actions)
    st.download_button("⬇️ 匯出 SOP（Word）", sop_buf, file_name="SOP.docx")

# SHAP 圖
if show_shap:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(single_df[columns])
    base_value = explainer.expected_value
    if isinstance(base_value,(list,np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv, list):
            sv = sv[0]
    exp = shap.Explanation(values=sv[0], base_values=base_value,
                           feature_names=columns, data=single_df[columns].iloc[0].values)
    shap.plots.waterfall(exp, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
