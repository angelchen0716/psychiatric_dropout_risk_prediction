import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import StringIO
import os
import urllib.request
import os

st.set_page_config(page_title='Psychiatric Dropout Risk', layout='wide')
st.title('Psychiatric Dropout Risk (Demo)')

# --- Model loading with robust fallbacks ---
FALLBACK_COLUMNS = [
    'age','sex_male','recent_ed_visits_90d','inpatient_admits_1y',
    'length_of_stay_last_admit','missed_appointment_ratio_6m',
    'dx_depression','dx_bipolar','dx_substance_use',
    'self_harm_history','assault_injury_history',
    'tobacco_dependence','alcohol_positive_test',
    'med_statins','med_antihypertensives','thyroid_replacement',
    'screening_mammography_recent','psa_recent','insurance_medicaid',
]

@st.cache_resource
def load_or_build_model():
    # 1) Try local artifact
    for p in ['dropout_model.pkl','models/dropout_model.pkl','artifacts/dropout_model.pkl']:
        if os.path.exists(p):
            return joblib.load(p) | {'_origin': f'local:{p}'}

    # 2) Try remote download via secret (set MODEL_URL in Streamlit secrets)
    try:
        url = st.secrets.get('MODEL_URL', None)
        if url:
            with urllib.request.urlopen(url) as r:
                blob = r.read()
            import io
            bundle = joblib.load(io.BytesIO(blob))
            bundle['_origin'] = 'download:MODEL_URL'
            return bundle
    except Exception:
        pass

    # 3) Build a small synthetic model so the demo still runs
    from xgboost import XGBClassifier
    rng = np.random.default_rng(42)
    n = 3000
    X = pd.DataFrame(0, index=np.arange(n), columns=FALLBACK_COLUMNS)
    # numeric features
    X['age'] = rng.integers(16, 80, n)
    X['recent_ed_visits_90d'] = rng.poisson(0.4, n)
    X['inpatient_admits_1y'] = rng.poisson(0.2, n)
    X['length_of_stay_last_admit'] = rng.gamma(2.0, 1.5, n)
    X['missed_appointment_ratio_6m'] = rng.uniform(0, 0.8, n)
    # binary features
    def bern(p):
        return (rng.random(n) < p).astype(int)
    X['sex_male'] = bern(0.5)
    X['dx_depression'] = bern(0.25)
    X['dx_bipolar'] = bern(0.06)
    X['dx_substance_use'] = bern(0.12)
    X['self_harm_history'] = bern(0.05)
    X['assault_injury_history'] = bern(0.05)
    X['tobacco_dependence'] = bern(0.25)
    X['alcohol_positive_test'] = bern(0.08)
    X['med_statins'] = bern(0.20)
    X['med_antihypertensives'] = bern(0.20)
    X['thyroid_replacement'] = bern(0.10)
    X['screening_mammography_recent'] = bern(0.20)
    X['psa_recent'] = bern(0.20)
    X['insurance_medicaid'] = bern(0.20)

    # outcome mechanism (heuristic, literature‑informed)
    logit = (
        -2.2
        + 0.9 * X['missed_appointment_ratio_6m']
        + 0.5 * (X['recent_ed_visits_90d'] >= 1).astype(int)
        + 0.6 * X['dx_substance_use']
        + 0.4 * X['dx_depression']
        + 0.7 * X['self_harm_history']
        + 0.3 * (X['inpatient_admits_1y'] > 0).astype(int)
        + 0.4 * X['insurance_medicaid']
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < proba).astype(int)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        eval_metric='logloss',
    )
    clf.fit(X, y)
    bundle = {'model': clf, 'columns': FALLBACK_COLUMNS, '_origin': 'synthetic-built'}
    try:
        joblib.dump(bundle, 'dropout_model.pkl')
    except Exception:
        pass
    return bundle

bundle = load_or_build_model()
model = bundle['model']
columns = bundle['columns']
if bundle.get('_origin') == 'synthetic-built':
    st.warning('⚠️ 未找到模型檔，已自動建立「合成示範模型」。如需正式結果，請上傳或部署真正的 `dropout_model.pkl`。')

# Sidebar controls
st.sidebar.header('Threshold & Options')
threshold = st.sidebar.slider('高風險門檻', 0.05, 0.95, 0.5, 0.01)
show_shap = st.sidebar.checkbox('顯示 SHAP 解釋圖', True)
export_plan = st.sidebar.checkbox('啟用處置計畫下載', True)

st.markdown('**兩種輸入方式：** 上傳 Excel 或使用下方手動輸入表單。')

# File upload
uploaded = st.file_uploader('上傳欄位名稱與模型相符的 Excel 檔', type=['xlsx'])

# Manual form defaults
defaults = {
    'age': 28, 'sex_male': 1,
    'recent_ed_visits_90d': 1, 'inpatient_admits_1y': 0,
    'length_of_stay_last_admit': 0, 'missed_appointment_ratio_6m': 0.2,
    'dx_depression': 1, 'dx_bipolar': 0, 'dx_substance_use': 0,
    'self_harm_history': 0, 'assault_injury_history': 0,
    'tobacco_dependence': 0, 'alcohol_positive_test': 0,
    'med_statins': 0, 'med_antihypertensives': 0, 'thyroid_replacement': 0,
    'screening_mammography_recent': 0, 'psa_recent': 0,
    'insurance_medicaid': 0,
}

with st.expander('手動輸入（單一病人）'):
    vals = {}
    for col in columns:
        if col in ['age', 'recent_ed_visits_90d','inpatient_admits_1y']:
            vals[col] = st.number_input(col, value=int(defaults.get(col,0)), step=1)
        elif col in ['length_of_stay_last_admit','missed_appointment_ratio_6m']:
            vals[col] = st.number_input(col, value=float(defaults.get(col,0.0)))
        else:
            vals[col] = st.selectbox(col, [0,1], index=int(defaults.get(col,0)))
    single_df = pd.DataFrame([vals])

def score(df):
    df = df[columns]
    proba = model.predict_proba(df)[:,1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

# 建議處置（中文）
ACTION_LIBRARY = {
    'self_harm_history': {1: ['立即與臨床醫師制定安全計畫', '提供自殺防治專線資訊', '考慮同週安排精神科回診']},
    'dx_substance_use': {1: ['轉介成癮治療/戒癮服務', '安排動機式晤談', '協調雙重診斷照護']},
    'alcohol_positive_test': {1: ['簡短酒精介入', '轉介至社區戒酒資源（AA/SMART）']},
    'recent_ed_visits_90d': lambda v: ['安排個管/積極外展', '制定危機計畫', '急診出院後即時交接'] if v >= 1 else [],
    'missed_appointment_ratio_6m': lambda v: ['多渠道提醒（簡訊/電話）', '提供遠距醫療或當日門診', '協助交通安排'] if v >= 0.3 else [],
    'dx_depression': {1: ['檢查藥物服從性', '提供憂鬱症與復原的心理教育']},
    'dx_bipolar': {1: ['情緒紀錄與預警徵象教育', '藥物服從性與副作用檢查']},
    'tobacco_dependence': {1: ['轉介戒菸服務', '提供尼古丁替代療法資訊']},
    'assault_injury_history': {1: ['轉介創傷知情照護', '篩檢 PTSD 與家庭安全']},
    'insurance_medicaid': {1: ['社工檢視福利資源', '評估交通與住房需求']},
}

RISK_TIER_RULES = [
    (lambda p, pred: p >= max(0.7, threshold), '高風險', ['7 天內回診', '48 小時內個管聯絡']),
    (lambda p, pred: pred==1 or p >= max(0.5, threshold), '中高風險', ['1–2 週內回診', '納入提醒系統 + 評估就醫障礙']),
    (lambda p, pred: True, '低風險', ['2–4 週內例行追蹤', '提供衛教與提醒'])
]

# ...（後續程式與原本相同，僅更換文案為中文）
