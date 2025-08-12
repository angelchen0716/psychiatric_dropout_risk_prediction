# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from io import BytesIO

st.set_page_config(page_title="Psychiatric Dropout Risk Predictor", layout="wide")

# =========================
# 模型載入或建立（找不到就用合成示範）
# =========================
@st.cache_resource
def load_or_build_model():
    try:
        bundle = joblib.load('dropout_model.pkl')
        return bundle['model'], bundle['columns']
    except FileNotFoundError:
        st.warning("⚠️ 未找到 dropout_model.pkl，已建立『合成示範模型』以供展示。")
        rng = np.random.default_rng(42)
        columns = [
            'age','sex_male','length_of_stay_last_admit','inpatient_admits_1y',
            'recent_ed_visits_90d','missed_appointment_ratio_6m',
            'dx_bipolar','dx_depression','dx_substance_use',
            'self_harm_history','assault_injury_history',
            'has_social_worker','post_discharge_followups',
            'medication_compliance_score','family_support_score',
            'insurance_medicaid'
        ]
        n = 2500
        X = pd.DataFrame(rng.integers(0, 2, size=(n, len(columns))), columns=columns)
        # 數值欄位合理化
        X['age'] = rng.integers(16, 85, n)
        X['length_of_stay_last_admit'] = rng.normal(3.5, 2.0, n).clip(0, 30)
        X['inpatient_admits_1y'] = rng.poisson(0.3, n).clip(0, 6)
        X['recent_ed_visits_90d'] = rng.poisson(0.6, n).clip(0, 10)
        X['missed_appointment_ratio_6m'] = rng.random(n)  # 0~1
        X['post_discharge_followups'] = rng.integers(0, 4, n)  # 0~3次
        X['medication_compliance_score'] = rng.normal(6.5, 2.0, n).clip(0, 10)  # 0~10
        X['family_support_score'] = rng.normal(5.0, 2.0, n).clip(0, 10)  # 0~10

        # 讓 y 與部分變數有關聯
        logit = (
            -2.2
            + 1.2 * X['self_harm_history']
            + 0.9 * X['dx_substance_use']
            + 0.6 * (X['recent_ed_visits_90d'] > 0)
            + 0.7 * (X['inpatient_admits_1y'] >= 1)
            + 1.1 * X['missed_appointment_ratio_6m']
            - 0.25 * X['medication_compliance_score']
            - 0.15 * X['family_support_score']
            - 0.3 * X['post_discharge_followups']
        )
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)

        model = XGBClassifier(
            n_estimators=300, max_depth=4, subsample=0.9, colsample_bytree=0.8,
            learning_rate=0.05, eval_metric='logloss', random_state=42
        )
        model.fit(X, y)
        return model, columns

model, columns = load_or_build_model()

# =========================
# 參數與工具
# =========================
BINARY_FIELDS_META = {
    'sex_male': ('Gender', {'Male (1)':1, 'Female (0)':0}),
    'dx_bipolar': ('Diagnosis: Bipolar', {'Yes (1)':1, 'No (0)':0}),
    'dx_depression': ('Diagnosis: Depression', {'Yes (1)':1, 'No (0)':0}),
    'dx_substance_use': ('Substance Use Dx', {'Yes (1)':1, 'No (0)':0}),
    'self_harm_history': ('Recent Self-harm', {'Yes (1)':1, 'No (0)':0}),
    'assault_injury_history': ('Assault/Injury History', {'Yes (1)':1, 'No (0)':0}),
    'has_social_worker': ('Has Social Worker', {'Yes (1)':1, 'No (0)':0}),
    'insurance_medicaid': ('Medicaid Insurance', {'Yes (1)':1, 'No (0)':0}),
}

MOD_CUT = 30  # Moderate 起點
HIGH_CUT = 50 # High 起點

def proba_to_percent(p: float) -> float:
    return float(p) * 100.0

def proba_to_score(p: float) -> int:
    return int(round(proba_to_percent(p)))

def classify(score: int) -> str:
    if score >= HIGH_CUT: return "High"
    if score >= MOD_CUT:  return "Moderate"
    return "Low"

# =========================
# 處置規則（時程｜負責｜動作）
# =========================
BASE_ACTIONS = {
    "High": [
        {"timeframe":"Today","owner":"Clinic scheduler","action":"Book return within 7 days (before discharge)."},
        {"timeframe":"Today","owner":"Social worker / Peer support","action":"Enroll in CM/peer support; create plan."},
        {"timeframe":"Today","owner":"Clinician","action":"Safety plan + crisis hotline; lethal means counseling."},
        {"timeframe":"Today","owner":"Addiction team","action":"If SUD positive: SBIRT and warm handoff today."},
        {"timeframe":"Today","owner":"Pharmacist","action":"Med rec + 7–14 day supply; consider LAI; blister pack."},
        {"timeframe":"48–72h","owner":"Care navigator","action":"Post‑discharge call to confirm meds/transport barriers."},
        {"timeframe":"T-3d / T-1d / T-2h","owner":"System","action":"Multi‑channel reminders (SMS + call + LINE)."},
        {"timeframe":"Scheduling","owner":"Transportation desk","action":"Arrange voucher/taxi and pickup time."},
    ],
    "Moderate": [
        {"timeframe":"1–2 weeks","owner":"Clinic scheduler","action":"Schedule return; offer evening/telehealth if needed."},
        {"timeframe":"T-2d / T-2h","owner":"System","action":"Activate reminders; address past no‑show barriers."},
        {"timeframe":"1 week","owner":"Care navigator","action":"Check-in call; troubleshoot transport/work/childcare."},
        {"timeframe":"Today","owner":"Clinician","action":"Education on relapse warning signs and adherence."},
    ],
    "Low": [
        {"timeframe":"2–4 weeks","owner":"Clinic scheduler","action":"Routine follow-up; consider group psychoeducation."},
        {"timeframe":"T-2d","owner":"System","action":"Standard reminder."},
        {"timeframe":"PRN","owner":"Clinician","action":"Escalate if early warning signs or no‑show."},
    ]
}

def personalized_actions(row: pd.Series):
    acts = []
    if int(row.get('self_harm_history',0)) == 1:
        acts += [
            {"timeframe":"Today","owner":"Clinician","action":"C-SSRS assessment; update safety plan; give wallet card."},
        ]
    if int(row.get('dx_substance_use',0)) == 1:
        acts += [
            {"timeframe":"Today","owner":"Addiction team","action":"AUDIT‑C/DAST; same‑day referral to addiction services."},
        ]
    if float(row.get('missed_appointment_ratio_6m',0)) >= 0.3:
        acts += [
            {"timeframe":"Scheduling","owner":"System + Transport","action":"Enhanced reminders + transport support; quick slots."},
        ]
    if int(row.get('recent_ed_visits_90d',0)) >= 1:
        acts += [
            {"timeframe":"72h","owner":"Care navigator","action":"ED→OP bridge within 72h; send summary to PCP."},
        ]
    return acts

# =========================
# 介面：左側 Patient Info（與你截圖風格類似）
# =========================
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 16, 90, 35)
    sex_choice = st.selectbox("Gender", list(BINARY_FIELDS_META['sex_male'][1].keys()))
    dx_choice = st.selectbox("Diagnosis", ["Bipolar","Depression","None"])
    los = st.slider("Length of Stay (days)", 0, 30, 3)
    prev_adm = st.slider("Previous Admissions", 0, 6, 1)
    has_sw_choice = st.radio("Has Social Worker", ("Yes","No"))
    med_comp = st.slider("Medication Compliance Score", 0.0, 10.0, 5.0, 0.1)
    recent_selfharm = st.radio("Recent Self-harm", ("Yes","No"))
    ed_visits = st.slider("ED visits (last 90 days)", 0, 10, 0)
    post_fu = st.slider("Post‑discharge follow-ups", 0, 3, 0)
    fam_support = st.slider("Family Support Score", 0.0, 10.0, 5.0, 0.1)
    missed_ratio = st.slider("Missed appt ratio (6m)", 0.0, 1.0, 0.0, 0.01)
    medicaid_choice = st.radio("Medicaid Insurance", ("Yes","No"))

# 將 sidebar 輸入組成 single_df（欄位名要對上模型）
single_row = {
    'age': age,
    'sex_male': BINARY_FIELDS_META['sex_male'][1][sex_choice],
    'length_of_stay_last_admit': float(los),
    'inpatient_admits_1y': int(prev_adm),
    'recent_ed_visits_90d': int(ed_visits),
    'missed_appointment_ratio_6m': float(missed_ratio),
    'dx_bipolar': 1 if dx_choice=="Bipolar" else 0,
    'dx_depression': 1 if dx_choice=="Depression" else 0,
    'dx_substance_use': 0,  # 可改成側欄開關；示範先 0
    'self_harm_history': 1 if recent_selfharm=="Yes" else 0,
    'assault_injury_history': 0,
    'has_social_worker': 1 if has_sw_choice=="Yes" else 0,
    'post_discharge_followups': int(post_fu),
    'medication_compliance_score': float(med_comp),
    'family_support_score': float(fam_support),
    'insurance_medicaid': 1 if medicaid_choice=="Yes" else 0
}
single_df = pd.DataFrame([single_row], columns=columns)

# =========================
# 主區塊：標題、預測、等級徽章
# =========================
st.title("🧠 Psychiatric Dropout Risk Predictor")

# 預測
proba = model.predict_proba(single_df[columns])[:,1][0]
score = proba_to_score(proba)
level = classify(score)

# 顯示結果
st.subheader("Predicted Dropout Risk (within 3 months)")
st.markdown(f"## **{proba_to_percent(proba):.1f}%**  &nbsp;|&nbsp; Score **{score}/100**")
badge_color = {"Low":"#16a34a","Moderate":"#f59e0b","High":"#ef4444"}[level]
st.markdown(
    f'<div style="background:{badge_color}22;border:1px solid {badge_color};'
    f'padding:10px;border-radius:10px;display:inline-block;color:#111;">'
    f'● {level} Risk</div>', unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SHAP 個案解釋
# =========================
with st.expander("SHAP Explanation", expanded=True):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(single_df[columns])
    base_value = explainer.expected_value
    # 兼容 xgboost 二元輸出
    if isinstance(base_value, (list, np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv, list):
            sv = sv[0]
    exp = shap.Explanation(
        values=sv[0],
        base_values=base_value,
        feature_names=columns,
        data=single_df[columns].iloc[0].values
    )
    shap.plots.waterfall(exp, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# =========================
# 建議處置：三欄卡片
# =========================
st.subheader("Recommended Actions")
def render_action_cards(rows):
    CARD_CSS = """
    <style>
    .card-grid{display:grid;grid-template-columns:160px 160px 1fr;gap:12px;}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:10px;margin:8px 0;}
    .tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#f1f5f9;
         font-size:12px;margin-right:6px;margin-bottom:6px;}
    .item{padding:8px;border-radius:10px;background:#f9fafb;border:1px dashed #e5e7eb;}
    </style>
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    for r in rows:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-grid">
                <div><span class="tag">⏱ Timeline</span><div class="item"><b>{r['timeframe']}</b></div></div>
                <div><span class="tag">👤 Owner</span><div class="item">{r['owner']}</div></div>
                <div><span class="tag">🛠 Action</span><div class="item">{r['action']}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

base_rows = BASE_ACTIONS[level].copy()
personal_rows = personalized_actions(single_df.iloc[0])
# 去重並保序
seen, rows = set(), []
for r in base_rows + personal_rows:
    key = (r['timeframe'], r['owner'], r['action'])
    if key not in seen:
        seen.add(key); rows.append(r)
render_action_cards(rows)

# =========================
# High 時：一鍵產生 SOP（TXT，免安裝新套件）
# =========================
if level == "High":
    def make_sop_txt(patient: pd.Series, score: int, label: str, action_rows: list) -> BytesIO:
        lines = []
        lines.append("Psychiatric Dropout Risk – Action SOP\n")
        lines.append(f"Risk score: {score}/100 | Risk level: {label}\n\n")
        lines.append("Actions (Timeline | Owner | Action)\n")
        for r in action_rows:
            lines.append(f"- {r['timeframe']} | {r['owner']} | {r['action']}")
        buf = BytesIO("\n".join(lines).encode("utf-8"))
        buf.seek(0)
        return buf
    st.info("⚠️ High 風險：可匯出處置 SOP（TXT）")
    st.download_button("⬇️ Export SOP (TXT)", make_sop_txt(single_df.iloc[0], score, level, rows),
                       file_name="dropout_risk_SOP.txt", mime="text/plain")

# =========================
# 批次：上傳 Excel，產出分數與等級 + 下載結果
# =========================
st.markdown("---")
st.subheader("Batch Prediction (Excel)")

# 下載欄位模板
def template_df():
    # 給使用者友善欄位（與模型一致）
    return pd.DataFrame(columns=columns)

tpl = template_df()
tpl_buf = BytesIO()
tpl.to_excel(tpl_buf, index=False)
tpl_buf.seek(0)
st.download_button("📥 Download Excel Template", tpl_buf, file_name="template_columns.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("📂 Upload Excel (columns must match the template)", type=["xlsx"])
if uploaded is not None:
    try:
        batch = pd.read_excel(uploaded)
        # 嘗試把 Yes/No 或 Male/Female 自動轉碼
        for col, meta in BINARY_FIELDS_META.items():
            label, mapping = meta
            if col in batch.columns:
                repl = {k.split(" (")[0]: v for k, v in mapping.items()}
                repl.update({"Yes":1,"No":0,"Male":1,"Female":0})
                batch[col] = batch[col].replace(repl)

        # 缺失欄位補齊（若使用者少放某些欄位）
        for c in columns:
            if c not in batch.columns:
                batch[c] = 0

        preds = model.predict_proba(batch[columns])[:,1]
        scores = (preds * 100).round().astype(int)
        labels = [classify(int(s)) for s in scores]
        out = batch.copy()
        out["risk_percent"] = (preds*100).round(1)
        out["risk_score_0_100"] = scores
        out["risk_level"] = labels

        st.success(f"完成批次預測：{len(out)} 筆")
        st.dataframe(out)

        out_buf = BytesIO()
        out.to_csv(out_buf, index=False).encode("utf-8")
        out_buf.seek(0)
        st.download_button("⬇️ Download Results (CSV)", out_buf, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"讀取或預測時發生錯誤：{e}")
