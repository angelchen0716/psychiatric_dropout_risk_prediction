# ✅ psychiatric_dropout demo App（加：處置卡片 + SOP 匯出 + 漂亮SHAP waterfall）
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from io import BytesIO

# 可選：若環境有安裝 python-docx 就能輸出 Word
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")

# -----------------------------
# 載入模型與欄位樣板
# -----------------------------
model = joblib.load("dropout_model.pkl")          # {'model':..., 'columns':...} 或 直接是xgb model，依你保存方式
sample = pd.read_csv("sample_input.csv")          # 訓練時的一熱(One-hot)欄位模板

# -----------------------------
# 分級（顯示%與0–100分）
# -----------------------------
MOD_CUT = 30   # Moderate 起點
HIGH_CUT = 50  # High 起點
def proba_to_percent(p: float) -> float: return float(p) * 100.0
def proba_to_score(p: float) -> int:     return int(round(proba_to_percent(p)))
def classify(score: int) -> str:
    if score >= HIGH_CUT: return "High"
    if score >= MOD_CUT:  return "Moderate"
    return "Low"

# -----------------------------
# （可選）臨床校正：確保自傷等關鍵事件能拉高風險
# -----------------------------
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(p):   return np.log(np.clip(p,1e-8,1-1e-8)/np.clip(1-p,1e-8,1-1e-8))
def recalibrate_probability(row: pd.Series, base_prob: float) -> float:
    z = _logit(base_prob)
    # 依需要調整權重
    if int(row.get('has_recent_self_harm_Yes', 0)) == 1: z += 1.5
    if int(row.get('self_harm_during_admission_Yes', 0)) == 1: z += 1.0
    return float(_sigmoid(z))

# -----------------------------
# 側邊輸入（延續你現有的選項，可再加更多）
# -----------------------------
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 90, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])

    diagnosis = st.selectbox("Diagnosis", [
        "Schizophrenia","Bipolar","Depression","Personality Disorder",
        "Substance Use Disorder","Dementia","Anxiety","PTSD","OCD","ADHD","Other/Unknown"
    ])

    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("# Previous Admissions (1y)", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", ["Yes", "No"], horizontal=True)
    compliance = st.slider("Medication Compliance Score (0–10)", 0.0, 10.0, 5.0, 0.1)
    recent_self_harm = st.radio("Recent Self-harm", ["Yes", "No"], horizontal=True)
    selfharm_adm = st.radio("Self-harm During Admission", ["Yes", "No"], horizontal=True)
    support = st.slider("Family Support Score (0–10)", 0.0, 10.0, 5.0, 0.1)
    followups = st.slider("Post-discharge Followups (booked)", 0, 10, 2)

# -----------------------------
# 建立單筆 X，對齊 sample 欄位
# -----------------------------
X_final = pd.DataFrame(columns=sample.columns); X_final.loc[0] = 0
# 連續/計數欄位（依你的 sample 欄位名）
cont = {
    'age': age,
    'length_of_stay': float(length_of_stay),
    'num_previous_admissions': int(num_adm),
    'medication_compliance_score': float(compliance),
    'family_support_score': float(support),
    'post_discharge_followups': int(followups),
}
for k,v in cont.items():
    if k in X_final.columns: X_final.at[0,k] = v

# one-hot
for col in [
    f'gender_{gender}',
    f'diagnosis_{diagnosis}',
    f'has_social_worker_{social_worker}',
    f'has_recent_self_harm_{recent_self_harm}',
    f'self_harm_during_admission_{selfharm_adm}',
]:
    if col in X_final.columns: X_final.at[0,col] = 1

# -----------------------------
# 預測 → 臨床校正 → 顯示
# -----------------------------
base_prob = model.predict_proba(X_final, validate_features=False)[0][1]
adj_prob  = recalibrate_probability(X_final.iloc[0], base_prob)  # 若不想校正就用 base_prob
percent   = proba_to_percent(adj_prob)
score     = proba_to_score(adj_prob)
level     = classify(score)

st.subheader("Predicted Dropout Risk (within 3 months)")
c1, c2 = st.columns(2)
with c1: st.metric("Probability", f"{percent:.1f}%")
with c2: st.metric("Risk Score (0–100)", f"{score}")
if level == "High":      st.error("🔴 High Risk")
elif level == "Moderate":st.warning("🟡 Moderate Risk")
else:                    st.success("🟢 Low Risk")

st.markdown("---")

# -----------------------------
# SHAP：單筆 waterfall（與你截圖相同風格）
# -----------------------------
with st.expander("SHAP Explanation", expanded=True):
    # 用 TreeExplainer 取值，再組 Explanation 以畫 waterfall
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_final)
    base_value = explainer.expected_value
    # 兼容 xgboost 二元輸出
    if isinstance(base_value,(list,np.ndarray)) and not np.isscalar(base_value):
        base_value = base_value[0]
        if isinstance(sv,list):
            sv = sv[0]
    exp = shap.Explanation(
        values=sv[0],
        base_values=base_value,
        feature_names=list(X_final.columns),
        data=X_final.iloc[0].values
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

# -----------------------------
# 處置卡片（時程｜負責角色｜動作）
# -----------------------------
st.subheader("Recommended Actions")
CARD_CSS = """
<style>
.card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:12px;margin:8px 0;}
.grid{display:grid;grid-template-columns:160px 160px 1fr;gap:12px;}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#f1f5f9;
     font-size:12px;margin-bottom:6px;}
.item{padding:8px;border-radius:10px;background:#f9fafb;border:1px dashed #e5e7eb;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

BASE_ACTIONS = {
    "High": [
        ("Today","Clinic scheduler","Book return within 7 days (before discharge)."),
        ("Today","Social worker / Peer support","Enroll in CM/peer support; create plan."),
        ("Today","Clinician","Safety plan + crisis hotline; lethal means counseling."),
        ("Today","Addiction team","If SUD positive: SBIRT and same‑day referral."),
        ("Today","Pharmacist","Med rec + 7–14 day supply; consider LAI; blister pack."),
        ("48–72h","Care navigator","Post‑discharge call; confirm meds/transport barriers."),
        ("T‑3d / T‑1d / T‑2h","System","Multi‑channel reminders (SMS + call + LINE)."),
        ("Scheduling","Transportation desk","Arrange voucher/taxi and pickup time.")
    ],
    "Moderate": [
        ("1–2 weeks","Clinic scheduler","Schedule return; offer evening/telehealth."),
        ("T‑2d / T‑2h","System","Activate reminders; address past no‑show barriers."),
        ("1 week","Care navigator","Check‑in call; troubleshoot transport/work/childcare."),
        ("Today","Clinician","Education on relapse signs and adherence.")
    ],
    "Low": [
        ("2–4 weeks","Clinic scheduler","Routine follow‑up; consider group psychoeducation."),
        ("T‑2d","System","Standard reminder."),
        ("PRN","Clinician","Escalate if early warning signs or no‑show.")
    ]
}

def personalized_actions(row: pd.Series):
    acts = []
    if int(row.get('has_recent_self_harm_Yes',0))==1:
        acts += [("Today","Clinician","C‑SSRS assessment; update safety plan; give wallet card.")]
    if int(row.get('self_harm_during_admission_Yes',0))==1:
        acts += [("Today","Clinician","Inpatient incident review; tighten safety plan; notify team.")]
    return acts

base_rows = BASE_ACTIONS[level][:]
pers_rows = personalized_actions(X_final.iloc[0])
# 合併去重
seen, rows = set(), []
for t in base_rows + pers_rows:
    if t not in seen:
        seen.add(t); rows.append(t)

def render_card(timeline, owner, action):
    st.markdown(
        f"""<div class="card">
               <div class="grid">
                  <div><span class="tag">⏱ Timeline</span><div class="item"><b>{timeline}</b></div></div>
                  <div><span class="tag">👤 Owner</span><div class="item">{owner}</div></div>
                  <div><span class="tag">🛠 Action</span><div class="item">{action}</div></div>
               </div>
            </div>""",
        unsafe_allow_html=True
    )
for tl, ow, ac in rows: render_card(tl, ow, ac)

# -----------------------------
# High：一鍵匯出 SOP（TXT 恒有效；DOCX 視環境提供）
# -----------------------------
if level == "High":
    st.info("⚠️ High 風險：可一鍵匯出處置 SOP")
    # TXT
    def make_sop_txt(patient_row: pd.Series, score: int, label: str, actions: list) -> BytesIO:
        lines = [
            "Psychiatric Dropout Risk – Action SOP",
            f"Risk score: {score}/100 | Risk level: {label}",
            "",
            "Actions (Timeline | Owner | Action)"
        ]
        for (tl, ow, ac) in actions:
            lines.append(f"- {tl} | {ow} | {ac}")
        buf = BytesIO("\n".join(lines).encode("utf-8")); buf.seek(0); return buf
    st.download_button("⬇️ Export SOP (TXT)", make_sop_txt(X_final.iloc[0], score, level, rows),
                       file_name="dropout_risk_SOP.txt", mime="text/plain")

    # DOCX（若環境有 python-docx）
    if HAS_DOCX:
        def make_sop_docx(patient_row: pd.Series, score: int, label: str, actions: list) -> BytesIO:
            doc = Document()
            doc.add_heading('Psychiatric Dropout Risk – Action SOP', level=1)
            doc.add_paragraph(f"Risk score: {score}/100 | Risk level: {label}")
            doc.add_heading('Actions', level=2)
            t = doc.add_table(rows=1, cols=3)
            hdr = t.rows[0].cells; hdr[0].text='Timeline'; hdr[1].text='Owner'; hdr[2].text='Action'
            for (tl, ow, ac) in actions:
                r = t.add_row().cells; r[0].text=tl; r[1].text=ow; r[2].text=ac
            buf = BytesIO(); doc.save(buf); buf.seek(0); return buf
        st.download_button("⬇️ Export SOP (Word)", make_sop_docx(X_final.iloc[0], score, level, rows),
                           file_name="dropout_risk_SOP.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.caption("（要匯出 Word：在 requirements.txt 加上 `python-docx` 後重新部署即可）")

st.caption("Model trained on simulated data. For demonstration only; not for clinical use.")
