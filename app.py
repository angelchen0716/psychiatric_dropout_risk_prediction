import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title='Psychiatric Dropout Risk', layout='wide')
st.title('Psychiatric Dropout Risk (Demo)')

# Load model
bundle = joblib.load('dropout_model.pkl')
model = bundle['model']
columns = bundle['columns']

# Sidebar controls
st.sidebar.header('Threshold & Options')
threshold = st.sidebar.slider('Risk threshold for HIGH risk', 0.05, 0.95, 0.5, 0.01)
show_shap = st.sidebar.checkbox('Show SHAP explanations', True)
export_plan = st.sidebar.checkbox('Enable care-plan download', True)

st.markdown('**Two input modes:** upload an Excel file or use the manual form below.')

# File upload
uploaded = st.file_uploader('Upload Excel (.xlsx) with column names matching the model', type=['xlsx'])

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

with st.expander('Manual input (single patient)'):
    vals = {}
    for col in columns:
        if col in ['age', 'recent_ed_visits_90d','inpatient_admits_1y']:
            vals[col] = st.number_input(col, value=int(defaults.get(col,0)), step=1)
        elif col in ['length_of_stay_last_admit','missed_appointment_ratio_6m']:
            vals[col] = st.number_input(col, value=float(defaults.get(col,0.0)))
        else:
            vals[col] = st.selectbox(col, [0,1], index=int(defaults.get(col,0)))
    single_df = pd.DataFrame([vals])

# Predict helpers

def score(df):
    df = df[columns]
    proba = model.predict_proba(df)[:,1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

# ---- Action recommender (rule-based, literature-informed) ---- #
ACTION_LIBRARY = {
    'self_harm_history': {
        1: ['Immediate safety planning with clinician', 'Provide crisis hotline info', 'Consider same-week psychiatry follow-up']
    },
    'dx_substance_use': {
        1: ['SBIRT referral / addiction services', 'Motivational interviewing session', 'Coordinate with dual-diagnosis care']
    },
    'alcohol_positive_test': {
        1: ['Brief alcohol intervention', 'Refer to community resources (AA/SMART)']
    },
    'recent_ed_visits_90d': lambda v: ['Assign care manager / assertive outreach', 'Create crisis plan', 'Warm handoff after ED'] if v >= 1 else [],
    'missed_appointment_ratio_6m': lambda v: ['Enroll in multi-channel reminders (SMS/phone)', 'Offer telehealth or same-day slots', 'Arrange transportation support'] if v >= 0.3 else [],
    'dx_depression': {1: ['Medication adherence check', 'Psychoeducation: depression & recovery']},
    'dx_bipolar': {1: ['Mood charting & early warning signs', 'Medication adherence & side-effect check']},
    'tobacco_dependence': {1: ['Smoking cessation referral', 'Offer NRT information']},
    'assault_injury_history': {1: ['Trauma-informed care referral', 'Screen for PTSD and safety at home']},
    'insurance_medicaid': {1: ['Social worker benefits review', 'Transportation / housing resources screen']},
}

RISK_TIER_RULES = [
    (lambda p, pred: p >= max(0.7, threshold), 'High', ['Follow-up ≤ 7 days', 'Care manager outreach within 48h']),
    (lambda p, pred: pred==1 or p >= max(0.5, threshold), 'Moderate-High', ['Follow-up 1–2 weeks', 'Reminder enrollment + barrier assessment']),
    (lambda p, pred: True, 'Lower', ['Follow-up 2–4 weeks', 'Education materials + reminders'])
]

def generate_actions(row: pd.Series, p: float, pred: int, shap_tuple=None):
    # Risk tier actions
    for rule, tier, actions in RISK_TIER_RULES:
        if rule(p, pred):
            tier_name = tier
            tier_actions = actions
            break
    # Feature-driven actions
    feature_actions = []
    for k, v in ACTION_LIBRARY.items():
        if callable(v):
            feature_actions += v(row.get(k, 0))
        else:
            choice = v.get(int(row.get(k, 0)), [])
            feature_actions += choice
    feature_actions = list(dict.fromkeys(feature_actions))  # dedupe, keep order

    # Why-these-actions (top positive SHAP drivers)
    drivers = []
    if shap_tuple is not None:
        shap_vals, feature_names, expected = shap_tuple
        order = np.argsort(-np.abs(shap_vals))
        for idx in order[:8]:
            name = feature_names[idx]
            val = row.get(name, None)
            contrib = shap_vals[idx]
            sign = '↑' if contrib > 0 else '↓'
            drivers.append(f"{name}={val} ({sign} risk)")
    return tier_name, tier_actions, feature_actions, drivers

# ------------------ Batch predictions ------------------ #
if uploaded:
    data = pd.read_excel(uploaded)
    proba, pred = score(data)
    out = data.copy()
    out['risk_score'] = proba
    out['high_risk'] = pred
    # Short, rule-only recommendations for batch
    short_recs = []
    for i, row in out.iterrows():
        actions = []
        for k, v in ACTION_LIBRARY.items():
            if callable(v):
                actions += v(row.get(k, 0))
            else:
                actions += v.get(int(row.get(k, 0)), [])
        actions = list(dict.fromkeys(actions))
        short_recs.append('; '.join(actions[:5]))
    out['recommendations'] = short_recs

    st.subheader('Batch results')
    st.dataframe(out.head(25))
    csv = out.to_csv(index=False).encode('utf-8')
    st.download_button('Download results (CSV)', csv, file_name='predictions.csv')

# ------------------ Single prediction & SHAP ------------------ #
st.subheader('Single‑patient result')
proba1, pred1 = score(single_df)
st.metric('Risk score (0–1)', f'{proba1[0]:.3f}', help='Probability of dropout within 90 days')
st.write('High‑risk flag at threshold', threshold, '→', bool(pred1[0]))

# SHAP for the single case (optional)
explainer = None
sv_single = None
if show_shap:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(single_df[columns])
    sv_single = sv[0]

# Action plan block
st.write('---')
st.write('### Suggested disposition & care plan')
row0 = single_df.iloc[0]
shap_info = (sv_single, columns, explainer.expected_value) if (show_shap and sv_single is not None) else None
risk_tier, tier_actions, feature_actions, drivers = generate_actions(row0, proba1[0], int(pred1[0]), shap_info)

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Disposition tier:** {risk_tier}")
    st.markdown('\n'.join([f'- {a}' for a in tier_actions]))
with col2:
    st.success('**Feature-driven actions**')
    if feature_actions:
        st.markdown('\n'.join([f'- {a}' for a in feature_actions]))
    else:
        st.write('No additional risk-specific actions suggested.')

with st.expander('Why these actions? (Top drivers for this case)'):
    if drivers:
        st.markdown('\n'.join([f'- {d}' for d in drivers]))
    else:
        st.write('Using threshold-based risk tiering only.')

# Visual SHAP
if show_shap:
    st.write('### SHAP explanation (single case)')
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_single, feature_names=columns, max_display=14)
    st.pyplot(bbox_inches='tight')

    st.write('### Global importance (sampled from manual + any uploaded)')
    sample = single_df
    if uploaded:
        sample = pd.concat([single_df, pd.read_excel(uploaded).head(200)], ignore_index=True)
    sv_global = explainer.shap_values(sample[columns])
    shap.summary_plot(sv_global, sample[columns], show=False)
    st.pyplot(bbox_inches='tight')

# Optional: export a plain-text care plan for documentation
if export_plan:
    plan = StringIO()
    print('Psychiatric Dropout Risk — Care Plan', file=plan)
    print(f"Risk score: {proba1[0]:.3f} (threshold {threshold}) — Tier: {risk_tier}", file=plan)
    print('\nTier actions:', file=plan)
    for a in tier_actions: print(f'- {a}', file=plan)
    print('\nFeature-driven actions:', file=plan)
    for a in feature_actions: print(f'- {a}', file=plan)
    if drivers:
        print('\nTop drivers:', file=plan)
        for d in drivers: print(f'- {d}', file=plan)
    st.download_button('Download care plan (.txt)', plan.getvalue().encode('utf-8'), file_name='care_plan.txt')

st.caption('Demo only. Trained on synthetic data shaped by peer‑reviewed literature. Not for clinical use.')
