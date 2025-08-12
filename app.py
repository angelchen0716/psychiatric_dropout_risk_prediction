import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

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

# Predict function

def score(df):
    df = df[columns]
    proba = model.predict_proba(df)[:,1]
    pred = (proba >= threshold).astype(int)
    return proba, pred

# Run predictions
if uploaded:
    data = pd.read_excel(uploaded)
    proba, pred = score(data)
    out = data.copy()
    out['risk_score'] = proba
    out['high_risk'] = pred
    st.subheader('Batch results')
    st.dataframe(out.head(25))
    csv = out.to_csv(index=False).encode('utf-8')
    st.download_button('Download results (CSV)', csv, file_name='predictions.csv')

# Single prediction & SHAP
st.subheader('Single‑patient result')
proba1, pred1 = score(single_df)
st.metric('Risk score (0–1)', f'{proba1[0]:.3f}', help='Probability of dropout within 90 days')
st.write('High‑risk flag at threshold', threshold, '→', bool(pred1[0]))

# Suggested next steps (actionable rules of thumb)
if pred1[0]==1:
    st.info('**Suggested actions:** schedule follow‑up within 7 days; enable appointment reminders; CM/peer support; address substance use; safety planning if self‑harm history; transportation support.')
else:
    st.success('**Suggested actions:** routine follow‑up in 2–4 weeks; digital reminders; education materials; reassess at any ED visit or missed appointment.')

if show_shap:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(single_df)
    st.write('### SHAP explanation (single case)')
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=columns, max_display=14)
    st.pyplot(bbox_inches='tight')

    st.write('### Global importance (sampled from manual + any uploaded)')
    sample = single_df
    if uploaded:
        sample = pd.concat([single_df, pd.read_excel(uploaded).head(200)], ignore_index=True)
    sv = explainer.shap_values(sample[columns])
    shap.summary_plot(sv, sample[columns], show=False)
    st.pyplot(bbox_inches='tight')

st.caption('Demo only. Trained on synthetic data shaped by peer‑reviewed literature. No clinical use.')
