import numpy as np
import pandas as pd
rng = np.random.default_rng(7)

N = 4000  # change as needed

# Demographics
age = rng.integers(16, 80, size=N)
sex = rng.choice([0,1], size=N)  # 0=female, 1=male

# Utilization & access
recent_ed_visits_90d = rng.poisson(0.5, size=N)
inpatient_admits_1y = rng.poisson(0.2, size=N)
length_of_stay_last_admit = np.where(inpatient_admits_1y>0, np.clip(rng.normal(4, 2, size=N), 0, None), 0)
missed_appointment_ratio_6m = np.clip(rng.normal(0.15, 0.15, size=N), 0, 1)
insurance_medicaid = rng.choice([0,1], p=[0.7,0.3], size=N)

# Diagnoses & events (binary)
dx_depression = rng.choice([0,1], p=[0.7,0.3], size=N)
dx_bipolar = rng.choice([0,1], p=[0.92,0.08], size=N)
dx_substance_use = rng.choice([0,1], p=[0.8,0.2], size=N)
self_harm_history = rng.choice([0,1], p=[0.9,0.1], size=N)
assault_injury_history = rng.choice([0,1], p=[0.92,0.08], size=N)

tobacco_dependence = rng.choice([0,1], p=[0.7,0.3], size=N)
alcohol_positive_test = rng.choice([0,1], p=[0.88,0.12], size=N)

# Older‑adult proxies (protective wrt FEP-ish risk profile)
med_statins = rng.choice([0,1], p=[0.7,0.3], size=N)
med_antihypertensives = rng.choice([0,1], p=[0.65,0.35], size=N)
thyroid_replacement = rng.choice([0,1], p=[0.9,0.1], size=N)
screening_mammography_recent = rng.choice([0,1], p=[0.85,0.15], size=N)
psa_recent = rng.choice([0,1], p=[0.87,0.13], size=N)

# Build a latent logit with literature‑guided directions
logit = (
    -3.0
    + 0.02 * (age - 35) * (age < 45)  # higher around young adult
    - 0.015 * (age - 55) * (age >= 45) # tapering in older
    + 0.15 * sex  # male slightly ↑
    + 0.35 * recent_ed_visits_90d
    + 0.40 * inpatient_admits_1y
    + 0.08 * length_of_stay_last_admit
    + 1.2  * missed_appointment_ratio_6m
    + 0.35 * dx_depression
    + 0.55 * dx_bipolar
    + 0.45 * dx_substance_use
    + 0.70 * self_harm_history
    + 0.35 * assault_injury_history
    + 0.25 * tobacco_dependence
    + 0.25 * alcohol_positive_test
    + 0.20 * insurance_medicaid
    - 0.30 * med_statins
    - 0.25 * med_antihypertensives
    - 0.20 * thyroid_replacement
    - 0.25 * screening_mammography_recent
    - 0.25 * psa_recent
)

p = 1 / (1 + np.exp(-logit))
dropout_within_90d = rng.binomial(1, p)

cols = {
    'age': age,
    'sex_male': sex,
    'recent_ed_visits_90d': recent_ed_visits_90d,
    'inpatient_admits_1y': inpatient_admits_1y,
    'length_of_stay_last_admit': length_of_stay_last_admit,
    'missed_appointment_ratio_6m': missed_appointment_ratio_6m,
    'dx_depression': dx_depression,
    'dx_bipolar': dx_bipolar,
    'dx_substance_use': dx_substance_use,
    'self_harm_history': self_harm_history,
    'assault_injury_history': assault_injury_history,
    'tobacco_dependence': tobacco_dependence,
    'alcohol_positive_test': alcohol_positive_test,
    'med_statins': med_statins,
    'med_antihypertensives': med_antihypertensives,
    'thyroid_replacement': thyroid_replacement,
    'screening_mammography_recent': screening_mammography_recent,
    'psa_recent': psa_recent,
    'insurance_medicaid': insurance_medicaid,
    'dropout_within_90d': dropout_within_90d
}

df = pd.DataFrame(cols)
df.to_excel('sample_patients.xlsx', index=False)
df.to_csv('synthetic_training.csv', index=False)
print('Wrote sample_patients.xlsx and synthetic_training.csv')
