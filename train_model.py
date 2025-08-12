import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('synthetic_training.csv')
X = df.drop(columns=['dropout_within_90d'])
y = df['dropout_within_90d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model (stable and fast)
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Eval
proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
print('ROC AUC:', auc)

# ROC plot
fpr, tpr, thr = roc_curve(y_test, proba)
plt.figure()
plt.plot(fpr, tpr, label=f'XGBoost (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'--',alpha=.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Dropout within 90 days')
plt.legend()
plt.tight_layout()
plt.savefig('roc.png', dpi=200)

# Confusion matrix at default threshold 0.5
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred)
print('Confusion matrix:\n', cm)
with open('classification_report.txt','w') as f:
    f.write(classification_report(y_test, pred))

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=200)

# Single waterfall for the first test case
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[0],
    feature_names=X_test.columns.tolist(),
    max_display=14
)
plt.tight_layout()
plt.savefig('shap_waterfall_case0.png', dpi=200)

# Save model & columns
joblib.dump({'model': model, 'columns': X.columns.tolist()}, 'dropout_model.pkl')
print('Saved: dropout_model.pkl, roc.png, shap_summary.png, shap_waterfall_case0.png, classification_report.txt')
