import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(y_true, y_proba, label='Model'):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0,1],[0,1],'--',alpha=.6)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.tight_layout()
