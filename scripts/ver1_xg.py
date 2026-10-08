import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


df = pd.read_csv('processed_customer_data_plus_improved.csv')

#('processed_customer_data_sparse_improved.parquet')

df['CHURN'] = df['CHURN'].astype(int)
X = df.drop(columns=['CHURN'])
y = df['CHURN']


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)


dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.0,
    'verbosity': 1,
    'seed': 42
}


evals = [(dtrain, 'train'), (dval, 'validation')]
eval_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=evals,
    early_stopping_rounds=30,
    evals_result=eval_result,
    verbose_eval=10
)

model_path = "xgb_model.json"
bst.save_model(model_path)
print(f"\n✓ Model saved to {model_path}")



y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)


acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, zero_division=0)
rec   = recall_score(y_test, y_pred, zero_division=0)
f1    = f1_score(y_test, y_pred, zero_division=0)
roc_auc_score_val  = roc_auc_score(y_test, y_pred_prob)

print(f"Test Accuracy:     {acc:.4f}")
print(f"Precision (yes):   {prec:.4f}")
print(f"Recall    (yes):   {rec:.4f}")
print(f"F1-score  (yes):   {f1:.4f}")
print(f"ROC AUC:           {roc_auc_score_val:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))






plot_dir = 'plot'
os.makedirs(plot_dir, exist_ok=True)


results = eval_result
plt.figure(figsize=(8, 5))
plt.plot(results['train']['logloss'], label='Train')
plt.plot(results['validation']['logloss'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('Logloss')
plt.title('Training vs Validation Logloss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'logloss_curve.png'))
plt.close()


plt.figure(figsize=(10, 8))
xgb.plot_importance(bst, max_num_features=20, importance_type='gain', height=0.6)
plt.title('Top 20 Feature Importances (by Gain)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
plt.close()


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
plt.close()


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
plt.close()

print(f"\n✓ All plots saved to ./{plot_dir}/")