import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score


df_orig = pd.read_csv('Baza customer Telecom v2.csv')  
df_orig['CHURN'] = df_orig['CHURN'].map({'No': 0, 'Yes': 1}).astype(int)

X_original = df_orig.drop(columns=['CHURN'])


drop_cols = ['PID', 'CRM_PID_Value_Segment', 'EffectiveSegment', 'KA_name']
X_original = X_original.drop(columns=drop_cols, errors='ignore')
y_original = df_orig['CHURN']


bst = xgb.Booster()
bst.load_model("xgb_model.json")


d_original = xgb.DMatrix(X_original)


y_pred_orig = (bst.predict(d_original) > 0.5).astype(int)
recall_orig = recall_score(y_original, y_pred_orig)

print(f"Recall on original data: {recall_orig:.4f}")
