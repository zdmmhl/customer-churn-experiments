import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)



df = pd.read_csv('processed_customer_data_improved.csv')



df['CHURN'] = df['CHURN'].astype(int)


X = df.drop(columns=['CHURN'])
y = df['CHURN']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


dummy_strat = DummyClassifier(strategy='stratified', random_state=42)
dummy_strat.fit(X_train, y_train)
y_pred = dummy_strat.predict(X_test)
y_prob = dummy_strat.predict_proba(X_test)[:, 1]


print("=== Stratified Random Baseline ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (yes):", precision_score(y_test, y_pred, pos_label=1, zero_division=0))
print("Recall    (yes):", recall_score(y_test, y_pred, pos_label=1, zero_division=0))
print("F1-score  (yes):", f1_score(y_test, y_pred, pos_label=1, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
