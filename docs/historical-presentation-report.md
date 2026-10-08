# Historical Customer Churn Study

An English report based on the team's COMP9444 presentation. The source describes 8,454 rows, 13 input attributes and a churn label. Raw customer records and presentation screenshots are not published.

## Planned and reported workflow

The presentation describes cleaning missing and malformed values, removing redundant columns, clipping outliers, categorical encoding, scaling, engineered revenue features, and SMOTE. It describes a stratified 64%/16%/20% train/validation/test split.

Models include logistic regression, decision trees, random forest, SVM, KNN, MLP, naive Bayes, XGBoost, LightGBM, CatBoost, Extra Trees, and voting ensembles. This is the team's reported study; the published repository contains selected scripts rather than every trained artifact.

## Historical metrics

| Metric | Baseline | Advanced |
|---|---:|---:|
| Accuracy | 0.9522 | 0.9548 |
| F1 | 0.9510 | 0.9537 |
| Precision | 0.9773 | 0.9768 |
| Recall | 0.9260 | 0.9317 |
| AUC | 0.9809 | 0.9809 |

The presentation itself reports no statistically significant difference, with p = 0.4094. The metric averaging, test population, and statistical-test method need clarification.

## Interpretation limits

The slides place preprocessing and SMOTE before splitting. If that ordering was used in the actual experiment, it introduces data leakage and makes the displayed evaluation optimistic. Transformations and resampling must be fit on training data only before trustworthy comparisons can be made.

The slides also translate predicted churn detection into “customers retained” and monetary gain. Those are assumed scenarios, not observed retention or revenue. No intervention experiment or causal business result is established. Slide-level revenue figures use different totals and should not be treated as reconciled financial evidence.

## Contribution and status

The deck assigns Jiawei Dong the motivation, problem statement, literature review, and conclusion presentation. That does not establish exclusive ownership of all model implementations. The study remains shared team work.

No models were retrained and no customer data were exposed during this documentation update. These historical metrics are preserved with their limitations rather than promoted as verified production performance.
