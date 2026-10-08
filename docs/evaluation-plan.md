# Evaluation correction plan

1. Split the original records before fitting preprocessing or resampling.
2. Fit imputation, encoding, scaling and feature statistics only on training data.
3. Apply SMOTE only within training/fold data. Preserve the original validation/test class distribution.
4. Audit target-derived feature statistics and any hyperparameter/model selection against held-out boundaries.
5. Re-evaluate the baseline and advanced models with the same split and report class-sensitive metrics.
6. Record dataset provenance, permitted use, configuration and seeds before making performance claims.

See the [imbalanced-learn leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

This plan is not a completed correction. Historical scripts and the cleared Notebook remain for review. A synthetic input schema can be added once one preprocessing path is selected; no placeholder dataset is claimed to reproduce the historical study.
