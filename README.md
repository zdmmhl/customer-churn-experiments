# Customer Churn Experiments

Historical exploratory analysis, preprocessing and XGBoost/T5 experiments for churn classification.

## Status: evaluation correction required

The historical workflow preprocesses and SMOTE-resamples the full dataset before splitting it. The main Notebook also contains this path. Metrics from that path should not be presented as leakage-free test performance.

See [evaluation plan](docs/evaluation-plan.md). The original computational workflow is retained so the issue can be inspected; Chinese commentary is omitted and two console messages are translated in this English edition. Dependencies are an unpinned starting list, not a reproduced environment.

## Contents

- `scripts/`: preprocessing and model scripts with the original computational workflow.
- `notebooks/churn-analysis.ipynb`: output-cleared main Notebook, not the duplicate under `pre`.
- No raw/processed customer CSV, Parquet or weights are included.


## Provenance

Originated in UNSW COMP9444 as team project material. Individual module attribution still needs confirmation. Source code is retained as an experimental archive; raw customer records, model checkpoints and Notebook outputs are excluded.

## Verification status

Python syntax was checked and Notebook outputs were cleared. Models were not retrained. The evaluation leakage described above remains unresolved; historical metrics are not evidence of generalization.


## Historical reports

The reports preserve saved coursework observations and team context. They are not fresh benchmark or runtime verification.

- [Historical presentation results and evaluation limits](docs/historical-presentation-report.md)
