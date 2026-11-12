# XGBoost Comparison Report

## Nested-CV performance

| Method | Balanced Accuracy | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|
| PCA + RBF-SVM | 0.7319 ± 0.0435 | 0.5723 ± 0.0570 | 0.7670 ± 0.0569 | 0.6828 ± 0.0583 |
| XGBoost | 0.6108 ± 0.1228 | 0.3626 ± 0.2712 | 0.7580 ± 0.0432 | 0.7404 ± 0.0809 |
| PCA + XGBoost | 0.6336 ± 0.1001 | 0.4340 ± 0.1708 | 0.7309 ± 0.0396 | 0.7392 ± 0.0704 |

## OOF performance

| Method | Balanced Accuracy | F1 | AUROC | Accuracy |
|---|---:|---:|---:|---:|
| PCA + RBF-SVM | 0.7327 | 0.5714 | 0.7550 | 0.6831 |
| XGBoost | 0.6053 | 0.3934 | 0.7474 | 0.7394 |
| PCA + XGBoost | 0.6329 | 0.4478 | 0.7248 | 0.7394 |

## Final full-data GridSearch best parameters

### XGBoost

- experiment: **01_last_xgboost**
- cv_balanced_accuracy: **0.6010582010582011**
- feature_dim: **147527**
- colsample_bytree: **0.5**
- learning_rate: **0.1**
- max_depth: **2**
- n_estimators: **300**
- subsample: **0.8**

### PCA + XGBoost

- experiment: **02_last_pca_xgboost**
- cv_balanced_accuracy: **0.6425925925925925**
- feature_dim: **647**
- n_components: **16**
- colsample_bytree: **1.0**
- learning_rate: **0.1**
- max_depth: **2**
- n_estimators: **100**
- subsample: **0.8**
- pca_explained_variance_ratio_sum: **0.9369271397590637**

## Interpretation note

- Nested outer-fold performance is the unbiased comparison result.
- Full-data GridSearch scores are for final hyperparameter/model selection, not held-out test performance.
- Both XGBoost variants use identical outer folds and identical geometry features; the primary difference is whether PCA is applied before XGBoost.