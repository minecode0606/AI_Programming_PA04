# RBF-SVM Grid Search Report

## 1. Experimental setup

- Samples: 142
- Robust (1): 36
- Spurious (0): 106
- Unique groups: 138
- Input tensor: [142, 36, 4096] = [N, layers, hidden]
- Outer CV: 5-fold StratifiedGroupKFold
- Inner GridSearch CV: 3-fold StratifiedGroupKFold
- Grid PCA dimensions: [4, 8, 16, 32, 64]
- Grid C: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
- Grid gamma: ['scale', 1e-05, 0.0001, 0.001, 0.01, 0.1]
- Total parameter combinations: 180
- Refit metric: balanced accuracy
- Geometry: norm=True, cosine=True

## 2. Nested-CV unbiased performance

| Metric | Fold mean ± std | OOF |
|---|---:|---:|
| Accuracy | 0.6828 ± 0.0583 | 0.6831 |
| Balanced Accuracy | 0.7319 ± 0.0435 | 0.7327 |
| F1 | 0.5723 ± 0.0570 | 0.5714 |
| AUROC | 0.7670 ± 0.0569 | 0.7550 |

## 3. Final full-data GridSearch best configuration

- PCA dimension: **32**
- C: **1.0**
- gamma: **0.0001**
- Inner-CV balanced accuracy: **0.6993**
- Final feature dimension: **1223**
- PCA explained variance ratio sum: **0.9512**

## 4. Fold-wise selected parameters

| Fold | PCA | C | gamma | Inner balanced acc. |
|---:|---:|---:|---:|---:|
| 1 | 16 | 1.0 | 0.0001 | 0.6616 |
| 2 | 16 | 1.0 | 0.0001 | 0.7451 |
| 3 | 4 | 1.0 | 0.001 | 0.7533 |
| 4 | 8 | 1.0 | 0.001 | 0.7228 |
| 5 | 4 | 10.0 | 0.0001 | 0.7096 |

## 5. Interpretation

- The nested-CV OOF metrics are the main unbiased performance estimates.
- The full-data GridSearch result is intended for choosing the final model after evaluation, not for reporting unbiased test performance.
- If fold-wise best hyperparameters vary strongly, the model is likely sensitive to the small dataset and the selected values should be interpreted cautiously.
- Balanced accuracy is the refit criterion because the class distribution may be imbalanced.

## 6. Saved artifacts

- `nested_cv/fold_metrics.csv`
- `nested_cv/oof_predictions.csv`
- `nested_cv/best_params_by_fold.csv`
- `final_grid_search_results.csv`
- `final_best_params.json`
- `final_best_model.joblib`
- `top10_grid_configs.csv`
- `figures/grid_heatmap_pca_*.png`
- `figures/best_balanced_accuracy_by_pca.png`
- `figures/nested_oof_metrics.png`
- `figures/nested_oof_confusion_matrix.png`