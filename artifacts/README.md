# Saved experiment artifacts

This directory is preserved from the supplied original ZIP, with no synthetic
CSV scores or missing binary files added. Existing folders include:

- `deepseek_r1_qwen3_8b_tat/`: manifest metadata (raw `.npy` files absent)
- `rbf_gridsearch/`: nested-CV fold scores, OOF predictions, search reports
- `last_rbf_vs_lstm/`: RBF-SVM vs LSTM comparison results
- `tat_pca_ablation/`: pooling/classifier ablations
- `geometry_ablation/`: norm/cosine feature ablations
- `xgboost_comparison/`: XGBoost baseline results

Re-running experiments may create further output files within these folders.
This archive does not claim to include fitted models or original raw activations.
