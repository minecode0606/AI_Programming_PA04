# Model checkpoints / download links

**No trained PCA + RBF-SVM `.joblib` probe checkpoint was supplied in the
original archive.** This directory intentionally contains no fabricated
weights. The paper's results CSV files are preserved in `../artifacts/`.

## Pretrained backbone (available)

- Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Original model and weights: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- Model files: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/tree/main

`src/extractor.py` downloads this pretrained backbone automatically via
Hugging Face Transformers on first use. It is **not** a trained robustness
classifier. Full BF16 model weights require approximately 16.4 GB of disk
space, plus additional space for the cache.

For an explicit local download (optional):

```bash
hf download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --local-dir checkpoints/backbone
```

The original extractor loads from Hugging Face's default cache, not from
`checkpoints/backbone`. This optional command only prepares local files;
change `ModelLoader.MODEL_ID` if you want to load from that local directory.

## Regenerate a trained probe

1. Run `python src/extractor.py` to generate activation arrays and extraction metadata.
2. Run `python src/grid_search_rbf.py` from the repository root to fit and evaluate.
3. The training script writes its fitted probe to
   `artifacts/rbf_gridsearch/final_best_model.joblib`.

Models saved by joblib may depend on the scikit-learn and Python versions.
Never unpickle checkpoints from an untrusted source.
