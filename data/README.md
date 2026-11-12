# Training and evaluation data

The original ZIP provided:

- `processed/math-robust-qwen-problems.csv`
  - **142 problem rows** with `problem` and `label` columns.
  - Class distribution: `True` (robust) = **36**, `False` (spurious) = **106**.
  - The unlabeled first CSV column is an exported row index, not a feature.
  - This is the provided labeled problem table; the source archive did not
    specify an external canonical download URL. We do not invent one.
- `../artifacts/deepseek_r1_qwen3_8b_tat/manifest.csv`
  - Extraction metadata and file paths for **142** problem examples.

## Preprocessing

`src/extractor.py` loads the source problems with the Transformers tokenizer,
applies its selected chat template, extracts residual states from each model
layer, and writes per-sample activations and token IDs. `src/lodder.py` reads
those activation files, constructs the trajectory of residual-state updates,
and exposes metadata and labels to the experiment scripts. Last-token layer
displacement trajectories are reduced by PCA; norm and adjacent-layer cosine
features may be appended according to the experiment configuration.

## Important: unavailable raw arrays

The original ZIP **does not contain any `.npy` activation or input-ID arrays**,
nor `extraction_config.json`. The supplied `manifest.csv` references those
missing arrays. They must be generated with `python src/extractor.py` before
running the full training/evaluation scripts. The original data and CSV
artifacts are not altered to disguise these missing dependencies.

## Splits and evaluation

The scripts use group-aware stratified cross-validation and seed **42**,
with an outer five-fold CV and three-fold inner search (see each experiment's
`Config` dataclass). They construct splits during execution; the archive did
not include a separate universal train/test split CSV. Saved fold-level
metrics and some out-of-fold predictions remain in `../artifacts/`.

## Data sources

The original data source is the user-provided `AIMO384729(1).zip` archive.
Specific external dataset provenance or license is not stated in that archive;
verify distribution permissions before publishing the problem texts publicly.
