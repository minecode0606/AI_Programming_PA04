from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install it with: pip install xgboost"
    ) from exc

from lodder import ActivationLoader


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    artifact_dir: str = "artifacts/deepseek_r1_qwen3_8b_tat"
    output_dir: str = "artifacts/xgboost_comparison"

    # 이전 PCA+RBF-SVM GridSearch 결과가 있으면 자동으로 비교표에 추가
    rbf_baseline_summary: str = "artifacts/rbf_gridsearch/summary.json"

    mode: str = "tat"
    n_layers: int | None = None

    # Nested CV
    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42

    # PCA + XGBoost에서 탐색할 PCA dimension
    pca_dims: Tuple[int, ...] = (8, 16, 32, 64)

    # XGBoost Grid: 2 x 2 x 2 x 2 x 2 = 32 combinations
    xgb_n_estimators: Tuple[int, ...] = (100, 300)
    xgb_max_depth: Tuple[int, ...] = (2, 4)
    xgb_learning_rate: Tuple[float, ...] = (0.03, 0.1)
    xgb_subsample: Tuple[float, ...] = (0.8, 1.0)
    xgb_colsample_bytree: Tuple[float, ...] = (0.5, 1.0)

    # Fixed XGBoost regularization
    xgb_min_child_weight: float = 1.0
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0

    # 기존 Last+PCA+RBF와 조건을 맞추기 위해 둘 다 유지
    use_norm: bool = True
    use_cosine: bool = True

    pca_iterated_power: int = 5

    # CPU parallelism
    grid_n_jobs: int = 10
    xgb_n_jobs: int = 1
    grid_verbose: int = 1

    save_outer_models: bool = False


@dataclass(frozen=True)
class Experiment:
    name: str
    use_pca: bool


EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment(name="01_last_xgboost", use_pca=False),
    Experiment(name="02_last_pca_xgboost", use_pca=True),
)


# ============================================================
# Reproducibility / metadata
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy_float32(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def get_labels(loader: ActivationLoader) -> np.ndarray:
    labels = []
    for i in range(len(loader)):
        labels.append(int(bool(loader.get_metadata(i)["label"])))
    return np.asarray(labels, dtype=np.int64)


def get_groups(loader: ActivationLoader) -> np.ndarray:
    """같은 problem_hash를 같은 fold에 넣어 leakage를 방지합니다."""
    groups = []
    for i in range(len(loader)):
        meta = loader.get_metadata(i)
        group = meta.get("problem_hash")
        if group is None or (isinstance(group, float) and np.isnan(group)):
            group = meta["sample_id"]
        groups.append(str(group))
    return np.asarray(groups)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) == 2
        else float("nan"),
    }

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


# ============================================================
# Load LAST-token TaT trajectory
# ============================================================

def resolve_n_layers(loader: ActivationLoader, config: Config) -> int:
    if config.n_layers is not None:
        return int(config.n_layers)
    if hasattr(loader, "num_layers"):
        return int(loader.num_layers)
    raise AttributeError(
        "Could not infer num_layers. Set Config.n_layers manually, e.g. 36."
    )


def tat_to_last(x: np.ndarray, n_layers: int) -> np.ndarray:
    """
    [T*L, D] -> [T,L,D] -> 마지막 token [L,D]
    """
    x = to_numpy_float32(x)
    if x.ndim != 2:
        raise ValueError(f"Expected [T*L,D], got {x.shape}")

    m, hidden = x.shape
    if m % n_layers != 0:
        raise ValueError(
            f"Trajectory length={m} is not divisible by n_layers={n_layers}."
        )

    n_tokens = m // n_layers
    grid = x.reshape(n_tokens, n_layers, hidden)
    return grid[-1].astype(np.float32, copy=False)


def preload_last(
    loader: ActivationLoader,
    config: Config,
) -> Tuple[np.ndarray, List[str], int]:
    n_layers = resolve_n_layers(loader, config)
    xs: List[np.ndarray] = []
    sample_ids: List[str] = []
    expected_shape = None

    print(f"Preloading LAST-token trajectory | n_layers={n_layers}")

    for i in range(len(loader)):
        sample = loader.load_sample(i, mode=config.mode, as_torch=True)
        x_last = tat_to_last(sample["x"], n_layers=n_layers)

        if expected_shape is None:
            expected_shape = x_last.shape
        elif x_last.shape != expected_shape:
            raise ValueError(
                f"Inconsistent sample shape at {i}: {x_last.shape} != {expected_shape}"
            )

        xs.append(x_last)
        sample_ids.append(str(sample["sample_id"]))

        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            print(f"  loaded {i + 1}/{len(loader)}")

    X = np.stack(xs, axis=0).astype(np.float32, copy=False)
    print("X_last:", X.shape, "(N, layers, hidden)")
    return X, sample_ids, n_layers


# ============================================================
# Shared geometry features
# ============================================================

def geometry_parts(
    X: np.ndarray,
    use_norm: bool,
    use_cosine: bool,
) -> List[np.ndarray]:
    parts: List[np.ndarray] = []

    if use_norm:
        parts.append(
            np.linalg.norm(X, axis=-1).astype(np.float32, copy=False)
        )

    if use_cosine:
        eps = 1e-8
        prev = X[:, :-1, :]
        nxt = X[:, 1:, :]
        numerator = np.sum(prev * nxt, axis=-1)
        denominator = (
            np.linalg.norm(prev, axis=-1)
            * np.linalg.norm(nxt, axis=-1)
            + eps
        )
        parts.append((numerator / denominator).astype(np.float32, copy=False))

    return parts


# ============================================================
# Feature transformers
# ============================================================

class RawLastGeometryTransformer(BaseEstimator, TransformerMixin):
    """
    XGBoost without PCA.

    [N,L,D] -> flatten [N,L*D] + norm [N,L] + cosine [N,L-1]

    For L=36,D=4096:
      36*4096 + 36 + 35 = 147527 features.
    """

    def __init__(self, use_norm: bool = True, use_cosine: bool = True):
        self.use_norm = use_norm
        self.use_cosine = use_cosine

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected [N,L,D], got {X.shape}")
        self.n_layers_ = int(X.shape[1])
        self.hidden_size_ = int(X.shape[2])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        parts = [X.reshape(n, -1)]
        parts.extend(
            geometry_parts(X, self.use_norm, self.use_cosine)
        )
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


class PCALastGeometryTransformer(BaseEstimator, TransformerMixin):
    """
    PCA + XGBoost.

    [N,L,D] -> PCA D->P -> [N,L,P] -> flatten [N,L*P] + geometry.

    PCA is inside the sklearn pipeline, so GridSearchCV fits PCA only on each
    inner-training split. This prevents validation leakage.
    """

    def __init__(
        self,
        n_components: int = 32,
        use_norm: bool = True,
        use_cosine: bool = True,
        random_state: int = 42,
        iterated_power: int = 5,
    ):
        self.n_components = n_components
        self.use_norm = use_norm
        self.use_cosine = use_cosine
        self.random_state = random_state
        self.iterated_power = iterated_power

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected [N,L,D], got {X.shape}")

        self.n_layers_ = int(X.shape[1])
        self.hidden_size_ = int(X.shape[2])

        rows = X.reshape(-1, X.shape[-1])
        max_allowed = min(rows.shape[0] - 1, rows.shape[1])
        if int(self.n_components) > max_allowed:
            raise ValueError(
                f"PCA n_components={self.n_components} > max_allowed={max_allowed}"
            )

        self.pca_ = PCA(
            n_components=int(self.n_components),
            svd_solver="randomized",
            random_state=int(self.random_state),
            iterated_power=int(self.iterated_power),
        )
        self.pca_.fit(rows)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        n, l, d = X.shape

        Z = self.pca_.transform(
            X.reshape(-1, d)
        ).astype(np.float32, copy=False)
        Z = Z.reshape(n, l, int(self.n_components))

        parts = [Z.reshape(n, -1)]
        parts.extend(
            geometry_parts(X, self.use_norm, self.use_cosine)
        )
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


# ============================================================
# Pipeline / GridSearchCV
# ============================================================

def make_xgb_classifier(config: Config, seed: int) -> XGBClassifier:
    # CPU version. tree_method='hist' is efficient for this setting.
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=config.xgb_n_jobs,
        min_child_weight=config.xgb_min_child_weight,
        reg_lambda=config.xgb_reg_lambda,
        reg_alpha=config.xgb_reg_alpha,
        verbosity=0,
    )


def make_pipeline(
    experiment: Experiment,
    config: Config,
    seed: int,
) -> Pipeline:
    if experiment.use_pca:
        feature = PCALastGeometryTransformer(
            n_components=32,
            use_norm=config.use_norm,
            use_cosine=config.use_cosine,
            random_state=seed,
            iterated_power=config.pca_iterated_power,
        )
    else:
        feature = RawLastGeometryTransformer(
            use_norm=config.use_norm,
            use_cosine=config.use_cosine,
        )

    return Pipeline(
        [
            ("feature", feature),
            ("model", make_xgb_classifier(config, seed)),
        ]
    )


def make_param_grid(experiment: Experiment, config: Config) -> Dict:
    grid = {
        "model__n_estimators": list(config.xgb_n_estimators),
        "model__max_depth": list(config.xgb_max_depth),
        "model__learning_rate": list(config.xgb_learning_rate),
        "model__subsample": list(config.xgb_subsample),
        "model__colsample_bytree": list(config.xgb_colsample_bytree),
    }

    if experiment.use_pca:
        grid["feature__n_components"] = list(config.pca_dims)

    return grid


def make_grid_search(
    experiment: Experiment,
    config: Config,
    seed: int,
) -> GridSearchCV:
    inner_cv = StratifiedGroupKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=seed,
    )

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "auroc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
    }

    return GridSearchCV(
        estimator=make_pipeline(experiment, config, seed),
        param_grid=make_param_grid(experiment, config),
        scoring=scoring,
        refit="balanced_accuracy",
        cv=inner_cv,
        n_jobs=config.grid_n_jobs,
        verbose=config.grid_verbose,
        return_train_score=False,
        error_score="raise",
    )


# ============================================================
# Nested CV
# ============================================================

def run_experiment(
    experiment: Experiment,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: List[str],
    outer_splits,
    config: Config,
    output_dir: Path,
) -> Dict:
    exp_dir = output_dir / experiment.name
    nested_dir = exp_dir / "nested_cv"
    nested_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    pred_rows = []
    best_param_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print("\n" + "=" * 100)
        print(f"{experiment.name} | OUTER FOLD {fold}/{config.outer_splits}")
        print("=" * 100)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train = groups[train_idx]

        # class imbalance 대응
        train_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        search = make_grid_search(
            experiment,
            config,
            seed=config.seed + 1000 * fold,
        )

        search.fit(
            X_train,
            y_train,
            groups=g_train,
            model__sample_weight=train_weights,
        )

        grid_df = pd.DataFrame(search.cv_results_)
        grid_df.to_csv(
            nested_dir / f"fold_{fold:02d}_grid_results.csv",
            index=False,
        )

        best_params = search.best_params_
        best_model = search.best_estimator_

        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)

        feature_dim = int(
            best_model.named_steps["feature"].transform(X_train[:1]).shape[1]
        )

        pca_dim = None
        pca_variance = None
        if experiment.use_pca:
            pca_dim = int(best_params["feature__n_components"])
            pca_variance = float(
                np.sum(
                    best_model.named_steps["feature"]
                    .pca_
                    .explained_variance_ratio_
                )
            )

        metrics.update(
            {
                "experiment": experiment.name,
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "feature_dim": feature_dim,
                "pca_dim": pca_dim,
                "pca_explained_variance": pca_variance,
                "inner_best_balanced_accuracy": float(search.best_score_),
            }
        )
        fold_rows.append(metrics)

        best_row = {
            "fold": fold,
            "inner_balanced_accuracy": float(search.best_score_),
        }
        for key, value in best_params.items():
            best_row[
                key.replace("model__", "").replace("feature__", "")
            ] = value
        best_param_rows.append(best_row)

        y_pred = (y_prob >= 0.5).astype(int)
        for idx, true, prob, pred in zip(test_idx, y_test, y_prob, y_pred):
            pred_rows.append(
                {
                    "experiment": experiment.name,
                    "fold": fold,
                    "sample_id": sample_ids[int(idx)],
                    "group": groups[int(idx)],
                    "y_true": int(true),
                    "probability": float(prob),
                    "y_pred": int(pred),
                }
            )

        print("Best params:", best_params)
        print(f"inner_bal_acc={search.best_score_:.4f}")
        print(
            f"OUTER TEST | acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | auroc={metrics['auroc']:.4f}"
        )

        if config.save_outer_models:
            joblib.dump(
                best_model,
                nested_dir / f"fold_{fold:02d}_best_model.joblib",
            )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    best_param_df = pd.DataFrame(best_param_rows)

    fold_df.to_csv(nested_dir / "fold_metrics.csv", index=False)
    pred_df.to_csv(nested_dir / "oof_predictions.csv", index=False)
    best_param_df.to_csv(nested_dir / "best_params_by_fold.csv", index=False)

    oof = compute_metrics(
        pred_df["y_true"].to_numpy(),
        pred_df["probability"].to_numpy(),
    )

    return {
        "experiment": experiment,
        "fold_df": fold_df,
        "pred_df": pred_df,
        "best_param_df": best_param_df,
        "oof": oof,
    }


# ============================================================
# Final full-data GridSearch
# ============================================================

def run_final_grid_search(
    experiment: Experiment,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Config,
    output_dir: Path,
) -> Dict:
    exp_dir = output_dir / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 100)
    print(f"FINAL FULL-DATA GRIDSEARCH: {experiment.name}")
    print("#" * 100)

    weights = compute_sample_weight(class_weight="balanced", y=y)

    search = make_grid_search(
        experiment,
        config,
        seed=config.seed + 999_999,
    )

    search.fit(
        X,
        y,
        groups=groups,
        model__sample_weight=weights,
    )

    grid_df = pd.DataFrame(search.cv_results_)
    grid_df.to_csv(exp_dir / "final_grid_search_results.csv", index=False)

    best_model = search.best_estimator_
    best_params = search.best_params_

    feature_dim = int(
        best_model.named_steps["feature"].transform(X[:1]).shape[1]
    )

    best = {
        "experiment": experiment.name,
        "cv_balanced_accuracy": float(search.best_score_),
        "feature_dim": feature_dim,
    }

    for key, value in best_params.items():
        best[
            key.replace("model__", "").replace("feature__", "")
        ] = value

    if experiment.use_pca:
        best["pca_explained_variance_ratio_sum"] = float(
            np.sum(
                best_model.named_steps["feature"]
                .pca_
                .explained_variance_ratio_
            )
        )

    with (exp_dir / "final_best_params.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    joblib.dump(best_model, exp_dir / "final_best_model.joblib")

    print("FINAL BEST:", best)
    return {"best": best, "grid_df": grid_df, "model": best_model}


# ============================================================
# Summary helpers
# ============================================================

def summarize_experiment(result: Dict) -> Dict:
    exp = result["experiment"]
    fold_df = result["fold_df"]
    oof = result["oof"]

    row = {"experiment": exp.name}
    for metric in ["accuracy", "balanced_accuracy", "f1", "auroc"]:
        values = fold_df[metric].to_numpy(dtype=float)
        row[f"fold_{metric}_mean"] = float(np.nanmean(values))
        row[f"fold_{metric}_std"] = float(
            np.nanstd(values, ddof=1) if len(values) > 1 else 0.0
        )
        row[f"oof_{metric}"] = float(oof[metric])
    return row


def load_rbf_baseline(summary_path: str) -> Dict | None:
    """이전 RBF GridSearch summary.json이 존재하면 비교표에 추가합니다."""
    path = Path(summary_path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fold_summary = data["nested_cv_fold_summary"]
        oof = data["nested_cv_oof"]

        row = {"experiment": "00_last_pca_rbf_svm"}
        for metric in ["accuracy", "balanced_accuracy", "f1", "auroc"]:
            row[f"fold_{metric}_mean"] = float(fold_summary[metric]["mean"])
            row[f"fold_{metric}_std"] = float(fold_summary[metric]["std"])
            row[f"oof_{metric}"] = float(oof[metric])
        return row
    except Exception as exc:
        print(f"Warning: could not load RBF baseline: {exc}")
        return None


# ============================================================
# Visualizations
# ============================================================

def plot_comparison(comparison_df: pd.DataFrame, fig_dir: Path) -> None:
    name_map = {
        "00_last_pca_rbf_svm": "PCA + RBF-SVM",
        "01_last_xgboost": "XGBoost",
        "02_last_pca_xgboost": "PCA + XGBoost",
    }

    metrics = [
        ("oof_balanced_accuracy", "Balanced Accuracy"),
        ("oof_f1", "F1"),
        ("oof_auroc", "AUROC"),
    ]

    x = np.arange(len(metrics))
    n_methods = len(comparison_df)
    width = 0.75 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        vals = [float(row[m]) for m, _ in metrics]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=name_map.get(row["experiment"], row["experiment"]),
        )

        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("OOF score")
    ax.set_title("Nested-CV OOF Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        fig_dir / "model_comparison_oof.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrix_for_result(result: Dict, fig_dir: Path) -> None:
    pred_df = result["pred_df"]
    exp_name = result["experiment"].name

    cm = confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Spurious (0)", "Robust (1)"])
    ax.set_yticklabels(["Spurious (0)", "Robust (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(exp_name)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(
        fig_dir / f"{exp_name}_confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pca_score(final_result: Dict, fig_dir: Path) -> None:
    """PCA+XGB에서 PCA dimension별 최고 inner-CV balanced accuracy."""
    grid_df = final_result["grid_df"].copy()
    col = "param_feature__n_components"
    if col not in grid_df.columns:
        return

    grid_df["pca_dim"] = grid_df[col].astype(int)
    best_by_p = (
        grid_df.groupby("pca_dim", as_index=False)["mean_test_balanced_accuracy"]
        .max()
        .sort_values("pca_dim")
    )

    best_by_p.to_csv(
        fig_dir / "pca_xgboost_best_score_by_pca.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        best_by_p["pca_dim"],
        best_by_p["mean_test_balanced_accuracy"],
        marker="o",
    )
    ax.set_xlabel("PCA dimension (P)")
    ax.set_ylabel("Best mean inner-CV balanced accuracy")
    ax.set_title("PCA + XGBoost: Best Grid-Search Score by PCA Dimension")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "pca_xgboost_best_score_by_pca.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_grid_heatmap(
    final_result: Dict,
    experiment: Experiment,
    fig_dir: Path,
) -> None:
    """
    max_depth x learning_rate heatmap.
    나머지 grid parameter에 대해서는 각 cell에서 최고 balanced accuracy를 사용.
    """
    grid_df = final_result["grid_df"].copy()
    grid_df["max_depth"] = grid_df["param_model__max_depth"].astype(int)
    grid_df["learning_rate"] = grid_df[
        "param_model__learning_rate"
    ].astype(float)

    pivot = grid_df.pivot_table(
        index="max_depth",
        columns="learning_rate",
        values="mean_test_balanced_accuracy",
        aggfunc="max",
    )

    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("max_depth")
    ax.set_title(
        f"{experiment.name}\nBest inner-CV balanced accuracy over remaining grid"
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(im, ax=ax, label="Best inner-CV balanced accuracy")
    fig.tight_layout()
    fig.savefig(
        fig_dir / f"{experiment.name}_grid_heatmap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def make_report(
    comparison_df: pd.DataFrame,
    final_results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    display_names = {
        "00_last_pca_rbf_svm": "PCA + RBF-SVM",
        "01_last_xgboost": "XGBoost",
        "02_last_pca_xgboost": "PCA + XGBoost",
    }

    lines = [
        "# XGBoost Comparison Report",
        "",
        "## Nested-CV performance",
        "",
        "| Method | Balanced Accuracy | F1 | AUROC | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in comparison_df.iterrows():
        name = display_names.get(row["experiment"], row["experiment"])
        lines.append(
            f"| {name} | "
            f"{row['fold_balanced_accuracy_mean']:.4f} ± "
            f"{row['fold_balanced_accuracy_std']:.4f} | "
            f"{row['fold_f1_mean']:.4f} ± {row['fold_f1_std']:.4f} | "
            f"{row['fold_auroc_mean']:.4f} ± {row['fold_auroc_std']:.4f} | "
            f"{row['fold_accuracy_mean']:.4f} ± "
            f"{row['fold_accuracy_std']:.4f} |"
        )

    lines += [
        "",
        "## OOF performance",
        "",
        "| Method | Balanced Accuracy | F1 | AUROC | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in comparison_df.iterrows():
        name = display_names.get(row["experiment"], row["experiment"])
        lines.append(
            f"| {name} | "
            f"{row['oof_balanced_accuracy']:.4f} | "
            f"{row['oof_f1']:.4f} | "
            f"{row['oof_auroc']:.4f} | "
            f"{row['oof_accuracy']:.4f} |"
        )

    lines += [
        "",
        "## Final full-data GridSearch best parameters",
        "",
    ]

    for exp in EXPERIMENTS:
        best = final_results[exp.name]["best"]
        lines.append(f"### {display_names[exp.name]}")
        lines.append("")
        for key, value in best.items():
            lines.append(f"- {key}: **{value}**")
        lines.append("")

    lines += [
        "## Interpretation note",
        "",
        "- Nested outer-fold performance is the unbiased comparison result.",
        "- Full-data GridSearch scores are for final hyperparameter/model selection, not held-out test performance.",
        "- Both XGBoost variants use identical outer folds and identical geometry features; the primary difference is whether PCA is applied before XGBoost.",
    ]

    (output_dir / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    config = Config()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    loader = ActivationLoader(
        config.artifact_dir,
        mmap_mode="r",
        verify_files=True,
    )
    print(loader)

    y = get_labels(loader)
    groups = get_groups(loader)
    X, sample_ids, n_layers = preload_last(loader, config)

    print(
        f"Samples={len(y)} | Robust={int((y == 1).sum())} | "
        f"Spurious={int((y == 0).sum())} | "
        f"groups={len(np.unique(groups))}"
    )

    raw_dim = (
        n_layers * X.shape[-1]
        + (n_layers if config.use_norm else 0)
        + (n_layers - 1 if config.use_cosine else 0)
    )

    print(f"Raw XGBoost feature dimension = {raw_dim}")
    print(
        "Raw XGBoost grid combinations =",
        len(config.xgb_n_estimators)
        * len(config.xgb_max_depth)
        * len(config.xgb_learning_rate)
        * len(config.xgb_subsample)
        * len(config.xgb_colsample_bytree),
    )
    print(
        "PCA+XGBoost grid combinations =",
        len(config.pca_dims)
        * len(config.xgb_n_estimators)
        * len(config.xgb_max_depth)
        * len(config.xgb_learning_rate)
        * len(config.xgb_subsample)
        * len(config.xgb_colsample_bytree),
    )

    # identical outer folds for both models
    outer_cv = StratifiedGroupKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.seed,
    )
    outer_splits = list(
        outer_cv.split(np.zeros(len(y)), y, groups=groups)
    )

    # save exact split membership
    split_rows = []
    for fold, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        for idx in tr_idx:
            split_rows.append(
                {
                    "fold": fold,
                    "split": "train",
                    "sample_id": sample_ids[int(idx)],
                }
            )
        for idx in te_idx:
            split_rows.append(
                {
                    "fold": fold,
                    "split": "test",
                    "sample_id": sample_ids[int(idx)],
                }
            )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "outer_splits.csv",
        index=False,
    )

    # nested-CV experiments
    nested_results = []
    for experiment in EXPERIMENTS:
        nested_results.append(
            run_experiment(
                experiment=experiment,
                X=X,
                y=y,
                groups=groups,
                sample_ids=sample_ids,
                outer_splits=outer_splits,
                config=config,
                output_dir=output_dir,
            )
        )

    comparison_rows = []

    # optional RBF baseline
    rbf_baseline = load_rbf_baseline(config.rbf_baseline_summary)
    if rbf_baseline is not None:
        comparison_rows.append(rbf_baseline)

    for result in nested_results:
        comparison_rows.append(summarize_experiment(result))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        output_dir / "comparison_summary.csv",
        index=False,
    )

    # final full-data GridSearch
    final_results = {}
    for experiment in EXPERIMENTS:
        final_results[experiment.name] = run_final_grid_search(
            experiment=experiment,
            X=X,
            y=y,
            groups=groups,
            config=config,
            output_dir=output_dir,
        )

    # visualizations
    plot_comparison(comparison_df, fig_dir)

    for result in nested_results:
        plot_confusion_matrix_for_result(result, fig_dir)

    for experiment in EXPERIMENTS:
        plot_grid_heatmap(
            final_results[experiment.name],
            experiment,
            fig_dir,
        )

    plot_pca_score(
        final_results["02_last_pca_xgboost"],
        fig_dir,
    )

    # report
    make_report(
        comparison_df=comparison_df,
        final_results=final_results,
        output_dir=output_dir,
    )

    print("\n" + "#" * 120)
    print("FINAL MODEL COMPARISON")
    print("#" * 120)

    display_cols = [
        "experiment",
        "fold_balanced_accuracy_mean",
        "fold_balanced_accuracy_std",
        "oof_balanced_accuracy",
        "fold_f1_mean",
        "fold_f1_std",
        "oof_f1",
        "fold_auroc_mean",
        "fold_auroc_std",
        "oof_auroc",
        "oof_accuracy",
    ]
    print(comparison_df[display_cols].to_string(index=False))

    print("\nFeature dimensions:")
    print(f"  XGBoost raw: {raw_dim}")
    print(f"  PCA + XGBoost: {n_layers}P + {n_layers + (n_layers - 1)}")

    print("\nSaved:")
    print(output_dir / "comparison_summary.csv")
    print(output_dir / "report.md")
    print(fig_dir)


if __name__ == "__main__":
    main()
