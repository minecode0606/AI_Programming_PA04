from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lodder import ActivationLoader


GammaType = Union[str, float]


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    artifact_dir: str = "artifacts/deepseek_r1_qwen3_8b_tat"
    output_dir: str = "artifacts/rbf_gridsearch"

    mode: str = "tat"
    n_layers: int | None = None

    # unbiased nested CV
    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42

    # GridSearch space
    pca_dims: Tuple[int, ...] = (4, 8, 16, 32, 64)
    c_values: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    gamma_values: Tuple[GammaType, ...] = (
        "scale",
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
    )

    class_weight: str | None = "balanced"

    # geometry features
    use_norm: bool = True
    use_cosine: bool = True

    pca_iterated_power: int = 5
    svm_cache_size_mb: int = 2048

    # GridSearchCV
    # Large activation tensors can be copied per worker, so 1 is safer.
    grid_n_jobs: int = 1
    grid_verbose: int = 1

    save_outer_models: bool = False


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
        label = loader.get_metadata(i)["label"]
        labels.append(int(bool(label)))
    return np.asarray(labels, dtype=np.int64)


def get_groups(loader: ActivationLoader) -> np.ndarray:
    """Keep identical problem_hash values in the same fold."""
    groups = []

    for i in range(len(loader)):
        meta = loader.get_metadata(i)
        group = meta.get("problem_hash")

        if group is None or (
            isinstance(group, float)
            and np.isnan(group)
        ):
            group = meta["sample_id"]

        groups.append(str(group))

    return np.asarray(groups)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    y_pred = (y_score >= 0.0).astype(np.int64)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(
            roc_auc_score(y_true, y_score)
        )
    else:
        result["auroc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    result.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )

    return result


# ============================================================
# Load Last-token TaT trajectory
# ============================================================

def resolve_n_layers(
    loader: ActivationLoader,
    config: Config,
) -> int:
    if config.n_layers is not None:
        return int(config.n_layers)

    if hasattr(loader, "num_layers"):
        return int(loader.num_layers)

    raise AttributeError(
        "Could not infer num_layers. "
        "Set Config.n_layers manually, e.g. 36."
    )


def tat_to_last(
    x: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """
    TaT:
        [T*L, D]

    reshape:
        [T, L, D]

    select LAST token:
        [L, D]
    """
    x = to_numpy_float32(x)

    if x.ndim != 2:
        raise ValueError(
            f"Expected [T*L, D], got {x.shape}"
        )

    m, hidden = x.shape

    if m % n_layers != 0:
        raise ValueError(
            f"Trajectory length={m} is not divisible by "
            f"n_layers={n_layers}."
        )

    n_tokens = m // n_layers

    grid = x.reshape(
        n_tokens,
        n_layers,
        hidden,
    )

    return grid[-1].astype(
        np.float32,
        copy=False,
    )


def preload_last(
    loader: ActivationLoader,
    config: Config,
) -> Tuple[np.ndarray, List[str], int]:
    n_layers = resolve_n_layers(loader, config)

    xs = []
    sample_ids = []
    expected_shape = None

    print(
        f"Preloading LAST-token trajectory | "
        f"n_layers={n_layers}"
    )

    for i in range(len(loader)):
        sample = loader.load_sample(
            i,
            mode=config.mode,
            as_torch=True,
        )

        x_last = tat_to_last(
            sample["x"],
            n_layers=n_layers,
        )

        if expected_shape is None:
            expected_shape = x_last.shape
        elif x_last.shape != expected_shape:
            raise ValueError(
                f"Inconsistent shape at sample {i}: "
                f"{x_last.shape} != {expected_shape}"
            )

        xs.append(x_last)
        sample_ids.append(str(sample["sample_id"]))

        if (
            (i + 1) % 20 == 0
            or (i + 1) == len(loader)
        ):
            print(
                f"  loaded {i + 1}/{len(loader)}"
            )

    X = np.stack(
        xs,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "X_last:",
        X.shape,
        "(N, layers, hidden)",
    )

    return X, sample_ids, n_layers


# ============================================================
# Custom sklearn transformer
# ============================================================

class LayerPCAGeometryTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Input:
        X [N, L, D]

    1) Fit PCA only on the current training split:
        [N*L, D] -> PCA

    2) PCA directional feature:
        [N, L, P] -> [N, L*P]

    3) Optional geometry:
        layer norm       -> [N, L]
        adjacent cosine  -> [N, L-1]

    Output:
        [N, L*P + geometry_dims]

    Since this transformer is inside sklearn Pipeline,
    GridSearchCV refits PCA inside every inner fold.
    """

    def __init__(
        self,
        n_components: int = 16,
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
            raise ValueError(
                f"Expected [N,L,D], got {X.shape}"
            )

        self.n_layers_ = int(X.shape[1])
        self.hidden_size_ = int(X.shape[2])

        rows = X.reshape(-1, X.shape[-1])

        max_allowed = min(
            rows.shape[0] - 1,
            rows.shape[1],
        )

        if int(self.n_components) > max_allowed:
            raise ValueError(
                f"n_components={self.n_components} exceeds "
                f"max_allowed={max_allowed}"
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

        if X.ndim != 3:
            raise ValueError(
                f"Expected [N,L,D], got {X.shape}"
            )

        n, l, d = X.shape

        rows = X.reshape(-1, d)

        Z = self.pca_.transform(rows).astype(
            np.float32,
            copy=False,
        )

        Z = Z.reshape(
            n,
            l,
            int(self.n_components),
        )

        directional = Z.reshape(n, -1)

        parts = [directional]

        if self.use_norm:
            norms = np.linalg.norm(
                X,
                axis=-1,
            ).astype(
                np.float32,
                copy=False,
            )
            parts.append(norms)

        if self.use_cosine:
            eps = 1e-8

            prev = X[:, :-1, :]
            nxt = X[:, 1:, :]

            numerator = np.sum(
                prev * nxt,
                axis=-1,
            )

            denominator = (
                np.linalg.norm(prev, axis=-1)
                * np.linalg.norm(nxt, axis=-1)
                + eps
            )

            cosine = (
                numerator / denominator
            ).astype(
                np.float32,
                copy=False,
            )

            parts.append(cosine)

        features = np.concatenate(
            parts,
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        return features


# ============================================================
# GridSearch pipeline
# ============================================================

def make_pipeline(
    config: Config,
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature",
                LayerPCAGeometryTransformer(
                    n_components=16,
                    use_norm=config.use_norm,
                    use_cosine=config.use_cosine,
                    random_state=seed,
                    iterated_power=config.pca_iterated_power,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight=config.class_weight,
                    cache_size=float(
                        config.svm_cache_size_mb
                    ),
                    random_state=seed,
                ),
            ),
        ]
    )


def make_param_grid(
    config: Config,
) -> Dict:
    return {
        "feature__n_components": list(
            config.pca_dims
        ),
        "svc__C": list(
            config.c_values
        ),
        "svc__gamma": list(
            config.gamma_values
        ),
    }


def make_scoring() -> Dict[str, str]:
    return {
        "balanced_accuracy": "balanced_accuracy",
        "auroc": "roc_auc",
        "f1": "f1",
        "accuracy": "accuracy",
    }


def make_grid_search(
    config: Config,
    seed: int,
) -> GridSearchCV:
    inner_cv = StratifiedGroupKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=seed,
    )

    return GridSearchCV(
        estimator=make_pipeline(
            config=config,
            seed=seed,
        ),
        param_grid=make_param_grid(config),
        scoring=make_scoring(),
        refit="balanced_accuracy",
        cv=inner_cv,
        n_jobs=config.grid_n_jobs,
        verbose=config.grid_verbose,
        return_train_score=False,
        error_score="raise",
    )


# ============================================================
# Nested outer evaluation
# ============================================================

def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: List[str],
    config: Config,
    output_dir: Path,
):
    outer_cv = StratifiedGroupKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.seed,
    )

    outer_splits = list(
        outer_cv.split(
            np.zeros(len(y)),
            y,
            groups=groups,
        )
    )

    fold_rows = []
    pred_rows = []
    best_param_rows = []

    outer_dir = output_dir / "nested_cv"
    outer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"NESTED OUTER FOLD "
            f"{fold}/{config.outer_splits}"
        )
        print("=" * 100)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        g_train = groups[train_idx]

        search = make_grid_search(
            config=config,
            seed=(
                config.seed
                + 1000 * fold
            ),
        )

        search.fit(
            X_train,
            y_train,
            groups=g_train,
        )

        cv_df = pd.DataFrame(
            search.cv_results_
        )

        cv_df.to_csv(
            outer_dir
            / f"fold_{fold:02d}_grid_results.csv",
            index=False,
        )

        best_params = search.best_params_

        print(
            "Best params:",
            best_params,
        )
        print(
            "Best inner balanced accuracy:",
            f"{search.best_score_:.6f}",
        )

        scores = (
            search.best_estimator_
            .decision_function(X_test)
        )

        metrics = compute_metrics(
            y_true=y_test,
            y_score=scores,
        )

        feature_step = (
            search.best_estimator_
            .named_steps["feature"]
        )

        sample_feature = (
            feature_step.transform(
                X_train[:1]
            )
        )

        metrics.update(
            {
                "fold": fold,
                "n_train": int(
                    len(train_idx)
                ),
                "n_test": int(
                    len(test_idx)
                ),
                "best_pca_dim": int(
                    best_params[
                        "feature__n_components"
                    ]
                ),
                "best_C": float(
                    best_params["svc__C"]
                ),
                "best_gamma": str(
                    best_params["svc__gamma"]
                ),
                "feature_dim": int(
                    sample_feature.shape[1]
                ),
                "inner_best_balanced_accuracy": float(
                    search.best_score_
                ),
                "pca_explained_variance_ratio_sum": float(
                    np.sum(
                        feature_step
                        .pca_
                        .explained_variance_ratio_
                    )
                ),
            }
        )

        fold_rows.append(metrics)

        best_param_rows.append(
            {
                "fold": fold,
                "pca_dim": int(
                    best_params[
                        "feature__n_components"
                    ]
                ),
                "C": float(
                    best_params["svc__C"]
                ),
                "gamma": str(
                    best_params["svc__gamma"]
                ),
                "inner_balanced_accuracy": float(
                    search.best_score_
                ),
            }
        )

        y_pred = (
            scores >= 0.0
        ).astype(int)

        for idx, true, score, pred in zip(
            test_idx,
            y_test,
            scores,
            y_pred,
        ):
            pred_rows.append(
                {
                    "fold": fold,
                    "sample_id": (
                        sample_ids[int(idx)]
                    ),
                    "group": (
                        groups[int(idx)]
                    ),
                    "y_true": int(true),
                    "decision_score": float(score),
                    "y_pred": int(pred),
                }
            )

        print(
            f"OUTER TEST | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auroc={metrics['auroc']:.4f}"
        )

        if config.save_outer_models:
            joblib.dump(
                search.best_estimator_,
                outer_dir
                / f"fold_{fold:02d}_best_model.joblib",
            )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    best_param_df = pd.DataFrame(best_param_rows)

    fold_df.to_csv(
        outer_dir / "fold_metrics.csv",
        index=False,
    )

    pred_df.to_csv(
        outer_dir / "oof_predictions.csv",
        index=False,
    )

    best_param_df.to_csv(
        outer_dir / "best_params_by_fold.csv",
        index=False,
    )

    oof_metrics = compute_metrics(
        y_true=pred_df[
            "y_true"
        ].to_numpy(),
        y_score=pred_df[
            "decision_score"
        ].to_numpy(),
    )

    return (
        fold_df,
        pred_df,
        best_param_df,
        oof_metrics,
    )


# ============================================================
# Final full-data GridSearch
# ============================================================

def run_final_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Config,
    output_dir: Path,
):
    print("\n" + "#" * 100)
    print("FINAL FULL-DATA GRID SEARCH")
    print("#" * 100)

    search = make_grid_search(
        config=config,
        seed=(
            config.seed
            + 999_999
        ),
    )

    search.fit(
        X,
        y,
        groups=groups,
    )

    results_df = pd.DataFrame(
        search.cv_results_
    )

    results_df.to_csv(
        output_dir
        / "final_grid_search_results.csv",
        index=False,
    )

    best_params = search.best_params_

    feature_step = (
        search.best_estimator_
        .named_steps["feature"]
    )

    sample_feature = (
        feature_step.transform(
            X[:1]
        )
    )

    best_record = {
        "pca_dim": int(
            best_params[
                "feature__n_components"
            ]
        ),
        "C": float(
            best_params["svc__C"]
        ),
        "gamma": str(
            best_params["svc__gamma"]
        ),
        "cv_balanced_accuracy": float(
            search.best_score_
        ),
        "feature_dim": int(
            sample_feature.shape[1]
        ),
        "pca_explained_variance_ratio_sum": float(
            np.sum(
                feature_step
                .pca_
                .explained_variance_ratio_
            )
        ),
    }

    with (
        output_dir
        / "final_best_params.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            best_record,
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(
        search.best_estimator_,
        output_dir
        / "final_best_model.joblib",
    )

    print(
        "FINAL BEST PARAMS:",
        best_record,
    )

    return (
        search,
        results_df,
        best_record,
    )


# ============================================================
# Visualization helpers
# ============================================================

def normalize_param_columns(
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    df = grid_df.copy()

    df["pca_dim"] = df[
        "param_feature__n_components"
    ].astype(int)

    df["C"] = df[
        "param_svc__C"
    ].astype(float)

    df["gamma"] = df[
        "param_svc__gamma"
    ].map(str)

    return df


def plot_heatmaps(
    grid_df: pd.DataFrame,
    config: Config,
    fig_dir: Path,
):
    """Save one C x gamma heatmap for every PCA dimension."""
    df = normalize_param_columns(grid_df)

    for pca_dim in sorted(
        df["pca_dim"].unique()
    ):
        sub = df[
            df["pca_dim"] == pca_dim
        ].copy()

        pivot = sub.pivot_table(
            index="gamma",
            columns="C",
            values="mean_test_balanced_accuracy",
            aggfunc="mean",
        )

        gamma_order = [
            str(g)
            for g in config.gamma_values
            if str(g) in pivot.index
        ]

        c_order = [
            float(c)
            for c in config.c_values
            if float(c) in pivot.columns
        ]

        pivot = pivot.reindex(
            index=gamma_order,
            columns=c_order,
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        im = ax.imshow(
            pivot.to_numpy(dtype=float),
            aspect="auto",
        )

        ax.set_title(
            f"Grid Search Balanced Accuracy "
            f"(PCA={pca_dim})"
        )
        ax.set_xlabel("C")
        ax.set_ylabel("gamma")

        ax.set_xticks(
            np.arange(
                len(pivot.columns)
            )
        )
        ax.set_xticklabels(
            [
                str(x)
                for x in pivot.columns
            ],
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            np.arange(
                len(pivot.index)
            )
        )
        ax.set_yticklabels(
            list(pivot.index)
        )

        values = pivot.to_numpy(
            dtype=float
        )

        for i in range(
            values.shape[0]
        ):
            for j in range(
                values.shape[1]
            ):
                value = values[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        fig.colorbar(
            im,
            ax=ax,
            label="Mean CV balanced accuracy",
        )

        fig.tight_layout()
        fig.savefig(
            fig_dir
            / f"grid_heatmap_pca_{pca_dim}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_best_score_by_pca(
    grid_df: pd.DataFrame,
    fig_dir: Path,
):
    df = normalize_param_columns(grid_df)

    best_by_p = (
        df.groupby(
            "pca_dim",
            as_index=False,
        )[
            "mean_test_balanced_accuracy"
        ]
        .max()
        .sort_values("pca_dim")
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.plot(
        best_by_p["pca_dim"],
        best_by_p[
            "mean_test_balanced_accuracy"
        ],
        marker="o",
    )

    ax.set_xlabel("PCA dimension (P)")
    ax.set_ylabel(
        "Best mean CV balanced accuracy"
    )
    ax.set_title(
        "Best Grid-Search Score by PCA Dimension"
    )
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        fig_dir
        / "best_balanced_accuracy_by_pca.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    best_by_p.to_csv(
        fig_dir
        / "best_score_by_pca.csv",
        index=False,
    )


def plot_oof_metrics(
    oof_metrics: Dict[str, float],
    fig_dir: Path,
):
    names = [
        "Balanced Accuracy",
        "F1",
        "AUROC",
    ]

    values = [
        oof_metrics[
            "balanced_accuracy"
        ],
        oof_metrics["f1"],
        oof_metrics["auroc"],
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    bars = ax.bar(
        names,
        values,
    )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(
        "Nested-CV OOF Performance"
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        fig_dir
        / "nested_oof_metrics.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrix(
    pred_df: pd.DataFrame,
    fig_dir: Path,
):
    cm = confusion_matrix(
        pred_df["y_true"],
        pred_df["y_pred"],
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(5, 4.5)
    )

    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        [
            "Spurious (0)",
            "Robust (1)",
        ]
    )
    ax.set_yticklabels(
        [
            "Spurious (0)",
            "Robust (1)",
        ]
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        "Nested-CV OOF Confusion Matrix"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(
        fig_dir
        / "nested_oof_confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Reporting
# ============================================================

def summarize_fold_metrics(
    fold_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    summary = {}

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
    ]:
        values = fold_df[
            metric
        ].to_numpy(dtype=float)

        summary[metric] = {
            "mean": float(
                np.nanmean(values)
            ),
            "std": float(
                np.nanstd(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            ),
        }

    return summary


def top_grid_rows(
    grid_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    df = normalize_param_columns(
        grid_df
    )

    cols = [
        "pca_dim",
        "C",
        "gamma",
        "mean_test_balanced_accuracy",
        "std_test_balanced_accuracy",
        "mean_test_auroc",
        "mean_test_f1",
        "mean_test_accuracy",
        "rank_test_balanced_accuracy",
    ]

    return (
        df.sort_values(
            [
                "rank_test_balanced_accuracy",
                "mean_test_auroc",
            ],
            ascending=[
                True,
                False,
            ],
        )[cols]
        .head(n)
        .reset_index(drop=True)
    )


def make_report(
    config: Config,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_df: pd.DataFrame,
    best_param_df: pd.DataFrame,
    oof_metrics: Dict[str, float],
    final_best: Dict,
    final_grid_df: pd.DataFrame,
    output_dir: Path,
):
    fold_summary = summarize_fold_metrics(
        fold_df
    )

    top10 = top_grid_rows(
        final_grid_df,
        n=10,
    )

    top10.to_csv(
        output_dir
        / "top10_grid_configs.csv",
        index=False,
    )

    param_frequency = (
        best_param_df.groupby(
            [
                "pca_dim",
                "C",
                "gamma",
            ],
            as_index=False,
        )
        .size()
        .sort_values(
            "size",
            ascending=False,
        )
    )

    param_frequency.to_csv(
        output_dir
        / "outer_fold_best_param_frequency.csv",
        index=False,
    )

    lines = []

    lines.append(
        "# RBF-SVM Grid Search Report"
    )
    lines.append("")

    lines.append(
        "## 1. Experimental setup"
    )
    lines.append("")
    lines.append(
        f"- Samples: {len(y)}"
    )
    lines.append(
        f"- Robust (1): {int((y == 1).sum())}"
    )
    lines.append(
        f"- Spurious (0): {int((y == 0).sum())}"
    )
    lines.append(
        f"- Unique groups: {len(np.unique(groups))}"
    )
    lines.append(
        f"- Input tensor: {list(X.shape)} = [N, layers, hidden]"
    )
    lines.append(
        f"- Outer CV: {config.outer_splits}-fold StratifiedGroupKFold"
    )
    lines.append(
        f"- Inner GridSearch CV: {config.inner_splits}-fold StratifiedGroupKFold"
    )
    lines.append(
        f"- Grid PCA dimensions: {list(config.pca_dims)}"
    )
    lines.append(
        f"- Grid C: {list(config.c_values)}"
    )
    lines.append(
        f"- Grid gamma: {list(config.gamma_values)}"
    )

    n_configs = (
        len(config.pca_dims)
        * len(config.c_values)
        * len(config.gamma_values)
    )

    lines.append(
        f"- Total parameter combinations: {n_configs}"
    )
    lines.append(
        "- Refit metric: balanced accuracy"
    )
    lines.append(
        f"- Geometry: norm={config.use_norm}, cosine={config.use_cosine}"
    )
    lines.append("")

    lines.append(
        "## 2. Nested-CV unbiased performance"
    )
    lines.append("")
    lines.append(
        "| Metric | Fold mean ± std | OOF |"
    )
    lines.append(
        "|---|---:|---:|"
    )

    for metric, label in [
        ("accuracy", "Accuracy"),
        (
            "balanced_accuracy",
            "Balanced Accuracy",
        ),
        ("f1", "F1"),
        ("auroc", "AUROC"),
    ]:
        mean = fold_summary[metric][
            "mean"
        ]
        std = fold_summary[metric][
            "std"
        ]
        oof = oof_metrics[metric]

        lines.append(
            f"| {label} | "
            f"{mean:.4f} ± {std:.4f} | "
            f"{oof:.4f} |"
        )

    lines.append("")

    lines.append(
        "## 3. Final full-data GridSearch best configuration"
    )
    lines.append("")
    lines.append(
        f"- PCA dimension: **{final_best['pca_dim']}**"
    )
    lines.append(
        f"- C: **{final_best['C']}**"
    )
    lines.append(
        f"- gamma: **{final_best['gamma']}**"
    )
    lines.append(
        f"- Inner-CV balanced accuracy: **{final_best['cv_balanced_accuracy']:.4f}**"
    )
    lines.append(
        f"- Final feature dimension: **{final_best['feature_dim']}**"
    )
    lines.append(
        f"- PCA explained variance ratio sum: "
        f"**{final_best['pca_explained_variance_ratio_sum']:.4f}**"
    )
    lines.append("")

    lines.append(
        "## 4. Fold-wise selected parameters"
    )
    lines.append("")
    lines.append(
        "| Fold | PCA | C | gamma | Inner balanced acc. |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|"
    )

    for _, row in best_param_df.iterrows():
        lines.append(
            f"| {int(row['fold'])} | "
            f"{int(row['pca_dim'])} | "
            f"{row['C']} | "
            f"{row['gamma']} | "
            f"{row['inner_balanced_accuracy']:.4f} |"
        )

    lines.append("")
    lines.append(
        "## 5. Interpretation"
    )
    lines.append("")
    lines.append(
        "- The nested-CV OOF metrics are the main unbiased performance estimates."
    )
    lines.append(
        "- The full-data GridSearch result is intended for choosing the final model after evaluation, not for reporting unbiased test performance."
    )
    lines.append(
        "- If fold-wise best hyperparameters vary strongly, the model is likely sensitive to the small dataset and the selected values should be interpreted cautiously."
    )
    lines.append(
        "- Balanced accuracy is the refit criterion because the class distribution may be imbalanced."
    )
    lines.append("")

    lines.append(
        "## 6. Saved artifacts"
    )
    lines.append("")
    lines.append(
        "- `nested_cv/fold_metrics.csv`"
    )
    lines.append(
        "- `nested_cv/oof_predictions.csv`"
    )
    lines.append(
        "- `nested_cv/best_params_by_fold.csv`"
    )
    lines.append(
        "- `final_grid_search_results.csv`"
    )
    lines.append(
        "- `final_best_params.json`"
    )
    lines.append(
        "- `final_best_model.joblib`"
    )
    lines.append(
        "- `top10_grid_configs.csv`"
    )
    lines.append(
        "- `figures/grid_heatmap_pca_*.png`"
    )
    lines.append(
        "- `figures/best_balanced_accuracy_by_pca.png`"
    )
    lines.append(
        "- `figures/nested_oof_metrics.png`"
    )
    lines.append(
        "- `figures/nested_oof_confusion_matrix.png`"
    )

    report_path = (
        output_dir / "report.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    summary_json = {
        "config": asdict(config),
        "dataset": {
            "n_samples": int(len(y)),
            "n_robust": int(
                (y == 1).sum()
            ),
            "n_spurious": int(
                (y == 0).sum()
            ),
            "n_groups": int(
                len(np.unique(groups))
            ),
            "input_shape": list(
                X.shape
            ),
        },
        "nested_cv_fold_summary": (
            fold_summary
        ),
        "nested_cv_oof": (
            oof_metrics
        ),
        "final_best": (
            final_best
        ),
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_json,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():
    config = Config()

    set_seed(config.seed)

    output_dir = Path(
        config.output_dir
    )
    fig_dir = (
        output_dir / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(config),
            f,
            ensure_ascii=False,
            indent=2,
        )

    loader = ActivationLoader(
        config.artifact_dir,
        mmap_mode="r",
        verify_files=True,
    )

    print(loader)

    y = get_labels(loader)
    groups = get_groups(loader)

    X, sample_ids, n_layers = preload_last(
        loader,
        config,
    )

    print(
        f"Samples={len(y)} | "
        f"Robust={int((y == 1).sum())} | "
        f"Spurious={int((y == 0).sum())} | "
        f"groups={len(np.unique(groups))} | "
        f"layers={n_layers} | "
        f"hidden={X.shape[-1]}"
    )

    n_grid = (
        len(config.pca_dims)
        * len(config.c_values)
        * len(config.gamma_values)
    )

    print(
        f"Grid combinations = {n_grid}"
    )

    # 1) Nested CV: unbiased performance estimate
    (
        fold_df,
        pred_df,
        best_param_df,
        oof_metrics,
    ) = run_nested_cv(
        X=X,
        y=y,
        groups=groups,
        sample_ids=sample_ids,
        config=config,
        output_dir=output_dir,
    )

    # 2) Final full-data GridSearch for final model selection
    (
        final_search,
        final_grid_df,
        final_best,
    ) = run_final_grid_search(
        X=X,
        y=y,
        groups=groups,
        config=config,
        output_dir=output_dir,
    )

    # 3) Visualizations
    plot_heatmaps(
        grid_df=final_grid_df,
        config=config,
        fig_dir=fig_dir,
    )

    plot_best_score_by_pca(
        grid_df=final_grid_df,
        fig_dir=fig_dir,
    )

    plot_oof_metrics(
        oof_metrics=oof_metrics,
        fig_dir=fig_dir,
    )

    plot_confusion_matrix(
        pred_df=pred_df,
        fig_dir=fig_dir,
    )

    # 4) Markdown + JSON report
    make_report(
        config=config,
        X=X,
        y=y,
        groups=groups,
        fold_df=fold_df,
        best_param_df=best_param_df,
        oof_metrics=oof_metrics,
        final_best=final_best,
        final_grid_df=final_grid_df,
        output_dir=output_dir,
    )

    # 5) Console summary
    fold_summary = summarize_fold_metrics(
        fold_df
    )

    print("\n" + "#" * 100)
    print("NESTED-CV PERFORMANCE")
    print("#" * 100)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
    ]:
        mean = fold_summary[metric][
            "mean"
        ]
        std = fold_summary[metric][
            "std"
        ]

        print(
            f"{metric:>18s}: "
            f"{mean:.4f} ± {std:.4f} | "
            f"OOF={oof_metrics[metric]:.4f}"
        )

    print("\nFINAL GRIDSEARCH BEST")
    print(
        f"PCA P = "
        f"{final_best['pca_dim']}"
    )
    print(
        f"C = "
        f"{final_best['C']}"
    )
    print(
        f"gamma = "
        f"{final_best['gamma']}"
    )
    print(
        f"CV balanced accuracy = "
        f"{final_best['cv_balanced_accuracy']:.4f}"
    )
    print(
        f"feature dim = "
        f"{final_best['feature_dim']}"
    )

    print("\nSaved report:")
    print(
        output_dir / "report.md"
    )

    print("\nSaved final model:")
    print(
        output_dir / "final_best_model.joblib"
    )

    print("\nSaved figures:")
    for path in sorted(
        fig_dir.glob("*.png")
    ):
        print(" -", path)


if __name__ == "__main__":
    main()
