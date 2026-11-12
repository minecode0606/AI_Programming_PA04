from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
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
    output_dir: str = "artifacts/geometry_ablation"

    mode: str = "tat"
    n_layers: int | None = None

    n_splits: int = 5
    inner_splits: int = 3
    seed: int = 42

    pca_dims: Tuple[int, ...] = (8, 16, 32)

    rbf_c_values: Tuple[float, ...] = (
        0.1,
        1.0,
        10.0,
        100.0,
    )

    rbf_gamma_values: Tuple[GammaType, ...] = (
        "scale",
        1e-4,
        1e-3,
        1e-2,
    )

    class_weight: str | None = "balanced"

    pca_iterated_power: int = 5
    svm_cache_size_mb: int = 2048

    save_fold_models: bool = False


# ============================================================
# Ablation definitions
# ============================================================

@dataclass(frozen=True)
class Experiment:
    name: str
    use_norm: bool
    use_cosine: bool


EXPERIMENTS: Tuple[Experiment, ...] = (
    # 둘 다 사용
    Experiment(
        name="01_last_pca_rbf_norm_cosine",
        use_norm=True,
        use_cosine=True,
    ),

    # norm 제거, cosine만 사용
    Experiment(
        name="02_last_pca_rbf_no_norm",
        use_norm=False,
        use_cosine=True,
    ),

    # cosine 제거, norm만 사용
    Experiment(
        name="03_last_pca_rbf_no_cosine",
        use_norm=True,
        use_cosine=False,
    ),

    # geometry 완전 제거
    Experiment(
        name="04_last_pca_rbf_no_geometry",
        use_norm=False,
        use_cosine=False,
    ),
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

    return np.asarray(
        x,
        dtype=np.float32,
    )


def get_labels(
    loader: ActivationLoader,
) -> np.ndarray:
    labels = []

    for i in range(len(loader)):
        label = loader.get_metadata(i)["label"]
        labels.append(
            int(bool(label))
        )

    return np.asarray(
        labels,
        dtype=np.int64,
    )


def get_groups(
    loader: ActivationLoader,
) -> np.ndarray:
    """
    같은 problem_hash는 같은 fold에 유지합니다.
    """
    groups = []

    for i in range(len(loader)):
        meta = loader.get_metadata(i)

        group = meta.get(
            "problem_hash"
        )

        if group is None or (
            isinstance(group, float)
            and np.isnan(group)
        ):
            group = meta["sample_id"]

        groups.append(
            str(group)
        )

    return np.asarray(
        groups
    )


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    y_pred = (
        y_score >= 0.0
    ).astype(
        np.int64
    )

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }

    if len(
        np.unique(y_true)
    ) == 2:
        metrics["auroc"] = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )
    else:
        metrics["auroc"] = float(
            "nan"
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )

    return metrics


# ============================================================
# Load LAST-token layer trajectory
# ============================================================

def resolve_n_layers(
    loader: ActivationLoader,
    config: Config,
) -> int:
    if config.n_layers is not None:
        return int(
            config.n_layers
        )

    if hasattr(
        loader,
        "num_layers",
    ):
        return int(
            loader.num_layers
        )

    raise AttributeError(
        "Could not infer num_layers. "
        "Set Config.n_layers manually."
    )


def tat_to_last(
    x: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """
    Input:
        [T*L, D]

    reshape:
        [T, L, D]

    select last token:
        [L, D]
    """
    x = to_numpy_float32(
        x
    )

    if x.ndim != 2:
        raise ValueError(
            f"Expected [T*L, D], got {x.shape}"
        )

    m, hidden = x.shape

    if m % n_layers != 0:
        raise ValueError(
            f"Trajectory length {m} is not divisible by "
            f"n_layers={n_layers}."
        )

    n_tokens = (
        m // n_layers
    )

    grid = x.reshape(
        n_tokens,
        n_layers,
        hidden,
    )

    last_layer = grid[-1]

    return last_layer.astype(
        np.float32,
        copy=False,
    )


def preload_last(
    loader: ActivationLoader,
    config: Config,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Returns
    -------
    X_last:
        [N, L, D]
    """
    n_layers = resolve_n_layers(
        loader,
        config,
    )

    xs = []
    sample_ids = []

    expected_shape = None

    print(
        f"Preloading LAST trajectory | "
        f"n_layers={n_layers}"
    )

    for i in range(
        len(loader)
    ):
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
            expected_shape = (
                x_last.shape
            )
        elif (
            x_last.shape
            != expected_shape
        ):
            raise ValueError(
                f"Inconsistent shape at {i}: "
                f"{x_last.shape} != {expected_shape}"
            )

        xs.append(
            x_last
        )

        sample_ids.append(
            str(
                sample["sample_id"]
            )
        )

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
        "(N, layers, hidden)"
    )

    return (
        X,
        sample_ids,
        n_layers,
    )


# ============================================================
# PCA
# ============================================================

def fit_pca(
    X_train: np.ndarray,
    n_components: int,
    seed: int,
    iterated_power: int,
) -> PCA:
    """
    X_train:
        [N, L, D]

    PCA fit rows:
        [N*L, D]
    """
    rows = X_train.reshape(
        -1,
        X_train.shape[-1],
    )

    max_allowed = min(
        rows.shape[0] - 1,
        rows.shape[1],
    )

    if n_components > max_allowed:
        raise ValueError(
            f"PCA n_components={n_components} "
            f"> max_allowed={max_allowed}"
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=iterated_power,
    )

    pca.fit(
        rows
    )

    return pca


def transform_pca(
    pca: PCA,
    X: np.ndarray,
) -> np.ndarray:
    """
    [N, L, D]
        ->
    [N, L, P]
    """
    n, l, d = X.shape

    rows = X.reshape(
        -1,
        d,
    )

    z = pca.transform(
        rows
    ).astype(
        np.float32,
        copy=False,
    )

    return z.reshape(
        n,
        l,
        -1,
    )


# ============================================================
# Geometry features
# ============================================================

def layer_norm_features(
    X: np.ndarray,
) -> np.ndarray:
    """
    X:
        [N, L, D]

    Output:
        [N, L]

    각 layer vector의 L2 norm.
    """
    return np.linalg.norm(
        X,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )


def adjacent_cosine_features(
    X: np.ndarray,
) -> np.ndarray:
    """
    X:
        [N, L, D]

    Output:
        [N, L-1]

    연속된 layer representation 간 cosine similarity.
    """
    eps = 1e-8

    prev = X[
        :,
        :-1,
        :,
    ]

    nxt = X[
        :,
        1:,
        :,
    ]

    numerator = np.sum(
        prev * nxt,
        axis=-1,
    )

    denominator = (
        np.linalg.norm(
            prev,
            axis=-1,
        )
        * np.linalg.norm(
            nxt,
            axis=-1,
        )
        + eps
    )

    cosine = (
        numerator
        / denominator
    )

    return cosine.astype(
        np.float32,
        copy=False,
    )


def build_features(
    Z: np.ndarray,
    X_raw: np.ndarray,
    pca_dim: int,
    experiment: Experiment,
) -> np.ndarray:
    """
    Base:
        PCA trajectory [N,L,P]
        -> flatten [N,L*P]

    Ablations:
        1) norm + cosine
        2) cosine only
        3) norm only
        4) no geometry
    """
    directional = Z[
        ...,
        :pca_dim,
    ].reshape(
        Z.shape[0],
        -1,
    )

    parts = [
        directional
    ]

    if experiment.use_norm:
        norms = layer_norm_features(
            X_raw
        )

        parts.append(
            norms
        )

    if experiment.use_cosine:
        cosines = adjacent_cosine_features(
            X_raw
        )

        parts.append(
            cosines
        )

    features = np.concatenate(
        parts,
        axis=1,
    )

    return features.astype(
        np.float32,
        copy=False,
    )


def expected_feature_dim(
    n_layers: int,
    pca_dim: int,
    experiment: Experiment,
) -> int:
    dim = (
        n_layers
        * pca_dim
    )

    if experiment.use_norm:
        dim += n_layers

    if experiment.use_cosine:
        dim += (
            n_layers - 1
        )

    return int(
        dim
    )


# ============================================================
# Nested CV utilities
# ============================================================

def make_inner_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    seed: int,
):
    unique_groups = np.unique(
        groups
    )

    groups_per_class = []

    for cls in [0, 1]:
        groups_per_class.append(
            len(
                np.unique(
                    groups[
                        y == cls
                    ]
                )
            )
        )

    n_splits = min(
        requested_splits,
        len(unique_groups),
        min(groups_per_class),
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough groups for inner CV. "
            f"groups_per_class={groups_per_class}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    return list(
        splitter.split(
            np.zeros(
                len(y)
            ),
            y,
            groups=groups,
        )
    )


def gamma_to_string(
    gamma: GammaType,
) -> str:
    return str(
        gamma
    )


# ============================================================
# Inner hyperparameter selection
# ============================================================

def select_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    experiment: Experiment,
    config: Config,
    seed: int,
    prefix: str,
) -> Tuple[Dict, pd.DataFrame]:

    inner_splits = make_inner_splits(
        y=y,
        groups=groups,
        requested_splits=config.inner_splits,
        seed=seed,
    )

    max_pca_dim = max(
        config.pca_dims
    )

    param_list = []

    for pca_dim in config.pca_dims:
        for C in config.rbf_c_values:
            for gamma in config.rbf_gamma_values:
                param_list.append(
                    {
                        "pca_dim": int(
                            pca_dim
                        ),
                        "C": float(
                            C
                        ),
                        "gamma": gamma,
                    }
                )

    store = {
        (
            p["pca_dim"],
            p["C"],
            gamma_to_string(
                p["gamma"]
            ),
        ): []
        for p in param_list
    }

    for inner_fold, (
        tr_idx,
        va_idx,
    ) in enumerate(
        inner_splits,
        start=1,
    ):
        X_tr = X[
            tr_idx
        ]

        X_va = X[
            va_idx
        ]

        y_tr = y[
            tr_idx
        ]

        y_va = y[
            va_idx
        ]

        pca = fit_pca(
            X_train=X_tr,
            n_components=max_pca_dim,
            seed=seed + inner_fold,
            iterated_power=config.pca_iterated_power,
        )

        Z_tr = transform_pca(
            pca,
            X_tr,
        )

        Z_va = transform_pca(
            pca,
            X_va,
        )

        feature_cache = {}

        for pca_dim in config.pca_dims:
            F_tr = build_features(
                Z=Z_tr,
                X_raw=X_tr,
                pca_dim=pca_dim,
                experiment=experiment,
            )

            F_va = build_features(
                Z=Z_va,
                X_raw=X_va,
                pca_dim=pca_dim,
                experiment=experiment,
            )

            scaler = StandardScaler()

            F_tr_s = scaler.fit_transform(
                F_tr
            )

            F_va_s = scaler.transform(
                F_va
            )

            feature_cache[
                pca_dim
            ] = (
                F_tr_s,
                F_va_s,
            )

        for params in param_list:
            pca_dim = params[
                "pca_dim"
            ]

            F_tr_s, F_va_s = (
                feature_cache[
                    pca_dim
                ]
            )

            clf = SVC(
                C=float(
                    params["C"]
                ),
                kernel="rbf",
                gamma=params[
                    "gamma"
                ],
                class_weight=config.class_weight,
                cache_size=float(
                    config.svm_cache_size_mb
                ),
                random_state=seed + inner_fold,
            )

            clf.fit(
                F_tr_s,
                y_tr,
            )

            scores = clf.decision_function(
                F_va_s
            )

            metrics = compute_metrics(
                y_true=y_va,
                y_score=scores,
            )

            key = (
                params[
                    "pca_dim"
                ],
                params["C"],
                gamma_to_string(
                    params[
                        "gamma"
                    ]
                ),
            )

            store[
                key
            ].append(
                metrics
            )

        print(
            f"{prefix}"
            f"inner fold "
            f"{inner_fold}/"
            f"{len(inner_splits)} done"
        )

    rows = []

    for (
        pca_dim,
        C,
        gamma_string,
    ), metric_list in store.items():

        bal = np.asarray(
            [
                m[
                    "balanced_accuracy"
                ]
                for m in metric_list
            ],
            dtype=float,
        )

        auc = np.asarray(
            [
                m[
                    "auroc"
                ]
                for m in metric_list
            ],
            dtype=float,
        )

        f1 = np.asarray(
            [
                m[
                    "f1"
                ]
                for m in metric_list
            ],
            dtype=float,
        )

        rows.append(
            {
                "pca_dim": int(
                    pca_dim
                ),
                "C": float(
                    C
                ),
                "gamma": gamma_string,
                "mean_balanced_accuracy": float(
                    np.nanmean(
                        bal
                    )
                ),
                "std_balanced_accuracy": float(
                    np.nanstd(
                        bal,
                        ddof=1,
                    )
                    if len(bal) > 1
                    else 0.0
                ),
                "mean_auroc": float(
                    np.nanmean(
                        auc
                    )
                ),
                "mean_f1": float(
                    np.nanmean(
                        f1
                    )
                ),
            }
        )

    search_df = pd.DataFrame(
        rows
    )

    # 기존 기준과 동일:
    # 1) Balanced Accuracy
    # 2) AUROC
    # 3) 낮은 variance
    # 4) 낮은 PCA dim
    # 5) 낮은 C
    search_df = search_df.sort_values(
        by=[
            "mean_balanced_accuracy",
            "mean_auroc",
            "std_balanced_accuracy",
            "pca_dim",
            "C",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best_row = search_df.iloc[
        0
    ]

    gamma_text = str(
        best_row[
            "gamma"
        ]
    )

    if gamma_text == "scale":
        gamma = "scale"
    else:
        gamma = float(
            gamma_text
        )

    best = {
        "pca_dim": int(
            best_row[
                "pca_dim"
            ]
        ),
        "C": float(
            best_row[
                "C"
            ]
        ),
        "gamma": gamma,
        "inner_balanced_accuracy": float(
            best_row[
                "mean_balanced_accuracy"
            ]
        ),
        "inner_auroc": float(
            best_row[
                "mean_auroc"
            ]
        ),
    }

    return (
        best,
        search_df,
    )


# ============================================================
# Fit one outer fold
# ============================================================

def fit_outer_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    experiment: Experiment,
    params: Dict,
    config: Config,
    seed: int,
):
    pca_dim = int(
        params[
            "pca_dim"
        ]
    )

    pca = fit_pca(
        X_train=X_train,
        n_components=pca_dim,
        seed=seed,
        iterated_power=config.pca_iterated_power,
    )

    Z_train = transform_pca(
        pca,
        X_train,
    )

    Z_test = transform_pca(
        pca,
        X_test,
    )

    F_train = build_features(
        Z=Z_train,
        X_raw=X_train,
        pca_dim=pca_dim,
        experiment=experiment,
    )

    F_test = build_features(
        Z=Z_test,
        X_raw=X_test,
        pca_dim=pca_dim,
        experiment=experiment,
    )

    scaler = StandardScaler()

    F_train_s = scaler.fit_transform(
        F_train
    )

    F_test_s = scaler.transform(
        F_test
    )

    clf = SVC(
        C=float(
            params["C"]
        ),
        kernel="rbf",
        gamma=params[
            "gamma"
        ],
        class_weight=config.class_weight,
        cache_size=float(
            config.svm_cache_size_mb
        ),
        random_state=seed,
    )

    clf.fit(
        F_train_s,
        y_train,
    )

    test_scores = clf.decision_function(
        F_test_s
    )

    bundle = {
        "pca": pca,
        "scaler": scaler,
        "classifier": clf,
        "experiment": asdict(
            experiment
        ),
        "params": params,
        "feature_dim": int(
            F_train.shape[1]
        ),
    }

    return (
        bundle,
        test_scores,
    )


# ============================================================
# Run experiment
# ============================================================

def run_experiment(
    experiment: Experiment,
    X_last: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: List[str],
    outer_splits,
    config: Config,
    n_layers: int,
) -> Dict:

    exp_dir = (
        Path(
            config.output_dir
        )
        / experiment.name
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "#"
        * 90
    )

    print(
        f"EXPERIMENT: "
        f"{experiment.name}"
    )

    print(
        f"use_norm="
        f"{experiment.use_norm} | "
        f"use_cosine="
        f"{experiment.use_cosine}"
    )

    print(
        "#"
        * 90
    )

    fold_rows = []
    prediction_rows = []

    for fold, (
        outer_train,
        outer_test,
    ) in enumerate(
        outer_splits,
        start=1,
    ):

        print(
            "\n"
            + "="
            * 80
        )

        print(
            f"{experiment.name} | "
            f"Fold {fold}/"
            f"{config.n_splits}"
        )

        print(
            "="
            * 80
        )

        X_train = X_last[
            outer_train
        ]

        X_test = X_last[
            outer_test
        ]

        y_train = labels[
            outer_train
        ]

        y_test = labels[
            outer_test
        ]

        g_train = groups[
            outer_train
        ]

        best, search_df = select_hyperparams(
            X=X_train,
            y=y_train,
            groups=g_train,
            experiment=experiment,
            config=config,
            seed=(
                config.seed
                + 1000
                * fold
            ),
            prefix=(
                f"[{experiment.name} "
                f"Fold {fold}] "
            ),
        )

        search_df.to_csv(
            exp_dir
            / f"fold_{fold:02d}_inner_search.csv",
            index=False,
        )

        bundle, scores = fit_outer_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            experiment=experiment,
            params=best,
            config=config,
            seed=(
                config.seed
                + 10_000
                + fold
            ),
        )

        metrics = compute_metrics(
            y_true=y_test,
            y_score=scores,
        )

        expected_dim = expected_feature_dim(
            n_layers=n_layers,
            pca_dim=best[
                "pca_dim"
            ],
            experiment=experiment,
        )

        assert (
            expected_dim
            == bundle[
                "feature_dim"
            ]
        )

        metrics.update(
            {
                "experiment": (
                    experiment.name
                ),
                "use_norm": (
                    experiment.use_norm
                ),
                "use_cosine": (
                    experiment.use_cosine
                ),
                "fold": fold,
                "pca_dim": best[
                    "pca_dim"
                ],
                "C": best[
                    "C"
                ],
                "gamma": gamma_to_string(
                    best[
                        "gamma"
                    ]
                ),
                "feature_dim": (
                    bundle[
                        "feature_dim"
                    ]
                ),
                "inner_balanced_accuracy": (
                    best[
                        "inner_balanced_accuracy"
                    ]
                ),
                "inner_auroc": (
                    best[
                        "inner_auroc"
                    ]
                ),
            }
        )

        fold_rows.append(
            metrics
        )

        y_pred = (
            scores
            >= 0.0
        ).astype(
            int
        )

        for (
            idx,
            true,
            score,
            pred,
        ) in zip(
            outer_test,
            y_test,
            scores,
            y_pred,
        ):
            prediction_rows.append(
                {
                    "experiment": (
                        experiment.name
                    ),
                    "fold": fold,
                    "sample_id": (
                        sample_ids[
                            int(idx)
                        ]
                    ),
                    "group": (
                        groups[
                            int(idx)
                        ]
                    ),
                    "y_true": int(
                        true
                    ),
                    "decision_score": float(
                        score
                    ),
                    "y_pred": int(
                        pred
                    ),
                }
            )

        print(
            f"[OUTER TEST] "
            f"PCA={best['pca_dim']} | "
            f"feature_dim="
            f"{bundle['feature_dim']} | "
            f"acc="
            f"{metrics['accuracy']:.4f} | "
            f"bal_acc="
            f"{metrics['balanced_accuracy']:.4f} | "
            f"f1="
            f"{metrics['f1']:.4f} | "
            f"auroc="
            f"{metrics['auroc']:.4f}"
        )

        if config.save_fold_models:
            joblib.dump(
                {
                    "bundle": (
                        bundle
                    ),
                    "config": asdict(
                        config
                    ),
                    "metrics": (
                        metrics
                    ),
                },
                exp_dir
                / f"fold_{fold:02d}_model.joblib",
            )

    fold_df = pd.DataFrame(
        fold_rows
    )

    pred_df = pd.DataFrame(
        prediction_rows
    )

    fold_df.to_csv(
        exp_dir
        / "fold_metrics.csv",
        index=False,
    )

    pred_df.to_csv(
        exp_dir
        / "oof_predictions.csv",
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

    return {
        "experiment": (
            experiment
        ),
        "fold_results": (
            fold_df
        ),
        "predictions": (
            pred_df
        ),
        "oof": (
            oof_metrics
        ),
    }


# ============================================================
# Comparison
# ============================================================

def make_comparison_table(
    results: List[Dict],
    output_dir: Path,
) -> pd.DataFrame:

    rows = []

    for result in results:
        exp = result[
            "experiment"
        ]

        fold_df = result[
            "fold_results"
        ]

        oof = result[
            "oof"
        ]

        row = {
            "experiment": (
                exp.name
            ),
            "use_norm": (
                exp.use_norm
            ),
            "use_cosine": (
                exp.use_cosine
            ),
        }

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auroc",
        ]:
            values = fold_df[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"fold_{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"fold_{metric}_std"
            ] = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else 0.0
            )

            row[
                f"oof_{metric}"
            ] = float(
                oof[
                    metric
                ]
            )

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        output_dir
        / "geometry_ablation_summary.csv",
        index=False,
    )

    return df


# ============================================================
# Main
# ============================================================

def main():
    config = Config()

    set_seed(
        config.seed
    )

    output_dir = Path(
        config.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(
                config
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    loader = ActivationLoader(
        config.artifact_dir,
        mmap_mode="r",
        verify_files=True,
    )

    print(
        loader
    )

    labels = get_labels(
        loader
    )

    groups = get_groups(
        loader
    )

    (
        X_last,
        sample_ids,
        n_layers,
    ) = preload_last(
        loader,
        config,
    )

    print(
        f"Samples="
        f"{len(labels)} | "
        f"Robust="
        f"{int((labels == 1).sum())} | "
        f"Spurious="
        f"{int((labels == 0).sum())} | "
        f"groups="
        f"{len(np.unique(groups))}"
    )

    # --------------------------------------------------------
    # 모든 ablation이 동일한 outer folds를 공유
    # --------------------------------------------------------
    outer_cv = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    outer_splits = list(
        outer_cv.split(
            np.zeros(
                len(labels)
            ),
            labels,
            groups=groups,
        )
    )

    all_results = []

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment=experiment,
            X_last=X_last,
            labels=labels,
            groups=groups,
            sample_ids=sample_ids,
            outer_splits=outer_splits,
            config=config,
            n_layers=n_layers,
        )

        all_results.append(
            result
        )

    comparison_df = make_comparison_table(
        results=all_results,
        output_dir=output_dir,
    )

    print(
        "\n"
        + "#"
        * 120
    )

    print(
        "GEOMETRY FEATURE ABLATION"
    )

    print(
        "#"
        * 120
    )

    display_cols = [
        "experiment",
        "use_norm",
        "use_cosine",
        "oof_accuracy",
        "oof_balanced_accuracy",
        "oof_f1",
        "oof_auroc",
        "fold_balanced_accuracy_mean",
        "fold_balanced_accuracy_std",
        "fold_auroc_mean",
        "fold_auroc_std",
    ]

    print(
        comparison_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nFeature dimensions "
        "(if PCA dimension = P):"
    )

    print(
        f"  all features   : "
        f"{n_layers}P + "
        f"{n_layers + (n_layers - 1)}"
    )

    print(
        f"  no norm        : "
        f"{n_layers}P + "
        f"{n_layers - 1}"
    )

    print(
        f"  no cosine      : "
        f"{n_layers}P + "
        f"{n_layers}"
    )

    print(
        f"  no geometry    : "
        f"{n_layers}P"
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "geometry_ablation_summary.csv"
    )


if __name__ == "__main__":
    main()
