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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# 이 파일을 lodder.py와 같은 디렉터리에 둔다고 가정합니다.
from lodder import ActivationLoader


GammaType = Union[str, float]


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    artifact_dir: str = "artifacts/deepseek_r1_qwen3_8b_tat"
    output_dir: str = "artifacts/tat_pca_ablation"

    # 현재 activation artifact 구조
    # "tat" -> [T * L, hidden_size] displacement trajectory
    mode: str = "tat"

    # loader.num_layers가 있으면 자동 사용
    n_layers: int | None = None

    # Nested Group CV
    n_splits: int = 5
    inner_splits: int = 3
    seed: int = 42

    # 이전 PCA + RBF-SVM 실험과 동일하게 유지
    pca_dims: Tuple[int, ...] = (8, 16, 32)

    # RBF-SVM search
    rbf_c_values: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    rbf_gamma_values: Tuple[GammaType, ...] = (
        "scale",
        1e-4,
        1e-3,
        1e-2,
    )

    # Linear SVM / Logistic Regression search
    linear_c_values: Tuple[float, ...] = (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )

    # class imbalance 대응
    class_weight: str | None = "balanced"

    # 현재 mean_last + PCA + RBF 실험은 geometry feature를 포함했음.
    # 공정 비교를 위해 모든 실험에서 동일하게 유지합니다.
    # 순수 PCA-only 비교를 원하면 False로 바꾸면 됩니다.
    include_geometry: bool = True

    pca_iterated_power: int = 5
    svm_cache_size_mb: int = 2048

    # 비교 실험에서는 OOF 성능이 핵심이므로 기본 False.
    # True로 하면 각 experiment별 full-data final model도 저장합니다.
    fit_final_models: bool = False

    # outer fold별 모델 저장 여부
    save_fold_models: bool = True


# ============================================================
# Experiment definitions
# ============================================================

@dataclass(frozen=True)
class Experiment:
    name: str
    pool: str
    classifier: str


EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment(
        name="01_mean_last_pca_rbf_svm",
        pool="mean_last",
        classifier="rbf",
    ),
    Experiment(
        name="02_last_pca_rbf_svm",
        pool="last",
        classifier="rbf",
    ),
    Experiment(
        name="03_mean_pca_rbf_svm",
        pool="mean",
        classifier="rbf",
    ),
    Experiment(
        name="04_mean_last_pca_linear_svm",
        pool="mean_last",
        classifier="linear_svm",
    ),
    Experiment(
        name="05_mean_last_pca_logreg",
        pool="mean_last",
        classifier="logreg",
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
    return np.asarray(x, dtype=np.float32)


def get_labels(loader: ActivationLoader) -> np.ndarray:
    labels = []

    for i in range(len(loader)):
        label = loader.get_metadata(i)["label"]
        labels.append(int(bool(label)))

    return np.asarray(labels, dtype=np.int64)


def get_groups(loader: ActivationLoader) -> np.ndarray:
    """
    같은 problem_hash를 반드시 같은 fold에 넣습니다.
    원본/중복 문제가 train/test 양쪽으로 새는 것을 막습니다.
    """
    groups = []

    for i in range(len(loader)):
        meta = loader.get_metadata(i)

        group = meta.get("problem_hash")

        if group is None or (
            isinstance(group, float) and np.isnan(group)
        ):
            group = meta["sample_id"]

        groups.append(str(group))

    return np.asarray(groups)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    모든 classifier에서 decision_function을 사용합니다.

    SVM:
        score >= 0 -> positive

    Logistic Regression:
        log-odds >= 0 -> p >= 0.5 -> positive

    따라서 세 classifier 모두 threshold=0으로 통일 가능합니다.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    y_pred = (y_score >= 0.0).astype(np.int64)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(
            roc_auc_score(y_true, y_score)
        )
    else:
        metrics["auroc"] = float("nan")

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
# Preload TaT trajectory once
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
        "Set Config.n_layers manually, e.g. n_layers=36."
    )


def tat_to_mean_last(
    x: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """
    TaT token-major trajectory:
        [T * L, D]

    -> reshape:
        [T, L, D]

    -> return:
        [2, L, D]

        channel 0 = mean over tokens
        channel 1 = last token

    이 한 번의 preload 결과에서 mean / last / mean_last를 모두 만들 수 있습니다.
    """
    x = to_numpy_float32(x)

    if x.ndim != 2:
        raise ValueError(
            f"Expected [T*L, D], got {x.shape}"
        )

    m, hidden = x.shape

    if m % n_layers != 0:
        raise ValueError(
            f"Trajectory length {m} is not divisible by "
            f"n_layers={n_layers}. "
            "Check trajectory flatten order / loader.num_layers."
        )

    n_tokens = m // n_layers

    grid = x.reshape(
        n_tokens,
        n_layers,
        hidden,
    )

    mean_layer = grid.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    last_layer = grid[-1].astype(
        np.float32,
        copy=False,
    )

    return np.stack(
        [mean_layer, last_layer],
        axis=0,
    )


def preload_all(
    loader: ActivationLoader,
    config: Config,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Returns
    -------
    X_mean_last:
        [N, 2, L, D]
    sample_ids:
        list[str]
    n_layers:
        int
    """
    if config.mode != "tat":
        raise ValueError(
            "This comparison script expects Config.mode='tat'. "
            "It derives mean/last/mean_last from the same full trajectory."
        )

    n_layers = resolve_n_layers(
        loader,
        config,
    )

    xs: List[np.ndarray] = []
    sample_ids: List[str] = []

    expected_shape = None

    print(
        f"Preloading TaT once | "
        f"mode={config.mode} | "
        f"n_layers={n_layers}"
    )

    for i in range(len(loader)):
        sample = loader.load_sample(
            i,
            mode=config.mode,
            as_torch=True,
        )

        fixed = tat_to_mean_last(
            sample["x"],
            n_layers=n_layers,
        )

        if expected_shape is None:
            expected_shape = fixed.shape
        elif fixed.shape != expected_shape:
            raise ValueError(
                f"Inconsistent sample shape at {i}: "
                f"{fixed.shape} != {expected_shape}"
            )

        xs.append(fixed)
        sample_ids.append(
            str(sample["sample_id"])
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
        "Base tensor:",
        X.shape,
        "(N, 2[mean,last], layers, hidden)"
    )

    return X, sample_ids, n_layers


def select_pool(
    X_mean_last: np.ndarray,
    pool: str,
) -> np.ndarray:
    """
    Input:
        [N, 2, L, D]
        channel 0 = mean
        channel 1 = last

    Output:
        mean      -> [N, 1, L, D]
        last      -> [N, 1, L, D]
        mean_last -> [N, 2, L, D]
    """
    if pool == "mean":
        return X_mean_last[:, 0:1]

    if pool == "last":
        return X_mean_last[:, 1:2]

    if pool == "mean_last":
        return X_mean_last

    raise ValueError(
        f"Unknown pool={pool!r}"
    )


# ============================================================
# PCA / geometry features
# ============================================================

def fit_pca(
    X_train: np.ndarray,
    n_components: int,
    seed: int,
    iterated_power: int,
) -> PCA:
    """
    X_train: [N, C, L, D]

    PCA는 반드시 해당 training split의
    [N*C*L, D] vector만 보고 학습합니다.
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
            f"PCA n_components={n_components} exceeds "
            f"max_allowed={max_allowed}."
        )

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=iterated_power,
    )

    pca.fit(rows)

    return pca


def transform_pca(
    pca: PCA,
    X: np.ndarray,
) -> np.ndarray:
    """
    [N,C,L,D] -> [N,C,L,K]
    """
    n, c, l, d = X.shape

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
        c,
        l,
        -1,
    )


def geometry_features(
    X: np.ndarray,
) -> np.ndarray:
    """
    Auxiliary geometric features.

    1) displacement norm per layer
       [N,C,L]

    2) cosine between adjacent layer updates
       [N,C,L-1]

    이것만 단독으로 쓰는 것이 아니라,
    PCA directional feature 뒤에 보조적으로 붙입니다.
    """
    eps = 1e-8

    norms = np.linalg.norm(
        X,
        axis=-1,
    )

    prev = X[:, :, :-1, :]
    nxt = X[:, :, 1:, :]

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

    cosines = numerator / denominator

    geom = np.concatenate(
        [
            norms.reshape(
                X.shape[0],
                -1,
            ),
            cosines.reshape(
                X.shape[0],
                -1,
            ),
        ],
        axis=1,
    )

    return geom.astype(
        np.float32,
        copy=False,
    )


def build_features(
    Z: np.ndarray,
    pca_dim: int,
    X_raw: np.ndarray,
    include_geometry: bool,
) -> np.ndarray:
    """
    PCA component 순서를 유지하고
    channel -> layer -> PC 순으로 flatten합니다.
    """
    directional = Z[
        ...,
        :pca_dim,
    ].reshape(
        Z.shape[0],
        -1,
    )

    if not include_geometry:
        return directional.astype(
            np.float32,
            copy=False,
        )

    geom = geometry_features(
        X_raw
    )

    return np.concatenate(
        [
            directional,
            geom,
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# Classifiers
# ============================================================

def make_classifier(
    classifier: str,
    C: float,
    gamma: GammaType | None,
    config: Config,
    seed: int,
):
    if classifier == "rbf":
        return SVC(
            C=float(C),
            kernel="rbf",
            gamma=gamma,
            class_weight=config.class_weight,
            cache_size=float(
                config.svm_cache_size_mb
            ),
            random_state=seed,
        )

    if classifier == "linear_svm":
        return LinearSVC(
            C=float(C),
            class_weight=config.class_weight,
            dual="auto",
            max_iter=20_000,
            random_state=seed,
        )

    if classifier == "logreg":
        return LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="liblinear",
            class_weight=config.class_weight,
            max_iter=10_000,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown classifier={classifier!r}"
    )


def candidate_params(
    experiment: Experiment,
    config: Config,
) -> List[Dict]:
    params = []

    if experiment.classifier == "rbf":
        for pca_dim in config.pca_dims:
            for C in config.rbf_c_values:
                for gamma in config.rbf_gamma_values:
                    params.append(
                        {
                            "pca_dim": int(pca_dim),
                            "C": float(C),
                            "gamma": gamma,
                        }
                    )
        return params

    if experiment.classifier in {
        "linear_svm",
        "logreg",
    }:
        for pca_dim in config.pca_dims:
            for C in config.linear_c_values:
                params.append(
                    {
                        "pca_dim": int(pca_dim),
                        "C": float(C),
                        "gamma": None,
                    }
                )
        return params

    raise ValueError(
        experiment.classifier
    )


def gamma_to_string(
    gamma: GammaType | None,
) -> str:
    if gamma is None:
        return ""
    return str(gamma)


# ============================================================
# Inner nested-CV model selection
# ============================================================

def make_inner_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    seed: int,
):
    unique_groups = np.unique(groups)

    groups_per_class = []

    for cls in [0, 1]:
        groups_per_class.append(
            len(
                np.unique(
                    groups[y == cls]
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
            np.zeros(len(y)),
            y,
            groups=groups,
        )
    )


def select_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    experiment: Experiment,
    config: Config,
    seed: int,
    prefix: str,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Leakage-safe inner model selection.

    각 inner fold에서:
        inner train
            -> PCA.fit
            -> feature construction
            -> StandardScaler.fit
            -> classifier.fit

        inner val
            -> transform only

    PCA/scaler가 inner validation을 보지 않습니다.
    """
    inner_splits = make_inner_splits(
        y=y,
        groups=groups,
        requested_splits=config.inner_splits,
        seed=seed,
    )

    max_pca_dim = max(
        config.pca_dims
    )

    param_list = candidate_params(
        experiment,
        config,
    )

    # 각 parameter config에 대한 metric list
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

    # PCA는 같은 fold에서 max component로 한 번만 fit하고
    # 앞쪽 8/16/32 component를 slice하여 사용합니다.
    for inner_fold, (
        tr_idx,
        va_idx,
    ) in enumerate(
        inner_splits,
        start=1,
    ):
        X_tr = X[tr_idx]
        X_va = X[va_idx]

        y_tr = y[tr_idx]
        y_va = y[va_idx]

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

        # pca_dim별 scaler도 한 번만 계산
        feature_cache = {}

        for pca_dim in config.pca_dims:
            F_tr = build_features(
                Z=Z_tr,
                pca_dim=pca_dim,
                X_raw=X_tr,
                include_geometry=config.include_geometry,
            )

            F_va = build_features(
                Z=Z_va,
                pca_dim=pca_dim,
                X_raw=X_va,
                include_geometry=config.include_geometry,
            )

            scaler = StandardScaler()

            F_tr_s = scaler.fit_transform(
                F_tr
            )
            F_va_s = scaler.transform(
                F_va
            )

            feature_cache[pca_dim] = (
                F_tr_s,
                F_va_s,
            )

        for params in param_list:
            pca_dim = params["pca_dim"]

            F_tr_s, F_va_s = feature_cache[
                pca_dim
            ]

            clf = make_classifier(
                classifier=experiment.classifier,
                C=params["C"],
                gamma=params["gamma"],
                config=config,
                seed=seed + inner_fold,
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
                params["pca_dim"],
                params["C"],
                gamma_to_string(
                    params["gamma"]
                ),
            )

            store[key].append(
                metrics
            )

        print(
            f"{prefix}inner fold "
            f"{inner_fold}/{len(inner_splits)} done"
        )

    rows = []

    for (
        pca_dim,
        C,
        gamma_string,
    ), metric_list in store.items():
        bal = np.asarray(
            [
                m["balanced_accuracy"]
                for m in metric_list
            ],
            dtype=float,
        )

        auc = np.asarray(
            [
                m["auroc"]
                for m in metric_list
            ],
            dtype=float,
        )

        f1 = np.asarray(
            [
                m["f1"]
                for m in metric_list
            ],
            dtype=float,
        )

        rows.append(
            {
                "pca_dim": int(pca_dim),
                "C": float(C),
                "gamma": gamma_string,
                "mean_balanced_accuracy": float(
                    np.nanmean(bal)
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
                    np.nanmean(auc)
                ),
                "mean_f1": float(
                    np.nanmean(f1)
                ),
                "n_inner_folds": len(
                    metric_list
                ),
            }
        )

    search_df = pd.DataFrame(
        rows
    )

    # 1순위: balanced accuracy
    # 2순위: AUROC
    # 3순위: fold variance가 작은 것
    # 4순위: 낮은 PCA 차원
    # 5순위: 작은 C
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

    best_row = search_df.iloc[0]

    gamma_text = str(
        best_row["gamma"]
    )

    if experiment.classifier == "rbf":
        if gamma_text == "scale":
            gamma: GammaType | None = "scale"
        else:
            gamma = float(
                gamma_text
            )
    else:
        gamma = None

    best = {
        "pca_dim": int(
            best_row["pca_dim"]
        ),
        "C": float(
            best_row["C"]
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

    return best, search_df


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
        params["pca_dim"]
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
        pca_dim=pca_dim,
        X_raw=X_train,
        include_geometry=config.include_geometry,
    )

    F_test = build_features(
        Z=Z_test,
        pca_dim=pca_dim,
        X_raw=X_test,
        include_geometry=config.include_geometry,
    )

    scaler = StandardScaler()

    F_train_s = scaler.fit_transform(
        F_train
    )

    F_test_s = scaler.transform(
        F_test
    )

    clf = make_classifier(
        classifier=experiment.classifier,
        C=params["C"],
        gamma=params["gamma"],
        config=config,
        seed=seed,
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

    return bundle, test_scores


# ============================================================
# One full nested-CV experiment
# ============================================================

def run_experiment(
    experiment: Experiment,
    X_mean_last: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: List[str],
    outer_splits,
    config: Config,
    n_layers: int,
) -> Dict:
    exp_dir = (
        Path(config.output_dir)
        / experiment.name
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X = select_pool(
        X_mean_last,
        experiment.pool,
    )

    print("\n" + "#" * 88)
    print(
        f"EXPERIMENT: {experiment.name}"
    )
    print(
        f"pool={experiment.pool} | "
        f"classifier={experiment.classifier} | "
        f"tensor={X.shape}"
    )
    print("#" * 88)

    fold_results = []
    prediction_rows = []

    for fold, (
        outer_train,
        outer_test,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"{experiment.name} | "
            f"Fold {fold}/{config.n_splits}"
        )
        print("=" * 80)

        X_train = X[
            outer_train
        ]
        X_test = X[
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

        print(
            f"train={len(outer_train)} | "
            f"test={len(outer_test)} | "
            f"train_pos={int(y_train.sum())} | "
            f"test_pos={int(y_test.sum())}"
        )

        best, search_df = select_hyperparams(
            X=X_train,
            y=y_train,
            groups=g_train,
            experiment=experiment,
            config=config,
            seed=(
                config.seed
                + 1000 * fold
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

        print(
            f"selected | "
            f"PCA={best['pca_dim']} | "
            f"C={best['C']} | "
            f"gamma={best['gamma']} | "
            f"inner_bal_acc="
            f"{best['inner_balanced_accuracy']:.4f} | "
            f"inner_auc="
            f"{best['inner_auroc']:.4f}"
        )

        bundle, test_scores = fit_outer_model(
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
            y_score=test_scores,
        )

        metrics.update(
            {
                "experiment": experiment.name,
                "pool": experiment.pool,
                "classifier": experiment.classifier,
                "fold": fold,
                "n_train": len(
                    outer_train
                ),
                "n_test": len(
                    outer_test
                ),
                "pca_dim": best[
                    "pca_dim"
                ],
                "C": best["C"],
                "gamma": gamma_to_string(
                    best["gamma"]
                ),
                "feature_dim": bundle[
                    "feature_dim"
                ],
                "inner_balanced_accuracy": best[
                    "inner_balanced_accuracy"
                ],
                "inner_auroc": best[
                    "inner_auroc"
                ],
                "pca_explained_variance_ratio_sum": float(
                    np.sum(
                        bundle[
                            "pca"
                        ].explained_variance_ratio_
                    )
                ),
            }
        )

        fold_results.append(
            metrics
        )

        y_pred = (
            test_scores >= 0.0
        ).astype(int)

        for idx, true, score, pred in zip(
            outer_test,
            y_test,
            test_scores,
            y_pred,
        ):
            prediction_rows.append(
                {
                    "experiment": experiment.name,
                    "fold": fold,
                    "sample_id": sample_ids[
                        int(idx)
                    ],
                    "group": groups[
                        int(idx)
                    ],
                    "y_true": int(true),
                    "decision_score": float(
                        score
                    ),
                    "y_pred": int(pred),
                }
            )

        print(
            f"[OUTER TEST] "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"balanced_acc="
            f"{metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auroc={metrics['auroc']:.4f}"
        )

        if config.save_fold_models:
            joblib.dump(
                {
                    "bundle": bundle,
                    "config": asdict(
                        config
                    ),
                    "n_layers": n_layers,
                    "metrics": metrics,
                },
                exp_dir
                / f"fold_{fold:02d}_model.joblib",
            )

    results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    results_df.to_csv(
        exp_dir
        / "fold_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        exp_dir
        / "oof_predictions.csv",
        index=False,
    )

    oof_metrics = compute_metrics(
        y_true=predictions_df[
            "y_true"
        ].to_numpy(),
        y_score=predictions_df[
            "decision_score"
        ].to_numpy(),
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "auroc",
    ]

    summary = {
        "experiment": asdict(
            experiment
        ),
        "config": asdict(
            config
        ),
        "oof": oof_metrics,
    }

    print("\n" + "-" * 80)
    print(
        f"SUMMARY: {experiment.name}"
    )
    print("-" * 80)

    for metric in metric_names:
        values = results_df[
            metric
        ].to_numpy(
            dtype=float
        )

        mean = float(
            np.nanmean(values)
        )

        std = float(
            np.nanstd(
                values,
                ddof=1,
            )
            if len(values) > 1
            else 0.0
        )

        summary[metric] = {
            "mean": mean,
            "std": std,
        }

        print(
            f"{metric:>18s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    print(
        "OOF | "
        f"acc={oof_metrics['accuracy']:.4f} | "
        f"bal_acc="
        f"{oof_metrics['balanced_accuracy']:.4f} | "
        f"f1={oof_metrics['f1']:.4f} | "
        f"auroc={oof_metrics['auroc']:.4f}"
    )

    with (
        exp_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Optional final model
    if config.fit_final_models:
        print(
            f"\n[{experiment.name}] "
            "Selecting final full-data model..."
        )

        best, final_search = select_hyperparams(
            X=X,
            y=labels,
            groups=groups,
            experiment=experiment,
            config=config,
            seed=(
                config.seed
                + 999_999
            ),
            prefix=(
                f"[{experiment.name} Final] "
            ),
        )

        final_search.to_csv(
            exp_dir
            / "final_inner_search.csv",
            index=False,
        )

        pca_dim = int(
            best["pca_dim"]
        )

        pca = fit_pca(
            X_train=X,
            n_components=pca_dim,
            seed=(
                config.seed
                + 888_888
            ),
            iterated_power=config.pca_iterated_power,
        )

        Z = transform_pca(
            pca,
            X,
        )

        F = build_features(
            Z=Z,
            pca_dim=pca_dim,
            X_raw=X,
            include_geometry=config.include_geometry,
        )

        scaler = StandardScaler()
        F_s = scaler.fit_transform(
            F
        )

        clf = make_classifier(
            classifier=experiment.classifier,
            C=best["C"],
            gamma=best["gamma"],
            config=config,
            seed=(
                config.seed
                + 777_777
            ),
        )

        clf.fit(
            F_s,
            labels,
        )

        joblib.dump(
            {
                "pca": pca,
                "scaler": scaler,
                "classifier": clf,
                "experiment": asdict(
                    experiment
                ),
                "params": best,
                "config": asdict(
                    config
                ),
                "n_layers": n_layers,
                "feature_dim": int(
                    F.shape[1]
                ),
            },
            exp_dir
            / "final_model.joblib",
        )

    return {
        "experiment": experiment,
        "fold_results": results_df,
        "predictions": predictions_df,
        "summary": summary,
    }


# ============================================================
# Combined comparison report
# ============================================================

def make_comparison_table(
    all_results: List[Dict],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    for result in all_results:
        exp = result[
            "experiment"
        ]

        fold_df = result[
            "fold_results"
        ]

        oof = result[
            "summary"
        ]["oof"]

        row = {
            "experiment": exp.name,
            "pool": exp.pool,
            "classifier": exp.classifier,
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
                np.nanmean(values)
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
                oof[metric]
            )

        rows.append(
            row
        )

    comparison_df = pd.DataFrame(
        rows
    )

    # 표시 편의상 OOF balanced accuracy,
    # OOF AUROC 순으로 정렬한 별도 파일도 저장
    comparison_df.to_csv(
        output_dir
        / "comparison_summary_original_order.csv",
        index=False,
    )

    ranked = comparison_df.sort_values(
        by=[
            "oof_balanced_accuracy",
            "oof_auroc",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    ranked.to_csv(
        output_dir
        / "comparison_summary.csv",
        index=False,
    )

    # sample별 예측을 wide format으로도 저장
    merged = None

    for result in all_results:
        exp_name = result[
            "experiment"
        ].name

        df = result[
            "predictions"
        ][
            [
                "sample_id",
                "group",
                "y_true",
                "fold",
                "decision_score",
                "y_pred",
            ]
        ].copy()

        df = df.rename(
            columns={
                "decision_score": (
                    f"{exp_name}__score"
                ),
                "y_pred": (
                    f"{exp_name}__pred"
                ),
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[
                    [
                        "sample_id",
                        f"{exp_name}__score",
                        f"{exp_name}__pred",
                    ]
                ],
                on="sample_id",
                how="inner",
            )

    if merged is not None:
        merged.to_csv(
            output_dir
            / "all_oof_predictions_wide.csv",
            index=False,
        )

    return ranked


# ============================================================
# Main
# ============================================================

def main():
    config = Config(
        artifact_dir=(
            "artifacts/"
            "deepseek_r1_qwen3_8b_tat"
        ),
        output_dir=(
            "artifacts/"
            "tat_pca_ablation"
        ),

        mode="tat",

        # loader.num_layers가 있으면 자동 사용.
        # 문제가 있으면 36으로 고정:
        n_layers=None,

        n_splits=5,
        inner_splits=3,
        seed=42,

        # 현재 실험과 같은 PCA search
        pca_dims=(
            8,
            16,
            32,
        ),

        rbf_c_values=(
            0.1,
            1.0,
            10.0,
            100.0,
        ),

        rbf_gamma_values=(
            "scale",
            1e-4,
            1e-3,
            1e-2,
        ),

        linear_c_values=(
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ),

        class_weight="balanced",

        # 기존 PCA+RBF 결과와 공정하게 맞춤
        include_geometry=True,

        pca_iterated_power=5,
        svm_cache_size_mb=2048,

        # 비교만 할 때는 False가 빠릅니다.
        fit_final_models=False,

        save_fold_models=True,
    )

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

    loader = ActivationLoader(
        config.artifact_dir,
        mmap_mode="r",
        verify_files=True,
    )

    print(loader)

    labels = get_labels(
        loader
    )

    groups = get_groups(
        loader
    )

    X_mean_last, sample_ids, n_layers = preload_all(
        loader=loader,
        config=config,
    )

    print(
        f"Samples={len(labels)} | "
        f"Robust={int((labels == 1).sum())} | "
        f"Spurious={int((labels == 0).sum())} | "
        f"groups={len(np.unique(groups))}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # 모든 experiment가 정확히 같은 outer fold를 사용합니다.
    # classifier/pooling 비교가 split 운에 의해 흔들리는 것을 줄입니다.
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

    # split 정보 저장
    split_rows = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        for idx in train_idx:
            split_rows.append(
                {
                    "sample_id": sample_ids[
                        int(idx)
                    ],
                    "fold": fold,
                    "split": "train",
                }
            )

        for idx in test_idx:
            split_rows.append(
                {
                    "sample_id": sample_ids[
                        int(idx)
                    ],
                    "fold": fold,
                    "split": "test",
                }
            )

    pd.DataFrame(
        split_rows
    ).to_csv(
        output_dir
        / "outer_splits.csv",
        index=False,
    )

    all_results = []

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment=experiment,
            X_mean_last=X_mean_last,
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
        all_results=all_results,
        output_dir=output_dir,
    )

    print("\n" + "#" * 100)
    print("FINAL COMPARISON")
    print("#" * 100)

    display_cols = [
        "experiment",
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
        "\nSaved comparison to:"
    )
    print(
        output_dir
        / "comparison_summary.csv"
    )
    print(
        output_dir
        / "all_oof_predictions_wide.csv"
    )


if __name__ == "__main__":
    main()
