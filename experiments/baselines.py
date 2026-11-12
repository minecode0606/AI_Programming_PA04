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
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset

# 이 파일을 lodder.py와 같은 디렉터리에 둔다고 가정합니다.
from lodder import ActivationLoader


GammaType = Union[str, float]


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    artifact_dir: str = "artifacts/deepseek_r1_qwen3_8b_tat"
    output_dir: str = "artifacts/last_rbf_vs_lstm"

    mode: str = "tat"
    n_layers: int | None = None

    # 동일 outer Group CV
    n_splits: int = 5
    inner_splits: int = 3
    seed: int = 42

    # --------------------------------------------------------
    # 02_last_pca_rbf_svm 설정
    # --------------------------------------------------------
    pca_dims: Tuple[int, ...] = (8, 16, 32)
    rbf_c_values: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    rbf_gamma_values: Tuple[GammaType, ...] = (
        "scale",
        1e-4,
        1e-3,
        1e-2,
    )

    class_weight: str | None = "balanced"
    include_geometry: bool = True
    pca_iterated_power: int = 5
    svm_cache_size_mb: int = 2048

    # --------------------------------------------------------
    # Plain LSTM 설정
    # input = last-token layer trajectory [L, D]
    # --------------------------------------------------------
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_bidirectional: bool = False

    lstm_lr: float = 1e-3
    lstm_weight_decay: float = 1e-4
    lstm_batch_size: int = 16
    lstm_max_epochs: int = 150
    lstm_patience: int = 20
    lstm_grad_clip: float = 1.0

    # inner validation에서 이 metric으로 best epoch 결정
    lstm_selection_metric: str = "balanced_accuracy"

    save_fold_models: bool = True


# ============================================================
# Reproducibility / metadata
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    같은 problem_hash를 같은 fold에 넣습니다.
    problem_hash가 없으면 sample_id를 group으로 사용합니다.
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
    두 방법 모두 raw decision score/logit을 사용합니다.

    RBF-SVM:
        decision_function >= 0 -> class 1

    LSTM:
        logit >= 0 -> sigmoid >= 0.5 -> class 1
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
# Load exactly the LAST-token trajectory
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


def tat_to_last(
    x: np.ndarray,
    n_layers: int,
) -> np.ndarray:
    """
    Input
    -----
    TaT token-major trajectory:
        [T * L, D]

    Reshape
    -------
        [T, L, D]

    Return
    ------
    마지막 token의 layer trajectory:
        [L, D]

    이것이 기존 02_last_pca_rbf_svm의 pool='last'와 동일한 입력입니다.
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
    """
    Returns
    -------
    X_last:
        [N, L, D]
    sample_ids:
        list[str]
    n_layers:
        int
    """
    if config.mode != "tat":
        raise ValueError(
            "This comparison expects Config.mode='tat'."
        )

    n_layers = resolve_n_layers(
        loader,
        config,
    )

    xs: List[np.ndarray] = []
    sample_ids: List[str] = []
    expected_shape = None

    print(
        f"Preloading LAST-token trajectories | "
        f"mode={config.mode} | n_layers={n_layers}"
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
                f"Inconsistent sample shape at {i}: "
                f"{x_last.shape} != {expected_shape}"
            )

        xs.append(x_last)
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
        "X_last:",
        X.shape,
        "(N, layers, hidden)"
    )

    return X, sample_ids, n_layers


# ============================================================
# Shared inner split utility
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


# ============================================================
# 02_last_pca_rbf_svm
# ============================================================

def fit_pca(
    X_train: np.ndarray,
    n_components: int,
    seed: int,
    iterated_power: int,
) -> PCA:
    """
    X_train: [N, 1, L, D]

    PCA는 training split의 [N*L, D]만 보고 fit합니다.
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
    기존 코드와 동일:
    - layer별 norm
    - 인접 layer trajectory cosine
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


def build_rbf_features(
    Z: np.ndarray,
    pca_dim: int,
    X_raw: np.ndarray,
    include_geometry: bool,
) -> np.ndarray:
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


def gamma_to_string(
    gamma: GammaType | None,
) -> str:
    if gamma is None:
        return ""
    return str(gamma)


def select_rbf_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Config,
    seed: int,
    prefix: str,
) -> Tuple[Dict, pd.DataFrame]:
    """
    기존 02_last_pca_rbf_svm과 동일한 nested inner-CV search.
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

    params = []

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

    store = {
        (
            p["pca_dim"],
            p["C"],
            gamma_to_string(
                p["gamma"]
            ),
        ): []
        for p in params
    }

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

        feature_cache = {}

        for pca_dim in config.pca_dims:
            F_tr = build_rbf_features(
                Z=Z_tr,
                pca_dim=pca_dim,
                X_raw=X_tr,
                include_geometry=config.include_geometry,
            )

            F_va = build_rbf_features(
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

        for p in params:
            F_tr_s, F_va_s = feature_cache[
                p["pca_dim"]
            ]

            clf = SVC(
                C=float(p["C"]),
                kernel="rbf",
                gamma=p["gamma"],
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
                p["pca_dim"],
                p["C"],
                gamma_to_string(
                    p["gamma"]
                ),
            )

            store[key].append(
                metrics
            )

        print(
            f"{prefix}RBF inner fold "
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

    if gamma_text == "scale":
        gamma: GammaType = "scale"
    else:
        gamma = float(
            gamma_text
        )

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


def fit_rbf_outer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
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

    F_train = build_rbf_features(
        Z=Z_train,
        pca_dim=pca_dim,
        X_raw=X_train,
        include_geometry=config.include_geometry,
    )

    F_test = build_rbf_features(
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

    clf = SVC(
        C=float(params["C"]),
        kernel="rbf",
        gamma=params["gamma"],
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
        "params": params,
        "feature_dim": int(
            F_train.shape[1]
        ),
    }

    return bundle, test_scores


# ============================================================
# Plain LSTM baseline
# ============================================================

class PlainLSTMClassifier(nn.Module):
    """
    Input:
        [B, L, D]

    LSTM은 layer trajectory를 시간축(sequence)으로 읽고,
    마지막 recurrent hidden state로 binary classification합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        effective_dropout = (
            float(dropout)
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = (
            2 if bidirectional else 1
        )

        self.head = nn.Linear(
            hidden_size * direction_factor,
            1,
        )

        self.bidirectional = bidirectional

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # 마지막 layer의 forward/backward hidden concat
            h = torch.cat(
                [
                    h_n[-2],
                    h_n[-1],
                ],
                dim=1,
            )
        else:
            h = h_n[-1]

        logits = self.head(
            h
        ).squeeze(-1)

        return logits


def fit_sequence_scaler(
    X_train: np.ndarray,
) -> StandardScaler:
    """
    hidden dimension별 standardization.
    fit은 training rows [N*L, D]에만 수행합니다.
    """
    scaler = StandardScaler()

    scaler.fit(
        X_train.reshape(
            -1,
            X_train.shape[-1],
        )
    )

    return scaler


def transform_sequence(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    n, l, d = X.shape

    z = scaler.transform(
        X.reshape(
            -1,
            d,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    return z.reshape(
        n,
        l,
        d,
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.from_numpy(
        np.asarray(
            X,
            dtype=np.float32,
        )
    )

    if y is None:
        dataset = TensorDataset(
            X_tensor
        )
    else:
        y_tensor = torch.from_numpy(
            np.asarray(
                y,
                dtype=np.float32,
            )
        )

        dataset = TensorDataset(
            X_tensor,
            y_tensor,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def make_pos_weight(
    y: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    n_pos = int(
        np.sum(y == 1)
    )

    n_neg = int(
        np.sum(y == 0)
    )

    if n_pos == 0:
        raise ValueError(
            "Training split has no positive samples."
        )

    value = float(
        n_neg / n_pos
    )

    return torch.tensor(
        [value],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def predict_lstm_scores(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    loader = make_loader(
        X=X,
        y=None,
        batch_size=batch_size,
        shuffle=False,
    )

    scores = []

    for (xb,) in loader:
        xb = xb.to(
            device,
            non_blocking=True,
        )

        logits = model(
            xb
        )

        scores.append(
            logits.detach().cpu().numpy()
        )

    return np.concatenate(
        scores,
        axis=0,
    ).astype(
        np.float64,
        copy=False,
    )


def train_lstm_for_epoch_count(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config,
    n_epochs: int,
    seed: int,
    device: torch.device,
):
    """
    지정된 epoch 수만큼 full outer-train으로 재학습합니다.
    """
    set_seed(seed)

    scaler = fit_sequence_scaler(
        X_train
    )

    X_train_s = transform_sequence(
        scaler,
        X_train,
    )

    model = PlainLSTMClassifier(
        input_size=X_train.shape[-1],
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
        bidirectional=config.lstm_bidirectional,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=make_pos_weight(
            y_train,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_lr,
        weight_decay=config.lstm_weight_decay,
    )

    train_loader = make_loader(
        X=X_train_s,
        y=y_train,
        batch_size=config.lstm_batch_size,
        shuffle=True,
    )

    for _ in range(n_epochs):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            if config.lstm_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.lstm_grad_clip,
                )

            optimizer.step()

    return model, scaler


def select_lstm_epoch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config: Config,
    seed: int,
    device: torch.device,
    prefix: str,
) -> Tuple[int, pd.DataFrame]:
    """
    outer-train 내부의 group-safe split 하나를 사용해
    early stopping epoch를 결정합니다.

    그 뒤 outer-train 전체를 best_epoch만큼 다시 학습하므로
    outer-test는 epoch 선택에 전혀 사용되지 않습니다.
    """
    splits = make_inner_splits(
        y=y_train,
        groups=groups_train,
        requested_splits=config.inner_splits,
        seed=seed,
    )

    fit_idx, val_idx = splits[0]

    X_fit = X_train[
        fit_idx
    ]

    X_val = X_train[
        val_idx
    ]

    y_fit = y_train[
        fit_idx
    ]

    y_val = y_train[
        val_idx
    ]

    scaler = fit_sequence_scaler(
        X_fit
    )

    X_fit_s = transform_sequence(
        scaler,
        X_fit,
    )

    X_val_s = transform_sequence(
        scaler,
        X_val,
    )

    set_seed(seed)

    model = PlainLSTMClassifier(
        input_size=X_train.shape[-1],
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
        bidirectional=config.lstm_bidirectional,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=make_pos_weight(
            y_fit,
            device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lstm_lr,
        weight_decay=config.lstm_weight_decay,
    )

    train_loader = make_loader(
        X=X_fit_s,
        y=y_fit,
        batch_size=config.lstm_batch_size,
        shuffle=True,
    )

    best_epoch = 1
    best_value = -np.inf
    best_auc = -np.inf
    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(
        1,
        config.lstm_max_epochs + 1,
    ):
        model.train()

        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(
                device,
                non_blocking=True,
            )

            yb = yb.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            if config.lstm_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.lstm_grad_clip,
                )

            optimizer.step()

            train_losses.append(
                float(loss.item())
            )

        val_scores = predict_lstm_scores(
            model=model,
            X=X_val_s,
            batch_size=config.lstm_batch_size,
            device=device,
        )

        val_metrics = compute_metrics(
            y_true=y_val,
            y_score=val_scores,
        )

        selection_value = float(
            val_metrics[
                config.lstm_selection_metric
            ]
        )

        auc_value = float(
            val_metrics["auroc"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(
                    np.mean(train_losses)
                ),
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
                "val_balanced_accuracy": val_metrics[
                    "balanced_accuracy"
                ],
                "val_f1": val_metrics[
                    "f1"
                ],
                "val_auroc": val_metrics[
                    "auroc"
                ],
            }
        )

        # 1순위 selection metric
        # 2순위 AUROC
        improved = (
            selection_value > best_value + 1e-12
            or (
                abs(
                    selection_value - best_value
                ) <= 1e-12
                and auc_value > best_auc + 1e-12
            )
        )

        if improved:
            best_value = selection_value
            best_auc = auc_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"{prefix}LSTM epoch={epoch:03d} | "
                f"val_bal_acc="
                f"{val_metrics['balanced_accuracy']:.4f} | "
                f"val_auc={val_metrics['auroc']:.4f} | "
                f"best_epoch={best_epoch}"
            )

        if (
            epochs_without_improvement
            >= config.lstm_patience
        ):
            print(
                f"{prefix}LSTM early stop | "
                f"best_epoch={best_epoch}"
            )
            break

    return (
        int(best_epoch),
        pd.DataFrame(
            history_rows
        ),
    )


def run_lstm_outer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    config: Config,
    seed: int,
    device: torch.device,
    prefix: str,
):
    best_epoch, history_df = select_lstm_epoch(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        config=config,
        seed=seed,
        device=device,
        prefix=prefix,
    )

    # best epoch를 정한 뒤 outer train 전체로 다시 학습
    model, scaler = train_lstm_for_epoch_count(
        X_train=X_train,
        y_train=y_train,
        config=config,
        n_epochs=best_epoch,
        seed=seed + 500_000,
        device=device,
    )

    X_test_s = transform_sequence(
        scaler,
        X_test,
    )

    test_scores = predict_lstm_scores(
        model=model,
        X=X_test_s,
        batch_size=config.lstm_batch_size,
        device=device,
    )

    return (
        model,
        scaler,
        test_scores,
        best_epoch,
        history_df,
    )


# ============================================================
# Run the two methods on exactly the same outer folds
# ============================================================

def run_comparison(
    X_last: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_ids: List[str],
    outer_splits,
    config: Config,
    n_layers: int,
):
    output_dir = Path(
        config.output_dir
    )

    rbf_dir = (
        output_dir
        / "02_last_pca_rbf_svm"
    )

    lstm_dir = (
        output_dir
        / "06_last_plain_lstm"
    )

    rbf_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lstm_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # RBF pipeline은 기존 형식 [N, C=1, L, D]
    X_rbf = X_last[
        :,
        None,
        :,
        :,
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nPyTorch device: {device}"
    )

    rbf_fold_rows = []
    lstm_fold_rows = []

    rbf_pred_rows = []
    lstm_pred_rows = []

    for fold, (
        outer_train,
        outer_test,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"OUTER FOLD {fold}/{config.n_splits}"
        )
        print("=" * 100)

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

        # ----------------------------------------------------
        # A) 02_last_pca_rbf_svm
        # ----------------------------------------------------
        print(
            "\n[A] 02_last_pca_rbf_svm"
        )

        rbf_best, rbf_search_df = select_rbf_hyperparams(
            X=X_rbf[
                outer_train
            ],
            y=y_train,
            groups=g_train,
            config=config,
            seed=(
                config.seed
                + 1000 * fold
            ),
            prefix=(
                f"[Fold {fold}] "
            ),
        )

        rbf_search_df.to_csv(
            rbf_dir
            / f"fold_{fold:02d}_inner_search.csv",
            index=False,
        )

        print(
            f"[Fold {fold}] RBF selected | "
            f"PCA={rbf_best['pca_dim']} | "
            f"C={rbf_best['C']} | "
            f"gamma={rbf_best['gamma']} | "
            f"inner_bal_acc="
            f"{rbf_best['inner_balanced_accuracy']:.4f}"
        )

        rbf_bundle, rbf_scores = fit_rbf_outer_fold(
            X_train=X_rbf[
                outer_train
            ],
            y_train=y_train,
            X_test=X_rbf[
                outer_test
            ],
            params=rbf_best,
            config=config,
            seed=(
                config.seed
                + 10_000
                + fold
            ),
        )

        rbf_metrics = compute_metrics(
            y_true=y_test,
            y_score=rbf_scores,
        )

        rbf_metrics.update(
            {
                "experiment": "02_last_pca_rbf_svm",
                "pool": "last",
                "classifier": "rbf",
                "fold": fold,
                "n_train": len(
                    outer_train
                ),
                "n_test": len(
                    outer_test
                ),
                "pca_dim": rbf_best[
                    "pca_dim"
                ],
                "C": rbf_best["C"],
                "gamma": gamma_to_string(
                    rbf_best["gamma"]
                ),
                "feature_dim": rbf_bundle[
                    "feature_dim"
                ],
            }
        )

        rbf_fold_rows.append(
            rbf_metrics
        )

        rbf_pred = (
            rbf_scores >= 0.0
        ).astype(int)

        for idx, true, score, pred in zip(
            outer_test,
            y_test,
            rbf_scores,
            rbf_pred,
        ):
            rbf_pred_rows.append(
                {
                    "experiment": "02_last_pca_rbf_svm",
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
            f"[RBF OUTER TEST] "
            f"acc={rbf_metrics['accuracy']:.4f} | "
            f"bal_acc="
            f"{rbf_metrics['balanced_accuracy']:.4f} | "
            f"f1={rbf_metrics['f1']:.4f} | "
            f"auroc={rbf_metrics['auroc']:.4f}"
        )

        if config.save_fold_models:
            joblib.dump(
                {
                    "bundle": rbf_bundle,
                    "config": asdict(
                        config
                    ),
                    "n_layers": n_layers,
                    "metrics": rbf_metrics,
                },
                rbf_dir
                / f"fold_{fold:02d}_model.joblib",
            )

        # ----------------------------------------------------
        # B) Plain LSTM on same last-token trajectory
        # ----------------------------------------------------
        print(
            "\n[B] 06_last_plain_lstm"
        )

        (
            lstm_model,
            lstm_scaler,
            lstm_scores,
            best_epoch,
            history_df,
        ) = run_lstm_outer_fold(
            X_train=X_last[
                outer_train
            ],
            y_train=y_train,
            groups_train=g_train,
            X_test=X_last[
                outer_test
            ],
            config=config,
            seed=(
                config.seed
                + 20_000
                + fold
            ),
            device=device,
            prefix=(
                f"[Fold {fold}] "
            ),
        )

        history_df.to_csv(
            lstm_dir
            / f"fold_{fold:02d}_history.csv",
            index=False,
        )

        lstm_metrics = compute_metrics(
            y_true=y_test,
            y_score=lstm_scores,
        )

        lstm_metrics.update(
            {
                "experiment": "06_last_plain_lstm",
                "pool": "last",
                "classifier": "lstm",
                "fold": fold,
                "n_train": len(
                    outer_train
                ),
                "n_test": len(
                    outer_test
                ),
                "best_epoch": int(
                    best_epoch
                ),
                "hidden_size": int(
                    config.lstm_hidden_size
                ),
                "num_layers": int(
                    config.lstm_num_layers
                ),
            }
        )

        lstm_fold_rows.append(
            lstm_metrics
        )

        lstm_pred = (
            lstm_scores >= 0.0
        ).astype(int)

        for idx, true, score, pred in zip(
            outer_test,
            y_test,
            lstm_scores,
            lstm_pred,
        ):
            lstm_pred_rows.append(
                {
                    "experiment": "06_last_plain_lstm",
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
            f"[LSTM OUTER TEST] "
            f"epoch={best_epoch} | "
            f"acc={lstm_metrics['accuracy']:.4f} | "
            f"bal_acc="
            f"{lstm_metrics['balanced_accuracy']:.4f} | "
            f"f1={lstm_metrics['f1']:.4f} | "
            f"auroc={lstm_metrics['auroc']:.4f}"
        )

        if config.save_fold_models:
            torch.save(
                {
                    "model_state_dict": (
                        lstm_model.state_dict()
                    ),
                    "input_size": int(
                        X_last.shape[-1]
                    ),
                    "hidden_size": int(
                        config.lstm_hidden_size
                    ),
                    "num_layers": int(
                        config.lstm_num_layers
                    ),
                    "dropout": float(
                        config.lstm_dropout
                    ),
                    "bidirectional": bool(
                        config.lstm_bidirectional
                    ),
                    "scaler_mean": (
                        lstm_scaler.mean_
                    ),
                    "scaler_scale": (
                        lstm_scaler.scale_
                    ),
                    "best_epoch": int(
                        best_epoch
                    ),
                    "metrics": lstm_metrics,
                    "config": asdict(
                        config
                    ),
                },
                lstm_dir
                / f"fold_{fold:02d}_model.pt",
            )

        # GPU 메모리 정리
        del lstm_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Save fold metrics / OOF predictions
    # ========================================================
    rbf_fold_df = pd.DataFrame(
        rbf_fold_rows
    )

    lstm_fold_df = pd.DataFrame(
        lstm_fold_rows
    )

    rbf_pred_df = pd.DataFrame(
        rbf_pred_rows
    )

    lstm_pred_df = pd.DataFrame(
        lstm_pred_rows
    )

    rbf_fold_df.to_csv(
        rbf_dir
        / "fold_metrics.csv",
        index=False,
    )

    lstm_fold_df.to_csv(
        lstm_dir
        / "fold_metrics.csv",
        index=False,
    )

    rbf_pred_df.to_csv(
        rbf_dir
        / "oof_predictions.csv",
        index=False,
    )

    lstm_pred_df.to_csv(
        lstm_dir
        / "oof_predictions.csv",
        index=False,
    )

    comparison_df = make_final_comparison(
        rbf_fold_df=rbf_fold_df,
        rbf_pred_df=rbf_pred_df,
        lstm_fold_df=lstm_fold_df,
        lstm_pred_df=lstm_pred_df,
    )

    comparison_df.to_csv(
        output_dir
        / "comparison_rbf_vs_lstm.csv",
        index=False,
    )

    paired_df = make_paired_predictions(
        rbf_pred_df=rbf_pred_df,
        lstm_pred_df=lstm_pred_df,
    )

    paired_df.to_csv(
        output_dir
        / "paired_oof_predictions.csv",
        index=False,
    )

    return comparison_df, paired_df


# ============================================================
# Final summary
# ============================================================

def one_summary_row(
    experiment: str,
    pool: str,
    classifier: str,
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Dict:
    oof = compute_metrics(
        y_true=pred_df[
            "y_true"
        ].to_numpy(),
        y_score=pred_df[
            "decision_score"
        ].to_numpy(),
    )

    row = {
        "experiment": experiment,
        "pool": pool,
        "classifier": classifier,
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

    return row


def make_final_comparison(
    rbf_fold_df: pd.DataFrame,
    rbf_pred_df: pd.DataFrame,
    lstm_fold_df: pd.DataFrame,
    lstm_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    rbf_row = one_summary_row(
        experiment="02_last_pca_rbf_svm",
        pool="last",
        classifier="rbf",
        fold_df=rbf_fold_df,
        pred_df=rbf_pred_df,
    )

    lstm_row = one_summary_row(
        experiment="06_last_plain_lstm",
        pool="last",
        classifier="lstm",
        fold_df=lstm_fold_df,
        pred_df=lstm_pred_df,
    )

    comparison = pd.DataFrame(
        [
            rbf_row,
            lstm_row,
        ]
    )

    # LSTM - RBF delta도 별도로 추가
    delta = {
        "experiment": "delta_lstm_minus_rbf",
        "pool": "last",
        "classifier": "delta",
    }

    metric_columns = [
        col
        for col in comparison.columns
        if col.startswith("fold_")
        or col.startswith("oof_")
    ]

    for col in metric_columns:
        delta[col] = float(
            comparison.loc[
                comparison["experiment"]
                == "06_last_plain_lstm",
                col,
            ].iloc[0]
            - comparison.loc[
                comparison["experiment"]
                == "02_last_pca_rbf_svm",
                col,
            ].iloc[0]
        )

    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [delta]
            ),
        ],
        ignore_index=True,
    )

    return comparison


def make_paired_predictions(
    rbf_pred_df: pd.DataFrame,
    lstm_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    rbf = rbf_pred_df[
        [
            "sample_id",
            "group",
            "fold",
            "y_true",
            "decision_score",
            "y_pred",
        ]
    ].copy()

    rbf = rbf.rename(
        columns={
            "decision_score": "rbf_score",
            "y_pred": "rbf_pred",
        }
    )

    lstm = lstm_pred_df[
        [
            "sample_id",
            "decision_score",
            "y_pred",
        ]
    ].copy()

    lstm = lstm.rename(
        columns={
            "decision_score": "lstm_score",
            "y_pred": "lstm_pred",
        }
    )

    paired = rbf.merge(
        lstm,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    paired[
        "rbf_correct"
    ] = (
        paired["rbf_pred"]
        == paired["y_true"]
    ).astype(int)

    paired[
        "lstm_correct"
    ] = (
        paired["lstm_pred"]
        == paired["y_true"]
    ).astype(int)

    paired[
        "lstm_minus_rbf_correct"
    ] = (
        paired["lstm_correct"]
        - paired["rbf_correct"]
    )

    return paired


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
            "last_rbf_vs_lstm"
        ),

        mode="tat",
        n_layers=None,

        n_splits=5,
        inner_splits=3,
        seed=42,

        # 기존 02_last_pca_rbf_svm search 그대로
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

        class_weight="balanced",
        include_geometry=True,
        pca_iterated_power=5,
        svm_cache_size_mb=2048,

        # Plain LSTM
        lstm_hidden_size=128,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        lstm_bidirectional=False,

        lstm_lr=1e-3,
        lstm_weight_decay=1e-4,
        lstm_batch_size=16,
        lstm_max_epochs=150,
        lstm_patience=20,
        lstm_grad_clip=1.0,
        lstm_selection_metric="balanced_accuracy",

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

    with (
        output_dir
        / "config.json"
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

    labels = get_labels(
        loader
    )

    groups = get_groups(
        loader
    )

    X_last, sample_ids, n_layers = preload_last(
        loader=loader,
        config=config,
    )

    print(
        f"Samples={len(labels)} | "
        f"Robust={int((labels == 1).sum())} | "
        f"Spurious={int((labels == 0).sum())} | "
        f"groups={len(np.unique(groups))}"
    )

    # 두 방법이 완전히 동일한 outer fold 사용
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

    # split 저장
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

    comparison_df, paired_df = run_comparison(
        X_last=X_last,
        labels=labels,
        groups=groups,
        sample_ids=sample_ids,
        outer_splits=outer_splits,
        config=config,
        n_layers=n_layers,
    )

    print("\n" + "#" * 120)
    print(
        "FINAL: 02_last_pca_rbf_svm vs 06_last_plain_lstm"
    )
    print("#" * 120)

    display_cols = [
        "experiment",
        "fold_accuracy_mean",
        "fold_accuracy_std",
        "oof_accuracy",
        "fold_balanced_accuracy_mean",
        "fold_balanced_accuracy_std",
        "oof_balanced_accuracy",
        "fold_f1_mean",
        "fold_f1_std",
        "oof_f1",
        "fold_auroc_mean",
        "fold_auroc_std",
        "oof_auroc",
    ]

    print(
        comparison_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nPaired OOF summary:"
    )

    print(
        f"RBF only correct: "
        f"{int(((paired_df['rbf_correct'] == 1) & (paired_df['lstm_correct'] == 0)).sum())}"
    )

    print(
        f"LSTM only correct: "
        f"{int(((paired_df['rbf_correct'] == 0) & (paired_df['lstm_correct'] == 1)).sum())}"
    )

    print(
        f"Both correct: "
        f"{int(((paired_df['rbf_correct'] == 1) & (paired_df['lstm_correct'] == 1)).sum())}"
    )

    print(
        f"Both wrong: "
        f"{int(((paired_df['rbf_correct'] == 0) & (paired_df['lstm_correct'] == 0)).sum())}"
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "comparison_rbf_vs_lstm.csv"
    )

    print(
        output_dir
        / "paired_oof_predictions.csv"
    )


if __name__ == "__main__":
    main()