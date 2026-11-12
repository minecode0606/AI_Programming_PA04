from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import pandas as pd
import torch


IndexLike = Union[int, str]


class ActivationLoader:
    """
    ActivationExtractor가 저장한 TaT activation artifact를 읽는 loader.

    기대 디렉터리 구조
    ------------------
    output_dir/
      activations/
        sample_000000.npy
        ...
      input_ids/
        sample_000000.npy
        ...
      manifest.csv
      extraction_config.json

    저장 activation
    ---------------
    H.shape == [L + 1, T, D]

    H[0]     : 첫 Transformer block 입력 residual state
    H[l + 1] : l번째 Transformer block 출력 residual state

    TaT displacement
    ----------------
    Delta = H[1:] - H[:-1]
    Delta.shape == [L, T, D]

    원 논문의 token-major TaT sequence는
        [d_(token1, layer0), ..., d_(token1, layerL-1),
         d_(token2, layer0), ..., d_(token2, layerL-1), ...]
    순서이므로,

        Delta.transpose(1, 0, 2).reshape(T * L, D)

    로 구성합니다.
    """

    def __init__(
        self,
        output_dir: str | Path,
        mmap_mode: Optional[str] = "r",
        verify_files: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.csv"
        self.config_path = self.output_dir / "extraction_config.json"

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.csv not found: {self.manifest_path}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"extraction_config.json not found: {self.config_path}"
            )

        self.manifest = pd.read_csv(self.manifest_path)

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = json.load(f)

        self.mmap_mode = mmap_mode

        required_columns = {
            "sample_id",
            "label",
            "n_tokens",
            "n_layer_states",
            "hidden_size",
            "activation_path",
            "input_ids_path",
        }
        missing = required_columns - set(self.manifest.columns)
        if missing:
            raise ValueError(
                f"manifest.csv is missing required columns: {sorted(missing)}"
            )

        # sample_id -> row index
        self._sample_to_index = {
            str(sample_id): idx
            for idx, sample_id in enumerate(self.manifest["sample_id"])
        }

        if verify_files:
            self._verify_files()

    def __len__(self) -> int:
        return len(self.manifest)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_samples={len(self)}, "
            f"num_layers={self.num_layers}, "
            f"hidden_size={self.hidden_size}, "
            f"output_dir='{self.output_dir}')"
        )

    @property
    def hidden_size(self) -> int:
        return int(self.config["hidden_size"])

    @property
    def num_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    @staticmethod
    def _normalize_label(label: Any) -> Any:
        """
        pandas가 bool/string/int를 섞어 읽을 수 있으므로
        흔한 표현은 bool로 정규화합니다.
        """
        if isinstance(label, (bool, np.bool_)):
            return bool(label)

        if isinstance(label, str):
            value = label.strip().lower()
            if value in {"true", "1", "yes", "robust"}:
                return True
            if value in {"false", "0", "no", "spurious"}:
                return False

        if isinstance(label, (int, np.integer)):
            if label == 1:
                return True
            if label == 0:
                return False

        return label

    def _resolve_index(self, sample: IndexLike) -> int:
        """
        int  -> manifest row index
        str  -> sample_id, e.g. 'sample_000012'
        """
        if isinstance(sample, (int, np.integer)):
            idx = int(sample)
            if idx < 0:
                idx += len(self)

            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"sample index out of range: {sample}"
                )
            return idx

        if isinstance(sample, str):
            if sample not in self._sample_to_index:
                raise KeyError(f"Unknown sample_id: {sample}")
            return self._sample_to_index[sample]

        raise TypeError(
            "sample must be an integer index or sample_id string."
        )

    def _row(self, sample: IndexLike) -> pd.Series:
        return self.manifest.iloc[self._resolve_index(sample)]

    def _full_path(self, relative_path: str) -> Path:
        path = self.output_dir / relative_path
        return path

    def _verify_files(self) -> None:
        missing = []

        for _, row in self.manifest.iterrows():
            act_path = self._full_path(str(row["activation_path"]))
            ids_path = self._full_path(str(row["input_ids_path"]))

            if not act_path.exists():
                missing.append(str(act_path))

            if not ids_path.exists():
                missing.append(str(ids_path))

        if missing:
            preview = "\n".join(missing[:10])
            extra = (
                f"\n... and {len(missing) - 10} more"
                if len(missing) > 10
                else ""
            )
            raise FileNotFoundError(
                "Some artifact files are missing:\n"
                f"{preview}{extra}"
            )

    def get_metadata(self, sample: IndexLike) -> Dict[str, Any]:
        row = self._row(sample)

        meta = row.to_dict()
        meta["label"] = self._normalize_label(meta.get("label"))

        return meta

    def load_activation(
        self,
        sample: IndexLike,
        *,
        mmap_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Raw residual states를 로드합니다.

        Returns
        -------
        H : np.ndarray
            shape [L + 1, T, D]
        """
        row = self._row(sample)
        path = self._full_path(str(row["activation_path"]))

        mode = self.mmap_mode if mmap_mode is None else mmap_mode
        H = np.load(path, mmap_mode=mode)

        expected = (
            int(row["n_layer_states"]),
            int(row["n_tokens"]),
            int(row["hidden_size"]),
        )

        if H.shape != expected:
            raise RuntimeError(
                f"Unexpected activation shape for {row['sample_id']}: "
                f"{H.shape}, expected {expected}"
            )

        return H

    def load_input_ids(
        self,
        sample: IndexLike,
        *,
        mmap_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        해당 sample의 token id sequence를 로드합니다.

        Returns
        -------
        input_ids : np.ndarray
            shape [T]
        """
        row = self._row(sample)
        path = self._full_path(str(row["input_ids_path"]))

        mode = self.mmap_mode if mmap_mode is None else mmap_mode
        input_ids = np.load(path, mmap_mode=mode)

        expected_tokens = int(row["n_tokens"])
        if input_ids.shape != (expected_tokens,):
            raise RuntimeError(
                f"Unexpected input_ids shape for {row['sample_id']}: "
                f"{input_ids.shape}, expected {(expected_tokens,)}"
            )

        return input_ids

    def load_displacement(
        self,
        sample: IndexLike,
        *,
        dtype: Optional[np.dtype] = np.float32,
    ) -> np.ndarray:
        """
        layer-wise displacement를 계산합니다.

        D[l, t] = H[l + 1, t] - H[l, t]

        Returns
        -------
        D : np.ndarray
            shape [L, T, D]
        """
        H = self.load_activation(sample)

        # mmap array를 그대로 빼면 결과는 새 ndarray로 materialize됩니다.
        D = H[1:] - H[:-1]

        if dtype is not None and D.dtype != dtype:
            D = D.astype(dtype, copy=False)

        return D

    def load_last_token_activation(
        self,
        sample: IndexLike,
        *,
        include_embedding_state: bool = True,
    ) -> np.ndarray:
        """
        모든 layer state에서 마지막 token activation만 가져옵니다.

        Returns
        -------
        [L + 1, D]  if include_embedding_state=True
        [L, D]      otherwise
        """
        H = self.load_activation(sample)

        if include_embedding_state:
            return np.asarray(H[:, -1, :])

        return np.asarray(H[1:, -1, :])

    def load_last_token_displacement(
        self,
        sample: IndexLike,
        *,
        dtype: Optional[np.dtype] = np.float32,
    ) -> np.ndarray:
        """
        마지막 token의 layer-wise displacement.

        Returns
        -------
        [L, D]
        """
        D = self.load_displacement(sample, dtype=dtype)
        return D[:, -1, :]

    def load_tat_sequence(
        self,
        sample: IndexLike,
        *,
        dtype: Optional[np.dtype] = np.float32,
        as_torch: bool = False,
    ):
        """
        원 논문 TaT 순서로 displacement trajectory를 펼칩니다.

        Stored:
            D.shape = [L, T, D]

        TaT:
            [token, layer, hidden]
            -> [T * L, hidden]

        Returns
        -------
        sequence : np.ndarray or torch.Tensor
            shape [T * L, hidden_size]
        """
        D = self.load_displacement(sample, dtype=dtype)

        # [L, T, D] -> [T, L, D] -> [T*L, D]
        sequence = (
            D.transpose(1, 0, 2)
            .reshape(-1, D.shape[-1])
        )

        if as_torch:
            return torch.from_numpy(
                np.ascontiguousarray(sequence)
            )

        return sequence

    def load_raw_sequence(
        self,
        sample: IndexLike,
        *,
        drop_initial_state: bool = True,
        dtype: Optional[np.dtype] = np.float32,
        as_torch: bool = False,
    ):
        """
        Raw trajectory ablation용 sequence.

        H.shape = [L + 1, T, D]

        drop_initial_state=True이면 H[1:]만 사용하여
        displacement와 layer-step 개수를 맞춥니다.

        Returns
        -------
        sequence : [T * L, D]      if drop_initial_state=True
                   [T * (L+1), D]  otherwise
        """
        H = self.load_activation(sample)

        states = H[1:] if drop_initial_state else H

        if dtype is not None and states.dtype != dtype:
            states = states.astype(dtype, copy=False)

        sequence = (
            states.transpose(1, 0, 2)
            .reshape(-1, states.shape[-1])
        )

        if as_torch:
            return torch.from_numpy(
                np.ascontiguousarray(sequence)
            )

        return sequence

    def load_sample(
        self,
        sample: IndexLike,
        *,
        mode: str = "tat",
        as_torch: bool = False,
    ) -> Dict[str, Any]:
        """
        한 sample을 학습하기 편한 dict 형태로 반환합니다.

        mode
        ----
        'tat'
            displacement full trajectory [T*L, D]

        'raw'
            raw full trajectory [T*L, D]

        'last_delta'
            last-token displacement [L, D]

        'last_raw'
            last-token raw activation [L+1, D]
        """
        meta = self.get_metadata(sample)

        if mode == "tat":
            x = self.load_tat_sequence(
                sample,
                as_torch=as_torch,
            )
        elif mode == "raw":
            x = self.load_raw_sequence(
                sample,
                as_torch=as_torch,
            )
        elif mode == "last_delta":
            x = self.load_last_token_displacement(sample)
            if as_torch:
                x = torch.from_numpy(
                    np.ascontiguousarray(x)
                )
        elif mode == "last_raw":
            x = self.load_last_token_activation(sample)
            if as_torch:
                x = torch.from_numpy(
                    np.ascontiguousarray(x)
                )
        else:
            raise ValueError(
                "mode must be one of: "
                "'tat', 'raw', 'last_delta', 'last_raw'"
            )

        label = meta["label"]

        if as_torch and isinstance(label, bool):
            label = torch.tensor(
                float(label),
                dtype=torch.float32,
            )

        return {
            "sample_id": meta["sample_id"],
            "x": x,
            "label": label,
            "length": int(x.shape[0]),
            "n_tokens": int(meta["n_tokens"]),
            "problem_hash": meta.get("problem_hash"),
            "problem": meta.get("problem"),
            "input_ids": self.load_input_ids(sample),
        }

    def iter_samples(
        self,
        *,
        mode: str = "tat",
        as_torch: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield self.load_sample(
                i,
                mode=mode,
                as_torch=as_torch,
            )


if __name__ == "__main__":
    loader = ActivationLoader(
        "artifacts/deepseek_r1_qwen3_8b_tat"
    )

    print(loader)
    print(loader.manifest.head())

    sample = loader.load_sample(
        0,
        mode="tat",
        as_torch=False,
    )

    print("sample_id :", sample["sample_id"])
    print("label     :", sample["label"])
    print("n_tokens  :", sample["n_tokens"])
    print("x.shape   :", sample["x"].shape)

    # Raw activation
    H = loader.load_activation(0)
    print("H.shape   :", H.shape)

    # Displacement
    D = loader.load_displacement(0)
    print("D.shape   :", D.shape)

    # Full TaT sequence
    S = loader.load_tat_sequence(0)
    print("S.shape   :", S.shape)
    import numpy as np
    from tqdm import tqdm

    loader = ActivationLoader(
        "artifacts/deepseek_r1_qwen3_8b_tat"
    )

    for i in tqdm(range(len(loader)), desc="Validating"):
        meta = loader.get_metadata(i)

        H = loader.load_activation(i)
        D = loader.load_displacement(i)
        S = loader.load_tat_sequence(i)

        T = int(meta["n_tokens"])
        L = loader.num_layers
        hidden_size = loader.hidden_size

        # Raw activation
        assert H.shape == (
            L + 1,
            T,
            hidden_size,
        ), (
            i,
            H.shape,
        )

        # Displacement
        assert D.shape == (
            L,
            T,
            hidden_size,
        ), (
            i,
            D.shape,
        )

        # TaT sequence
        assert S.shape == (
            T * L,
            hidden_size,
        ), (
            i,
            S.shape,
        )

        # NaN / Inf 검사
        assert np.isfinite(H).all(), (
            f"NaN/Inf in H: sample {i}"
        )

        assert np.isfinite(D).all(), (
            f"NaN/Inf in D: sample {i}"
        )

    print(
        f"All {len(loader)} samples are valid!"
    )