from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

    def load_qwen8b_low(
        self,
        cache_dir: Optional[str] = None,
        device_map: str | Dict[str, Any] = "auto",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = "sdpa",
    ):
        """
        DeepSeek-R1-0528-Qwen3-8B를 inference 전용으로 로드합니다.

        주의:
        - activation geometry를 보존하기 위해 4/8-bit quantization은 사용하지 않습니다.
        - BF16 weight로 로드하여 메모리를 줄입니다.
        - 반환값: (model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID,
            cache_dir=cache_dir,
            use_fast=True,
        )

        load_kwargs = dict(
            pretrained_model_name_or_path=self.MODEL_ID,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        model.eval()
        model.config.use_cache = False

        return model, tokenizer


class ActivationExtractor:
    """
    TaT용 residual-stream activation을 추출하고 .npy로 저장합니다.

    저장 activation의 정의
    ----------------------
    H.shape == [L + 1, T, D]

    H[0]     : 첫 Transformer block 입력 residual state
    H[l + 1] : l번째 Transformer block 출력 residual state

    따라서 TaT displacement는 나중에 그대로
        D = H[1:] - H[:-1]
    로 계산할 수 있습니다.
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str | Path,
        max_length: Optional[int] = None,
        save_dtype: str = "float32",
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt

        dtype_map = {
            "float32": (torch.float32, np.float32),
            "float16": (torch.float16, np.float16),
        }
        if save_dtype not in dtype_map:
            raise ValueError("save_dtype must be 'float32' or 'float16'.")
        self.save_dtype_name = save_dtype
        self.save_torch_dtype, self.save_np_dtype = dtype_map[save_dtype]

        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise TypeError(
                "Expected a Qwen3ForCausalLM-like model with model.model.layers."
            )

        self.layers = self.model.model.layers
        self.num_layers = len(self.layers)
        self.hidden_size = int(self.model.config.hidden_size)

        self.activation_dir = self.output_dir / "activations"
        self.input_ids_dir = self.output_dir / "input_ids"
        self.activation_dir.mkdir(parents=True, exist_ok=True)
        self.input_ids_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_dir / "manifest.csv"
        self.config_path = self.output_dir / "extraction_config.json"
        self._write_extraction_config()

    def _write_extraction_config(self) -> None:
        config = {
            "model_id": getattr(self.model.config, "_name_or_path", None),
            "architecture": self.model.__class__.__name__,
            "num_hidden_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "model_torch_dtype": str(getattr(self.model.config, "torch_dtype", None)),
            "saved_activation_dtype": self.save_dtype_name,
            "activation_layout": "[layer_state, token, hidden]",
            "activation_definition": {
                "H[0]": "input to transformer block 0",
                "H[l+1]": "output of transformer block l before final model norm",
                "displacement": "H[1:] - H[:-1]",
            },
            "use_chat_template": self.use_chat_template,
            "add_generation_prompt": self.use_chat_template,
            "system_prompt": self.system_prompt,
            "max_length": self.max_length,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _get_input_device(self) -> torch.device:
        device = self.model.get_input_embeddings().weight.device
        if device.type == "meta":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        return device

    def _tokenize(self, problem: str):
        if self.use_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": problem})

            kwargs = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if self.max_length is not None:
                kwargs.update(truncation=True, max_length=self.max_length)

            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            kwargs = dict(return_tensors="pt")
            if self.max_length is not None:
                kwargs.update(truncation=True, max_length=self.max_length)
            encoded = self.tokenizer(problem, **kwargs)

        input_device = self._get_input_device()
        return {k: v.to(input_device) for k, v in encoded.items()}

    @staticmethod
    def _extract_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("Layer returned an empty tuple/list.")
            return ActivationExtractor._extract_tensor(output[0])
        raise TypeError(f"Unsupported layer output type: {type(output)!r}")

    def extract(self, problem: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        한 문제의 residual-stream states를 추출합니다.

        Returns
        -------
        activations : np.ndarray
            shape [num_layers + 1, num_tokens, hidden_size]
        input_ids : np.ndarray
            shape [num_tokens]
        """
        encoded = self._tokenize(problem)
        input_ids = encoded["input_ids"][0].detach().cpu().numpy().astype(np.int32)

        captured: List[Optional[torch.Tensor]] = [None] * (self.num_layers + 1)
        hooks = []

        def first_layer_pre_hook(module, args, kwargs):
            if len(args) > 0:
                hidden = args[0]
            else:
                hidden = kwargs["hidden_states"]
            captured[0] = hidden.detach().to(
                device="cpu", dtype=self.save_torch_dtype
            )

        hooks.append(
            self.layers[0].register_forward_pre_hook(
                first_layer_pre_hook, with_kwargs=True
            )
        )

        for layer_idx, layer in enumerate(self.layers):
            def make_hook(idx):
                def hook(module, args, kwargs, output):
                    hidden = self._extract_tensor(output)
                    captured[idx + 1] = hidden.detach().to(
                        device="cpu", dtype=self.save_torch_dtype
                    )
                return hook

            hooks.append(
                layer.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            )

        try:
            # lm_head/logits는 필요 없으므로 base transformer만 forward합니다.
            base_inputs = {
                k: v
                for k, v in encoded.items()
                if k in {"input_ids", "attention_mask", "position_ids"}
            }
            with torch.inference_mode():
                _ = self.model.model(
                    **base_inputs,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for h in hooks:
                h.remove()

        missing = [i for i, x in enumerate(captured) if x is None]
        if missing:
            raise RuntimeError(f"Failed to capture hidden states at indices: {missing}")

        # 각 captured tensor는 [1, T, D]. batch 차원을 제거해 [L+1, T, D]로 저장합니다.
        first = captured[0]
        assert first is not None
        num_tokens = first.shape[1]
        activations = np.empty(
            (self.num_layers + 1, num_tokens, self.hidden_size),
            dtype=self.save_np_dtype,
        )

        for i, state in enumerate(captured):
            assert state is not None
            arr = state.squeeze(0).numpy()
            if arr.shape != (num_tokens, self.hidden_size):
                raise RuntimeError(
                    f"Unexpected hidden state shape at {i}: {arr.shape}; "
                    f"expected {(num_tokens, self.hidden_size)}"
                )
            activations[i] = arr

        del captured
        return activations, input_ids

    @staticmethod
    def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
        tmp_path = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp_path, array)
        os.replace(tmp_path, path)

    def save_sample(
        self,
        problem: str,
        sample_index: int,
        label: Any = None,
        source_row: Optional[int] = None,
    ) -> Dict[str, Any]:
        sample_name = f"sample_{sample_index:06d}"
        activation_path = self.activation_dir / f"{sample_name}.npy"
        input_ids_path = self.input_ids_dir / f"{sample_name}.npy"

        activations, input_ids = self.extract(problem)

        self._atomic_save_npy(activation_path, activations)
        self._atomic_save_npy(input_ids_path, input_ids)

        problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]

        return {
            "sample_id": sample_name,
            "source_row": sample_index if source_row is None else source_row,
            "label": label,
            "problem_hash": problem_hash,
            "n_tokens": int(input_ids.shape[0]),
            "n_layer_states": int(activations.shape[0]),
            "hidden_size": int(activations.shape[2]),
            "activation_path": str(activation_path.relative_to(self.output_dir)),
            "input_ids_path": str(input_ids_path.relative_to(self.output_dir)),
            "problem": problem,
        }

    def extract_csv(
        self,
        csv_path: str | Path,
        problem_col: str = "problem",
        label_col: str = "label",
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        CSV 전체를 추출합니다.

        결과 구조
        ---------
        output_dir/
          activations/sample_XXXXXX.npy    # [L+1, T, D]
          input_ids/sample_XXXXXX.npy       # [T]
          manifest.csv
          extraction_config.json
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        if problem_col not in df.columns:
            raise KeyError(
                f"Column {problem_col!r} not found. Available: {list(df.columns)}"
            )
        if label_col not in df.columns:
            raise KeyError(
                f"Column {label_col!r} not found. Available: {list(df.columns)}"
            )

        existing_rows: List[Dict[str, Any]] = []
        done_ids = set()
        if resume and self.manifest_path.exists():
            existing_df = pd.read_csv(self.manifest_path)
            existing_rows = existing_df.to_dict("records")
            done_ids = set(existing_df["sample_id"].astype(str))

        rows = list(existing_rows)

        for sample_index, (source_row, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Extracting activations")
        ):
            sample_name = f"sample_{sample_index:06d}"
            activation_path = self.activation_dir / f"{sample_name}.npy"
            input_ids_path = self.input_ids_dir / f"{sample_name}.npy"

            if (
                resume
                and sample_name in done_ids
                and activation_path.exists()
                and input_ids_path.exists()
            ):
                continue

            problem = str(row[problem_col])
            label = row[label_col]

            try:
                meta = self.save_sample(
                    problem=problem,
                    sample_index=sample_index,
                    label=label,
                    source_row=int(source_row),
                )
                rows = [r for r in rows if r.get("sample_id") != sample_name]
                rows.append(meta)

                # 중간에 작업이 끊겨도 복구할 수 있도록 매 sample마다 manifest 갱신
                manifest = pd.DataFrame(rows).sort_values("source_row")
                manifest.to_csv(self.manifest_path, index=False)

            except Exception as e:
                print(f"\n[ERROR] sample={sample_name}, source_row={source_row}: {e}")
                raise

        manifest = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
        manifest = manifest.sort_values("source_row").reset_index(drop=True)
        manifest.to_csv(self.manifest_path, index=False)
        return manifest


if __name__ == "__main__":
    CSV_PATH = "data/processed/math-robust-qwen-problems.csv"
    OUTPUT_DIR = "artifacts/deepseek_r1_qwen3_8b_tat"

    loader = ModelLoader()
    model, tokenizer = loader.load_qwen8b_low(
        cache_dir=None,          # 예: "/data/hf_cache"
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    extractor = ActivationExtractor(
        model=model,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
        max_length=None,         # 현재 데이터가 짧다면 자르지 않는 것을 권장
        save_dtype="float32",   # displacement 계산 안정성을 위해 FP32 저장
        use_chat_template=True,
        system_prompt=None,
    )

    manifest = extractor.extract_csv(
        csv_path=CSV_PATH,
        problem_col="problem",
        label_col="label",
        resume=True,
    )

    print(manifest.head())
    print(f"Saved {len(manifest)} samples to: {OUTPUT_DIR}")

