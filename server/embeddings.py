from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any

import numpy as np


class DynamicEmbedder:
    def __init__(self, dimension: int = 256, model_name: str | None = None) -> None:
        self.dimension = dimension
        self.model_name = model_name or os.getenv(
            "OMNIGUARD_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._lock = threading.Lock()
        self._model_loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._use_transformer = os.getenv("OMNIGUARD_USE_TRANSFORMER_EMBEDDER", "0") == "1"

    def _try_load_model(self) -> None:
        if not self._use_transformer or self._model_loaded:
            return
        with self._lock:
            if self._model_loaded:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                if torch.cuda.is_available() and os.getenv("OMNIGUARD_EMBED_ON_GPU", "0") == "1":
                    self._model = self._model.to("cuda")
            except Exception:
                self._tokenizer = None
                self._model = None
            finally:
                self._model_loaded = True

    def _hash_embedding(self, text: str) -> list[float]:
        cleaned = re.sub(r"\s+", " ", text.strip().lower())
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not cleaned:
            return vector.tolist()
        tokens = cleaned.split(" ")
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                idx = int.from_bytes(digest[i : i + 2], byteorder="big") % self.dimension
                vector[idx] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector.tolist()

    def _transformer_embedding(self, text: str) -> list[float] | None:
        if self._tokenizer is None or self._model is None:
            return None
        try:
            import torch

            inputs = self._tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            if next(self._model.parameters()).is_cuda:
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vec = pooled[0].detach().cpu().numpy().astype(np.float32)
            if vec.shape[0] >= self.dimension:
                vec = vec[: self.dimension]
            else:
                vec = np.pad(vec, (0, self.dimension - vec.shape[0]))
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
            return vec.tolist()
        except Exception:
            return None

    def encode(self, text: str) -> list[float]:
        self._try_load_model()
        transformer_vector = self._transformer_embedding(text)
        if transformer_vector is not None:
            return transformer_vector
        return self._hash_embedding(text)
