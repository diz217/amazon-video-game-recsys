"""Frozen language encoder for product content."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import truststore
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def encode_items(items_path: Path, output_dir: Path, model_name: str = DEFAULT_MODEL,
                 batch_size: int = 64, device: str | None = None,
                 revision: str = DEFAULT_REVISION) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(items_path).sort_values("item_id")
    if items.item_id.tolist() != list(range(len(items))):
        raise ValueError("item_id must be dense, starting at zero")
    truststore.inject_into_ssl()
    model = SentenceTransformer(model_name, device=device, revision=revision)
    texts = [value if value else "unknown video game" for value in items.text]
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    np.save(output_dir / "item_text.npy", vectors)
    (output_dir / "text_encoder.json").write_text(
        json.dumps({"model": model_name, "revision": revision,
                    "items": len(items), "dimension": vectors.shape[1]}, indent=2),
        encoding="utf-8",
    )
    return vectors.shape
