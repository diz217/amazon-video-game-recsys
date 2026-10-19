"""Load fitted artifacts once and serve recommendations from product history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .protocol import Example
from .ranking import rerank
from .retrieval import index_vectors, retrieve
from .towers import load_towers, transform_items, transform_users


class RecommendRequest(BaseModel):
    history_asins: list[str] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=10, ge=1, le=50)
    as_of_timestamp: int | None = None


def create_app(workspace: Path, alpha: float = 0.5) -> FastAPI:
    workspace = Path(workspace).resolve()
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    items = pd.read_parquet(workspace / "data" / "processed" / "items.parquet").sort_values("item_id")
    text_vectors = np.load(workspace / "artifacts" / "item_text.npy")
    if len(items) != len(text_vectors):
        raise ValueError("Item table and text vectors differ")
    model = load_towers(workspace / "artifacts" / "two_tower.pt")
    item_tower = transform_items(model, text_vectors)
    item_vectors = np.concatenate([np.sqrt(alpha) * text_vectors,
                                   np.sqrt(1 - alpha) * item_tower], axis=1)
    index = index_vectors(item_vectors, "hnsw")
    first_seen = items.first_seen.to_numpy(dtype="int64")
    asin_to_id = dict(zip(items.parent_asin, items.item_id))
    ranker_path = workspace / "artifacts" / "ranker.txt"

    app = FastAPI(title="Amazon Text Recommendation Demo", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ready", "catalog_items": len(items), "ranker_loaded": ranker_path.exists()}

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        unknown = [asin for asin in request.history_asins if asin not in asin_to_id]
        if unknown:
            raise HTTPException(400, f"Unknown catalog ASINs: {unknown[:5]}")
        history = tuple(dict.fromkeys(int(asin_to_id[asin]) for asin in request.history_asins))
        timestamp = request.as_of_timestamp or int(first_seen.max())
        if any(first_seen[item] > timestamp for item in history):
            raise HTTPException(400, "History contains an item first seen after as_of_timestamp")
        example = Example("api", -1, timestamp, history[-20:], "live")
        user_tower = transform_users(model, text_vectors, [example])
        content_query = text_vectors[list(example.history)].mean(axis=0)
        content_query /= max(np.linalg.norm(content_query), 1e-12)
        query = np.concatenate([np.sqrt(alpha) * content_query,
                                np.sqrt(1 - alpha) * user_tower[0]])[None, :]
        candidates = retrieve(index, query.astype("float32"), [example], first_seen, k=100)[0]
        if ranker_path.exists() and len(candidates.item_ids):
            candidates = rerank(ranker_path, [example], [candidates], text_vectors, items)[0]
        recommendations = [{"parent_asin": str(items.iloc[int(item)].parent_asin),
                            "title": str(items.iloc[int(item)].title), "score": float(score)}
                           for item, score in zip(candidates.item_ids[:request.top_k],
                                                  candidates.scores[:request.top_k])]
        return {"recommendations": recommendations, "candidate_count": len(candidates.item_ids),
                "as_of_timestamp": timestamp}

    return app
