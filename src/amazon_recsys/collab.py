"""Train-only item-item collaborative filtering baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .protocol import Example
from .retrieval import CandidateList


def train_itemcf(events: pd.DataFrame, n_items: int, output_dir: Path,
                 neighbors: int = 200) -> dict:
    positive = events[(events.split == "train") & (events.positive == 1)]
    user_ids, users = pd.factorize(positive.user_id, sort=True)
    matrix = sparse.csr_matrix((np.ones(len(positive), dtype="float32"),
                                (user_ids, positive.item_id.to_numpy())),
                               shape=(len(users), n_items))
    matrix.data[:] = 1
    counts = np.asarray(matrix.power(2).sum(axis=0)).ravel()
    inverse = np.divide(1, np.sqrt(counts), out=np.zeros_like(counts), where=counts > 0)
    normalized = matrix @ sparse.diags(inverse)
    similarity = (normalized.T @ normalized).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    neighbor_ids = np.full((n_items, neighbors), -1, dtype="int32")
    neighbor_scores = np.zeros((n_items, neighbors), dtype="float32")
    for item in range(n_items):
        start, stop = similarity.indptr[item:item + 2]
        scores = similarity.data[start:stop]
        ids = similarity.indices[start:stop]
        selected = np.argsort(-scores, kind="stable")[:neighbors]
        count = len(selected)
        neighbor_ids[item, :count] = ids[selected]
        neighbor_scores[item, :count] = scores[selected]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "itemcf.npz", ids=neighbor_ids, scores=neighbor_scores)
    return {"train_positive_events": len(positive), "users": len(users),
            "items": n_items, "neighbors_per_item": neighbors}


def recommend_itemcf(model_path: Path, examples: list[Example], first_seen: np.ndarray,
                     k: int = 100) -> list[CandidateList]:
    with np.load(model_path) as model:
        ids, scores = model["ids"], model["scores"]
    output = []
    for example in examples:
        aggregate: dict[int, float] = {}
        previous = set(example.history)
        for historical_item in example.history:
            for item, score in zip(ids[historical_item], scores[historical_item]):
                item = int(item)
                if item >= 0 and item not in previous and first_seen[item] <= example.timestamp:
                    aggregate[item] = aggregate.get(item, 0.0) + float(score)
        ranked = sorted(aggregate.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
        output.append(CandidateList(np.array([item for item, _ in ranked], dtype="int32"),
                                    np.array([score for _, score in ranked], dtype="float32")))
    return output
