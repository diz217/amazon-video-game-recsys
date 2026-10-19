"""Candidate generation, retrieval metrics, and serving-time eligibility."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd

from .protocol import Example


@dataclass
class CandidateList:
    item_ids: np.ndarray
    scores: np.ndarray


def index_vectors(vectors: np.ndarray, method: str = "hnsw"):
    dim = vectors.shape[1]
    if method == "flat":
        index = faiss.IndexFlatIP(dim)
    elif method == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 100
        index.hnsw.efSearch = 128
    else:
        raise ValueError(f"Unknown index method: {method}")
    index.add(np.ascontiguousarray(vectors.astype("float32")))
    return index


def retrieve(index, queries: np.ndarray, examples: list[Example],
             first_seen: np.ndarray, k: int = 100, oversample: int = 1000) -> list[CandidateList]:
    if len(queries) != len(examples):
        raise ValueError("One query vector required for each example")
    raw_k = min(index.ntotal, max(k, oversample))
    scores, ids = index.search(np.ascontiguousarray(queries.astype("float32")), raw_k)
    output = []
    for row, example in enumerate(examples):
        previous = set(example.history)
        eligible = [(int(i), float(s)) for i, s in zip(ids[row], scores[row])
                    if i >= 0 and i not in previous and first_seen[i] <= example.timestamp]
        eligible = eligible[:k]
        output.append(CandidateList(np.array([i for i, _ in eligible], dtype="int32"),
                                    np.array([s for _, s in eligible], dtype="float32")))
    return output


def popular_candidates(examples: list[Example], items: pd.DataFrame, k: int = 100) -> list[CandidateList]:
    ordered = items.sort_values(["train_positive_count", "item_id"], ascending=[False, True])
    ids = ordered.item_id.to_numpy()
    first_seen = items.sort_values("item_id").first_seen.to_numpy()
    counts = items.sort_values("item_id").train_positive_count.to_numpy()
    output = []
    for example in examples:
        previous = set(example.history)
        eligible = []
        for value in ids:
            item = int(value)
            if item not in previous and first_seen[item] <= example.timestamp:
                eligible.append(item)
                if len(eligible) == k:
                    break
        output.append(CandidateList(np.array(eligible, dtype="int32"),
                                    np.log1p(counts[eligible]).astype("float32")))
    return output


def metrics(examples: list[Example], candidates: list[CandidateList],
            k_values: tuple[int, ...] = (10, 100)) -> dict[str, float]:
    if len(examples) != len(candidates):
        raise ValueError("Examples and candidate lists differ in length")
    if not examples:
        raise ValueError("No evaluable examples")
    report = {"queries": len(examples)}
    for k in k_values:
        ranks = []
        for example, result in zip(examples, candidates):
            matches = np.flatnonzero(result.item_ids[:k] == example.item_id)
            ranks.append(int(matches[0]) + 1 if len(matches) else 0)
        report[f"recall@{k}"] = sum(rank > 0 for rank in ranks) / len(ranks)
        report[f"ndcg@{k}"] = sum(1 / np.log2(rank + 1) for rank in ranks if rank) / len(ranks)
    return report


def metrics_by_train_popularity(examples: list[Example], candidates: list[CandidateList],
                                train_counts: np.ndarray) -> dict[str, dict[str, float]]:
    buckets = {"zero": [], "one_to_four": [], "five_plus": []}
    for index, example in enumerate(examples):
        count = train_counts[example.item_id]
        name = "zero" if count == 0 else "one_to_four" if count < 5 else "five_plus"
        buckets[name].append(index)
    return {name: metrics([examples[i] for i in indices], [candidates[i] for i in indices])
            for name, indices in buckets.items() if indices}
