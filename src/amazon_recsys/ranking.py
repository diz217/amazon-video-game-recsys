"""LightGBM reranker trained on held-out validation candidate lists."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .protocol import Example
from .retrieval import CandidateList

FEATURES = ["retrieval_score", "text_cosine", "log_train_popularity", "history_length", "text_length"]


def candidate_features(examples: list[Example], candidates: list[CandidateList],
                       text_vectors: np.ndarray, items: pd.DataFrame):
    ordered = items.sort_values("item_id")
    popularity = ordered.train_positive_count.to_numpy()
    lengths = ordered.text.fillna("").str.len().to_numpy()
    x_rows, labels, groups = [], [], []
    for example, result in zip(examples, candidates):
        history_vector = text_vectors[list(example.history)].mean(axis=0)
        history_vector /= max(np.linalg.norm(history_vector), 1e-12)
        for item, score in zip(result.item_ids, result.scores):
            item = int(item)
            x_rows.append([float(score), float(history_vector @ text_vectors[item]),
                           float(np.log1p(popularity[item])), float(len(example.history)),
                           float(lengths[item])])
            labels.append(int(item == example.item_id))
        groups.append(len(result.item_ids))
    return np.asarray(x_rows, dtype="float32"), np.asarray(labels, dtype="int32"), groups


def train_ranker(examples: list[Example], candidates: list[CandidateList],
                 text_vectors: np.ndarray, items: pd.DataFrame, output_dir: Path) -> dict:
    if len(examples) < 4:
        raise ValueError("Need at least four validation examples for ranker training")
    order = np.argsort([example.timestamp for example in examples])
    split = len(order) // 2
    train_ids, valid_ids = order[:split], order[split:]
    def build(indices):
        pairs = [(examples[int(i)], candidates[int(i)]) for i in indices
                 if examples[int(i)].item_id in candidates[int(i)].item_ids]
        if not pairs:
            raise ValueError("No positives survived retrieval")
        return candidate_features([p[0] for p in pairs], [p[1] for p in pairs], text_vectors, items)
    x_train, y_train, group_train = build(train_ids)
    x_valid, y_valid, group_valid = build(valid_ids)
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=150, learning_rate=0.05,
                          num_leaves=15, min_child_samples=40, random_state=42, verbosity=-1)
    model.fit(x_train, y_train, group=group_train,
              eval_set=[(x_valid, y_valid)], eval_group=[group_valid],
              eval_at=[10], callbacks=[lgb.early_stopping(15, verbose=False)])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_dir / "ranker.txt"))
    report = {"train_queries_with_retrieved_positive": len(group_train),
              "valid_queries_with_retrieved_positive": len(group_valid),
              "train_queries_total": len(train_ids), "valid_queries_total": len(valid_ids),
              "best_iteration": model.best_iteration_, "features": FEATURES}
    (output_dir / "ranker_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def rerank(model_path: Path, examples: list[Example], candidates: list[CandidateList],
           text_vectors: np.ndarray, items: pd.DataFrame) -> list[CandidateList]:
    booster = lgb.Booster(model_file=str(model_path))
    x, _, groups = candidate_features(examples, candidates, text_vectors, items)
    predictions = booster.predict(x)
    output, offset = [], 0
    for candidate, size in zip(candidates, groups):
        ranking = np.argsort(-predictions[offset:offset + size], kind="stable")
        output.append(CandidateList(candidate.item_ids[ranking], predictions[offset:offset + size][ranking]))
        offset += size
    return output
