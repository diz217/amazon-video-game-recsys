"""Command line workflow for reproducible recommendation experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download, prepare
from .protocol import make_examples
from .retrieval import index_vectors, metrics, metrics_by_train_popularity, popular_candidates, retrieve
from .text import DEFAULT_REVISION, encode_items
from .towers import history_mean, load_towers, train_towers, transform_items, transform_users


def _load(workspace: Path):
    processed = workspace / "data" / "processed"
    items = pd.read_parquet(processed / "items.parquet").sort_values("item_id")
    events = pd.read_parquet(processed / "events.parquet")
    examples = make_examples(events)
    vectors = np.load(workspace / "artifacts" / "item_text.npy")
    if len(vectors) != len(items):
        raise ValueError("Item metadata and text embeddings have different lengths")
    return items, examples, vectors


def _sample(examples, limit: int, seed: int = 42):
    if limit <= 0 or len(examples) <= limit:
        return examples
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(examples), limit, replace=False))
    return [examples[i] for i in indices]


def _candidates(workspace: Path, mode: str, method: str, items: pd.DataFrame,
                examples, text_vectors: np.ndarray, k: int, alpha: float = 0.5):
    start = time.perf_counter()
    if mode == "popular":
        candidates = popular_candidates(examples, items, k=k)
    elif mode == "itemcf":
        from .collab import recommend_itemcf
        candidates = recommend_itemcf(workspace / "artifacts" / "itemcf.npz", examples,
                                      items.first_seen.to_numpy(dtype="int64"), k=k)
    else:
        if mode == "content":
            item_vectors = text_vectors
            queries = history_mean(text_vectors, examples)
            queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        elif mode in {"tower", "hybrid", "sequence", "sequence_hybrid"}:
            if mode.startswith("sequence"):
                from .sequence import load_sequence, transform_items as sequence_items
                from .sequence import transform_users as sequence_users
                model = load_sequence(workspace / "artifacts" / "sequence_tower.pt")
                item_vectors = sequence_items(model, text_vectors)
                queries = sequence_users(model, text_vectors, examples)
            else:
                model = load_towers(workspace / "artifacts" / "two_tower.pt")
                item_vectors = transform_items(model, text_vectors)
                queries = transform_users(model, text_vectors, examples)
            if mode in {"hybrid", "sequence_hybrid"}:
                if not 0 <= alpha <= 1:
                    raise ValueError("alpha must be between zero and one")
                content_queries = history_mean(text_vectors, examples)
                content_queries /= np.maximum(np.linalg.norm(content_queries, axis=1, keepdims=True), 1e-12)
                item_vectors = np.concatenate([np.sqrt(alpha) * text_vectors,
                                               np.sqrt(1 - alpha) * item_vectors], axis=1)
                queries = np.concatenate([np.sqrt(alpha) * content_queries,
                                          np.sqrt(1 - alpha) * queries], axis=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        index = index_vectors(item_vectors, method=method)
        first_seen = items.first_seen.to_numpy(dtype="int64")
        candidates = retrieve(index, queries, examples, first_seen, k=k)
    return candidates, (time.perf_counter() - start) / max(len(examples), 1) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("prepare")
    enc = sub.add_parser("encode")
    enc.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    enc.add_argument("--revision", default=DEFAULT_REVISION)
    enc.add_argument("--batch-size", type=int, default=64)
    train = sub.add_parser("train-towers")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--max-examples", type=int, default=200_000)
    sequence = sub.add_parser("train-sequence")
    sequence.add_argument("--epochs", type=int, default=3)
    sequence.add_argument("--max-examples", type=int, default=200_000)
    sub.add_parser("train-itemcf")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--split", choices=["valid", "test"], default="valid")
    evaluate.add_argument("--mode", choices=["popular", "itemcf", "content", "tower", "hybrid",
                                             "sequence", "sequence_hybrid"], default="hybrid")
    evaluate.add_argument("--index", choices=["flat", "hnsw"], default="hnsw")
    evaluate.add_argument("--max-queries", type=int, default=5000)
    evaluate.add_argument("--ranker", action="store_true")
    evaluate.add_argument("--alpha", type=float, default=0.5, help="Content weight in hybrid retrieval")
    rank = sub.add_parser("train-ranker")
    rank.add_argument("--max-queries", type=int, default=10000)
    rank.add_argument("--alpha", type=float, default=0.5)
    bench = sub.add_parser("benchmark-index")
    bench.add_argument("--max-queries", type=int, default=1000)
    bench.add_argument("--alpha", type=float, default=0.5)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.command == "download":
        result = download(workspace / "data" / "raw")
    elif args.command == "prepare":
        result = prepare(workspace / "data" / "raw", workspace / "data" / "processed")
    elif args.command == "encode":
        result = {"shape": encode_items(workspace / "data" / "processed" / "items.parquet",
                                        workspace / "artifacts", args.model, args.batch_size,
                                        revision=args.revision)}
    elif args.command == "train-towers":
        _, examples, vectors = _load(workspace)
        result = train_towers(vectors, examples["train"], workspace / "artifacts",
                              epochs=args.epochs, max_examples=args.max_examples)
    elif args.command == "train-sequence":
        from .sequence import train_sequence
        _, examples, vectors = _load(workspace)
        result = train_sequence(vectors, examples["train"], workspace / "artifacts",
                                epochs=args.epochs, max_examples=args.max_examples)
    elif args.command == "train-itemcf":
        from .collab import train_itemcf
        events = pd.read_parquet(workspace / "data" / "processed" / "events.parquet")
        items = pd.read_parquet(workspace / "data" / "processed" / "items.parquet")
        result = train_itemcf(events, len(items), workspace / "artifacts")
    elif args.command == "serve":
        import uvicorn
        from .service import create_app
        uvicorn.run(create_app(workspace, alpha=args.alpha), host=args.host, port=args.port)
        return
    elif args.command in {"evaluate", "train-ranker", "benchmark-index"}:
        from .ranking import rerank, train_ranker
        items, examples, vectors = _load(workspace)
        if args.command == "benchmark-index":
            import faiss
            selected = _sample(examples["valid"], args.max_queries)
            model = load_towers(workspace / "artifacts" / "two_tower.pt")
            item_tower = transform_items(model, vectors)
            user_tower = transform_users(model, vectors, selected)
            content_queries = history_mean(vectors, selected)
            content_queries /= np.maximum(np.linalg.norm(content_queries, axis=1, keepdims=True), 1e-12)
            item_vectors = np.concatenate([np.sqrt(args.alpha) * vectors,
                                           np.sqrt(1 - args.alpha) * item_tower], axis=1)
            query_vectors = np.concatenate([np.sqrt(args.alpha) * content_queries,
                                            np.sqrt(1 - args.alpha) * user_tower], axis=1)
            measurements = {}
            retrieved = {}
            for method in ("flat", "hnsw"):
                start = time.perf_counter()
                index = index_vectors(item_vectors, method)
                build_s = time.perf_counter() - start
                start = time.perf_counter()
                _, ids = index.search(np.ascontiguousarray(query_vectors), 100)
                query_ms = (time.perf_counter() - start) * 1000 / len(selected)
                retrieved[method] = ids
                measurements[method] = {"build_seconds": build_s, "query_ms_per_query": query_ms}
            overlap = np.mean([len(set(exact) & set(approx)) / 100
                               for exact, approx in zip(retrieved["flat"], retrieved["hnsw"])])
            result = {"queries": len(selected), "index_items": len(items), "alpha": args.alpha,
                      "faiss_threads": faiss.omp_get_max_threads(), "measurements": measurements,
                      "hnsw_overlap_with_flat_at_100": float(overlap)}
            (workspace / "artifacts" / "index_benchmark.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
        elif args.command == "train-ranker":
            selected = _sample(examples["valid"], args.max_queries)
            candidates, ms = _candidates(workspace, "hybrid", "hnsw", items, selected, vectors, 100,
                                         alpha=args.alpha)
            result = train_ranker(selected, candidates, vectors, items, workspace / "artifacts")
            result["retrieval_ms_per_query_including_index_build"] = ms
        else:
            selected = _sample(examples[args.split], args.max_queries)
            candidates, ms = _candidates(workspace, args.mode, args.index, items, selected, vectors, 100,
                                         alpha=args.alpha)
            retrieval_report = metrics(selected, candidates)
            result = {"split": args.split, "mode": args.mode, "index": args.index,
                      "alpha": args.alpha if args.mode in {"hybrid", "sequence_hybrid"} else None,
                      "max_queries": args.max_queries, "retrieval": retrieval_report,
                      "target_train_popularity": metrics_by_train_popularity(
                          selected, candidates, items.train_positive_count.to_numpy()),
                      "retrieval_ms_per_query_including_index_build": ms}
            if args.ranker:
                if args.mode != "hybrid" or args.split != "test":
                    raise ValueError("Final reranking is only supported for hybrid retrieval on test")
                reranked = rerank(workspace / "artifacts" / "ranker.txt", selected,
                                  candidates, vectors, items)
                result["reranked"] = metrics(selected, reranked)
                result["reranked_target_train_popularity"] = metrics_by_train_popularity(
                    selected, reranked, items.train_positive_count.to_numpy())
            suffix = f"_{args.alpha}" if args.mode in {"hybrid", "sequence_hybrid"} else ""
            output = workspace / "artifacts" / f"eval_{args.split}_{args.mode}_{args.index}{suffix}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
