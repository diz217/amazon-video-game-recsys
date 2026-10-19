# Project decisions and experiment gates

1. **Task:** personalized next-item recommendation based on past positive reviews. This differs from query-product search; there is no natural-language query in the interaction data.
2. **First dataset:** Amazon Reviews 2023, `Video_Games`, official 5-core rating file plus raw product metadata. The 5-core filter gives a practical warm-history benchmark. A separate 0-core follow-up is required for a serious new-item cold-start claim.
3. **Positive signal:** rating >= 4. Ratings below four are not treated as exposure-verified negatives.
4. **Split:** global timestamp cutoffs from the official benchmark. User histories are built from actions strictly earlier than the target timestamp. Model training uses only train events; ranker fitting uses validation; final score uses test.
5. **Text model:** frozen BGE-base-en-v1.5 for product text. Train only two small projection towers. This tests text contribution while staying practical on one 24-GB GPU.
6. **Retrieval:** FAISS exact FlatIP is the reference. HNSW is the approximate serving index. Index quality is measured against FlatIP, and final recommendations are measured against held-out events.
7. **Ranking:** LightGBM LambdaRank uses retrieval score, frozen text similarity, training-set item popularity, user-history length, and text length. No all-time product review aggregates.
8. **Primary ablations:** popularity; item-item collaborative filtering; content-only; learned towers; learned towers plus ranker. Follow-ups: title-only text, 5 vs 20 history items, and HNSW parameters. Sparse-item metrics are grouped by item train-positive count (0, 1–4, 5+).
9. **Reporting:** record exact file hashes, model name/revision, sample count, seed, metrics, latency, and the dataset's proxy-label limits. Do not infer online conversion improvement from offline metrics.

Before claiming cold-start recommendation, add the unfiltered 0-core interactions and evaluate new products that have metadata but no pre-cutoff positive reviews. Before claiming production readiness, add a request-serving API, model/version manifests, and latency benchmarks that exclude index construction.
