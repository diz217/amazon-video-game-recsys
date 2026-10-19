# Validation record

Run environment: Windows, Python 3.11.7, NVIDIA RTX 3090 Ti (24 GB). Model: `BAAI/bge-base-en-v1.5` at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; 768-dimensional frozen product embeddings. Two-Tower: 200,000 seeded train examples, 3 epochs, batch 256. The train loss moved from 4.7519 to 4.4737. Hybrid content weight `alpha=0.5` was selected by validation Recall@100 among 0.25, 0.5, and 0.75. All results below are observed, not replayed or synthetic.

Raw source SHA256:

- `Video_Games.csv.gz`: `a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c`
- `meta_Video_Games.jsonl.gz`: `ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9`

Prepared data: 814,586 events, 94,762 users, 25,612 items; all selected items have some title/description/feature text. Global timestamp split: 736,827 train, 34,510 validation, 43,249 test interactions. After requiring a prior positive history and one first held-out positive per user: 12,167 validation and 11,470 test queries. Of test targets, 6,205 had zero positive interactions in training. This is a **zero-training-positive** bucket; it does not establish zero prior reviews or true new-product release time.

## Ablations

Full test cohort (`--max-queries 0`):

| Model | Recall@100 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Popularity | 0.01988 | 0.00575 | 0.00271 |
| Item CF | 0.01944 | 0.00645 | 0.00332 |
| Frozen text, FlatIP | 0.06574 | 0.01421 | 0.00752 |
| Two-Tower, FlatIP | 0.05519 | 0.00907 | 0.00403 |
| 50/50 hybrid, FlatIP | 0.07323 | 0.01648 | 0.00846 |
| 50/50 hybrid, HNSW | 0.07271 | 0.01639 | 0.00837 |
| GRU sequence hybrid, HNSW | 0.07681 | 0.01726 | 0.00910 |
| Hybrid + LambdaRank, HNSW | 0.07271 | 0.02258 | 0.01132 |

The GRU encoder uses the last 20 positive item text vectors in order, with the same 200,000 seeded train examples, three epochs, batch 256 and contrastive objective as the mean-history tower. Its epoch losses were 4.7028, 4.4946 and 4.4011. On all 12,167 validation queries, frozen text, mean-history hybrid and GRU hybrid achieved Recall@100 of 0.06435, 0.07685 and 0.08375 respectively, using FlatIP and content weight 0.5. GRU hybrid weights 0.25 and 0.75 achieved 0.08161 and 0.07882. Thus 0.5 was selected on validation Recall@100, before evaluating the GRU model on test.

Paired HNSW hit counts for mean-history versus GRU hybrid were 937 versus 1,022 at Recall@100 on validation (273 queries gained, 188 lost; exact two-sided sign/McNemar p=0.000088) and 834 versus 885 on test (225 gained, 174 lost; p=0.0122). At Recall@10 the test counts were 187 versus 198 (66 gained, 55 lost; p=0.363). These tests treat user queries as independent and do not account for shared items or our prior inspection of the test set. The test comparison is therefore exploratory, and the top-10 gain is not statistically convincing.

The ranker was fitted on mean-history hybrid validation candidates: 403 first-half and 369 second-half queries had a positive item inside the retrieved 100. Other validation queries cannot train a candidate reranker because the target was not retrieved. On test, reranking changed Recall@10 from 0.01660 to 0.03529 for zero-training-positive targets, but from 0.02024 to 0.00184 for 1–4 and from 0.01508 to 0.00910 for 5+ targets. The overall gain hides this subgroup regression. Its five feature gains are dominated by training-set item popularity, which has a shifting target distribution: zero-training-positive targets were 21.7% of its early-validation training window but 54.1% of test queries. This is a plausible contributor to subgroup instability, not proof of the sole cause. The existing ranker has not been fitted to GRU candidates.

For scale, the median eligible catalog at test is 24,868 items (range 24,050–25,611). A uniformly random ordering would achieve expected Recall@100=0.00403 and Recall@10=0.000403 for this single-target protocol. This is a reference calculation, not a trained recommendation baseline.

## Index quality

On 1,000 seeded validation queries, FAISS FlatIP built in 0.035 s and searched in 0.075 ms/query; HNSW built in 0.682 s and searched in 0.042 ms/query. HNSW returned 97.788% of FlatIP's top-100 items on average. These are in-process measurements with 24 FAISS threads; they exclude text encoding, ranker inference, HTTP overhead, and cold startup. End-to-end API latency should be benchmarked separately before a service-level claim.

## Checks and limitations

The local smoke test returned HTTP 200 from `/health` and `/recommend`, returned three recommendations, and excluded the item supplied as history. Unit tests verify chronological histories, candidate filtering, and train-only collaborative construction.

Reviews and ratings provide a proxy for interactions but no impression logs, purchases, or counterfactual negatives. Product metadata is a 2023 snapshot, so historical edits cannot be reconstructed. A 5-core cohort also underrepresents truly sparse users and items. The test set was used for frozen-model reporting; no post-test model selection is claimed.
