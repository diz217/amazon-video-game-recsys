import numpy as np

from amazon_recsys.protocol import Example
from amazon_recsys.retrieval import index_vectors, metrics, retrieve


def test_future_and_seen_items_are_filtered():
    vectors = np.eye(4, dtype="float32")
    index = index_vectors(vectors, "flat")
    example = Example("u", 1, 20, (0,), "test")
    candidates = retrieve(index, np.array([[1, 1, 0, 1]], dtype="float32"), [example],
                          np.array([0, 10, 0, 30]), k=3, oversample=4)
    assert candidates[0].item_ids.tolist() == [1, 2]
    assert metrics([example], candidates)["recall@10"] == 1.0
