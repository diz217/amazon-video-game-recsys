import numpy as np

from amazon_recsys.protocol import Example
from amazon_recsys.sequence import SequenceTower, history_ids, transform_items, transform_users


def test_sequence_encoding_keeps_history_order_and_item_alignment():
    vectors = np.eye(3, dtype="float32")
    examples = [Example("u", 2, 3, (0, 1), "valid"),
                Example("v", 2, 3, (1, 0), "valid")]
    ids = history_ids(examples, max_history=3)
    assert ids.tolist() == [[0, 1, 2], [0, 2, 1]]
    model = SequenceTower(input_dim=3, hidden_dim=4, output_dim=2)
    assert transform_items(model, vectors).shape == (3, 2)
    assert transform_users(model, vectors, examples).shape == (2, 2)
