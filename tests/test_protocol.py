import pandas as pd

from amazon_recsys.protocol import make_examples


def test_history_excludes_target_and_future_events():
    rows = [
        ("u", 0, 5, 10, 1, "train"),
        ("u", 1, 5, 20, 1, "train"),
        ("u", 2, 2, 30, 0, "valid"),
        ("u", 3, 5, 40, 1, "valid"),
        ("u", 4, 5, 50, 1, "test"),
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp", "positive", "split"])
    examples = make_examples(frame)
    assert examples["train"][0].history == (0,)
    assert examples["valid"][0].history == (0, 1)
    assert examples["test"][0].history == (0, 1, 3)
