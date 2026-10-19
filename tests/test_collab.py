import pandas as pd

from amazon_recsys.collab import recommend_itemcf, train_itemcf
from amazon_recsys.protocol import Example


def test_itemcf_only_uses_train_and_prior_catalog_items(tmp_path):
    frame = pd.DataFrame([
        ("a", 0, 1, "train"), ("a", 1, 1, "train"),
        ("b", 0, 1, "train"), ("b", 1, 1, "train"),
        ("c", 0, 1, "test"), ("c", 2, 1, "test"),
    ], columns=["user_id", "item_id", "positive", "split"])
    train_itemcf(frame, 3, tmp_path, neighbors=2)
    result = recommend_itemcf(tmp_path / "itemcf.npz", [Example("x", 1, 5, (0,), "test")],
                              first_seen=pd.Series([0, 0, 10]).to_numpy(), k=2)
    assert result[0].item_ids.tolist() == [1]
