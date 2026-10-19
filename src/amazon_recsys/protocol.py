"""Causal user histories and positive next-item examples."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Example:
    user_id: str
    item_id: int
    timestamp: int
    history: tuple[int, ...]
    split: str


def make_examples(events: pd.DataFrame, max_history: int = 20) -> dict[str, list[Example]]:
    """Only previous positive actions enter the history; first held-out positive per user."""
    if max_history < 1:
        raise ValueError("max_history must be positive")
    frame = events.sort_values(["timestamp", "user_id", "item_id"])
    histories: dict[str, list[int]] = defaultdict(list)
    heldout_users: dict[str, set[str]] = {"valid": set(), "test": set()}
    examples: dict[str, list[Example]] = {"train": [], "valid": [], "test": []}
    for row in frame.itertuples(index=False):
        user = str(row.user_id)
        split = str(row.split)
        item = int(row.item_id)
        history = histories[user]
        if int(row.positive) == 1:
            if history and (split == "train" or user not in heldout_users[split]):
                examples[split].append(
                    Example(user, item, int(row.timestamp), tuple(history[-max_history:]), split)
                )
                if split != "train":
                    heldout_users[split].add(user)
            if item not in history:
                history.append(item)
    return examples
