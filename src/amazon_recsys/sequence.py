"""Sequence-aware content retrieval with a GRU user encoder.

The item tower consumes frozen text vectors, so items with no positive training
events still have a representation. Only training-split interactions fit weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import Example


def history_ids(examples: list[Example], max_history: int = 20) -> np.ndarray:
    """Left-pad histories; zero is reserved for padding, item IDs are offset by one."""
    ids = np.zeros((len(examples), max_history), dtype="int64")
    for row, example in enumerate(examples):
        history = example.history[-max_history:]
        ids[row, -len(history):] = np.asarray(history, dtype="int64") + 1
    return ids


class SequenceTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.user = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.user_output = nn.Linear(hidden_dim, output_dim)
        self.item = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))

    def encode_user(self, history_vectors: torch.Tensor) -> torch.Tensor:
        _, final = self.user(history_vectors)
        return F.normalize(self.user_output(final[-1]), dim=-1)

    def encode_item(self, item_vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(item_vectors), dim=-1)


def train_sequence(vectors: np.ndarray, examples: list[Example], output_dir: Path,
                   epochs: int = 3, batch_size: int = 256, max_examples: int = 200_000,
                   seed: int = 42, learning_rate: float = 0.001) -> dict:
    if not examples:
        raise ValueError("No training examples with a prior positive history")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(examples), min(max_examples, len(examples)), replace=False)
    sample = [examples[int(index)] for index in selected]
    histories = history_ids(sample)
    targets = np.asarray([example.item_id for example in sample], dtype="int64")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = SequenceTower(vectors.shape[1]).to(device)
    # Padding is a zero vector; the frozen item table has one-based indices.
    table = torch.cat([torch.zeros((1, vectors.shape[1]), device=device),
                       torch.from_numpy(vectors).to(device)], dim=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        permutation = rng.permutation(len(sample))
        epoch_losses = []
        model.train()
        for start in range(0, len(sample), batch_size):
            indices = permutation[start:start + batch_size]
            if len(indices) < 2:
                continue
            user_batch = table[torch.from_numpy(histories[indices]).to(device)]
            item_batch = table[torch.from_numpy(targets[indices] + 1).to(device)]
            scores = model.encode_user(user_batch) @ model.encode_item(item_batch).T / 0.07
            ids = targets[indices]
            duplicates = torch.from_numpy((ids[:, None] == ids[None, :]) &
                                          ~np.eye(len(ids), dtype=bool)).to(device)
            scores = scores.masked_fill(duplicates, -1e4)
            labels = torch.arange(len(indices), device=device)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"sequence epoch={epoch + 1} loss={losses[-1]:.4f}", flush=True)
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": vectors.shape[1]},
               output_dir / "sequence_tower.pt")
    report = {"training_examples": len(sample), "epochs": epochs, "batch_size": batch_size,
              "seed": seed, "loss_by_epoch": losses, "architecture": "GRU over frozen item text vectors"}
    (output_dir / "sequence_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_sequence(path: Path) -> SequenceTower:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SequenceTower(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def transform_items(model: SequenceTower, vectors: np.ndarray,
                    batch_size: int = 4096) -> np.ndarray:
    return np.concatenate([
        model.encode_item(torch.from_numpy(vectors[start:start + batch_size])).numpy()
        for start in range(0, len(vectors), batch_size)
    ]).astype("float32")


@torch.no_grad()
def transform_users(model: SequenceTower, vectors: np.ndarray, examples: list[Example],
                    batch_size: int = 512) -> np.ndarray:
    ids = history_ids(examples)
    table = torch.cat([torch.zeros((1, vectors.shape[1])), torch.from_numpy(vectors)], dim=0)
    return np.concatenate([
        model.encode_user(table[torch.from_numpy(ids[start:start + batch_size])]).numpy()
        for start in range(0, len(ids), batch_size)
    ]).astype("float32")
