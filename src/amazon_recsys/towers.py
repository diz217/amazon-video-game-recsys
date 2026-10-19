"""Small trainable towers on top of frozen product-language embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import Example


def history_mean(vectors: np.ndarray, examples: list[Example]) -> np.ndarray:
    return np.stack([vectors[list(example.history)].mean(axis=0) for example in examples]).astype("float32")


class TwoTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.user = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.item = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def encode_user(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user(vectors), dim=-1)

    def encode_item(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(vectors), dim=-1)


def train_towers(vectors: np.ndarray, examples: list[Example], output_dir: Path,
                 epochs: int = 3, batch_size: int = 256, max_examples: int = 200_000,
                 seed: int = 42, learning_rate: float = 0.001) -> dict:
    if not examples:
        raise ValueError("No training examples with a prior positive history")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(examples), min(max_examples, len(examples)), replace=False)
    sample = [examples[int(index)] for index in selected]
    users = history_mean(vectors, sample)
    items = vectors[[example.item_id for example in sample]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = TwoTower(vectors.shape[1]).to(device)
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
            user_batch = torch.from_numpy(users[indices]).to(device)
            item_batch = torch.from_numpy(items[indices]).to(device)
            scores = model.encode_user(user_batch) @ model.encode_item(item_batch).T / 0.07
            labels = torch.arange(len(indices), device=device)
            # Duplicate targets in one batch are false negatives; mask them.
            ids = np.array([sample[int(i)].item_id for i in indices])
            duplicates = torch.from_numpy((ids[:, None] == ids[None, :]) &
                                          ~np.eye(len(ids), dtype=bool)).to(device)
            scores = scores.masked_fill(duplicates, -1e4)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"epoch={epoch + 1} loss={losses[-1]:.4f}", flush=True)
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": vectors.shape[1]},
               output_dir / "two_tower.pt")
    report = {"training_examples": len(sample), "epochs": epochs, "batch_size": batch_size,
              "seed": seed, "loss_by_epoch": losses}
    (output_dir / "two_tower_train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_towers(path: Path) -> TwoTower:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TwoTower(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def transform_items(model: TwoTower, vectors: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    return np.concatenate([
        model.encode_item(torch.from_numpy(vectors[start:start + batch_size])).numpy()
        for start in range(0, len(vectors), batch_size)
    ]).astype("float32")


@torch.no_grad()
def transform_users(model: TwoTower, vectors: np.ndarray, examples: list[Example],
                    batch_size: int = 4096) -> np.ndarray:
    means = history_mean(vectors, examples)
    return np.concatenate([
        model.encode_user(torch.from_numpy(means[start:start + batch_size])).numpy()
        for start in range(0, len(means), batch_size)
    ]).astype("float32")
