"""Official data ingestion and leakage-aware event preparation."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
import truststore

SOURCE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
FILES = {
    "Video_Games.csv.gz": "benchmark/5core/rating_only/Video_Games.csv.gz",
    "meta_Video_Games.jsonl.gz": "raw/meta_categories/meta_Video_Games.jsonl.gz",
}
EXPECTED_SHA256 = {
    "Video_Games.csv.gz": "a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c",
    "meta_Video_Games.jsonl.gz": "ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9",
}
TRAIN_END = 1628643414042
VALID_END = 1658002729837


def download(raw_dir: Path) -> dict[str, str]:
    """Download the two original gzip files and record SHA256 hashes."""
    truststore.inject_into_ssl()
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, relative in FILES.items():
        destination = raw_dir / name
        if not destination.exists():
            partial = destination.with_suffix(destination.suffix + ".partial")
            with requests.get(SOURCE + relative, stream=True, timeout=90) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            partial.replace(destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
        if hashes[name] != EXPECTED_SHA256[name]:
            raise ValueError(f"Unexpected SHA256 for {destination}: {hashes[name]}")
    (raw_dir / "sources.json").write_text(
        json.dumps({"base_url": SOURCE, "files": FILES, "sha256": hashes}, indent=2),
        encoding="utf-8",
    )
    return hashes


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(part) for part in value if part)
    return str(value) if value is not None else ""


def prepare(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    """Normalize product text and interaction rows; exclude aggregate rating fields."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(raw_dir / "Video_Games.csv.gz")
    required = {"user_id", "parent_asin", "rating", "timestamp"}
    if not required.issubset(events):
        raise ValueError(f"Missing event columns: {required - set(events)}")
    events = events.dropna(subset=list(required))
    events["timestamp"] = events["timestamp"].astype("int64")
    events = events.sort_values(["timestamp", "user_id", "parent_asin"])
    events = events.drop_duplicates(["user_id", "parent_asin"], keep="first")
    events["positive"] = (events["rating"] >= 4).astype("int8")
    events["split"] = "test"
    events.loc[events.timestamp < VALID_END, "split"] = "valid"
    events.loc[events.timestamp < TRAIN_END, "split"] = "train"

    seen = set(events.parent_asin)
    metadata = {}
    with gzip.open(raw_dir / "meta_Video_Games.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            asin = item.get("parent_asin")
            if asin not in seen:
                continue
            title = _as_text(item.get("title"))
            description = _as_text(item.get("description"))
            features = _as_text(item.get("features"))
            categories = item.get("categories") or []
            # Product-level average_rating/rating_number summarize future interactions.
            text = " ".join(part for part in [title, description, features] if part).strip()
            row = {"parent_asin": asin, "title": title, "text": text[:1200],
                   "categories": " | ".join(str(x) for x in categories)}
            if asin not in metadata or len(text) > len(metadata[asin]["text"]):
                metadata[asin] = row
    first_seen = events.groupby("parent_asin")["timestamp"].min().to_dict()
    train_counts = events[(events.split == "train") & (events.positive == 1)].groupby("parent_asin").size().to_dict()
    items = []
    for asin in sorted(seen):
        row = metadata.get(asin, {"parent_asin": asin, "title": "", "text": "", "categories": ""})
        items.append({**row, "first_seen": int(first_seen[asin]),
                      "train_positive_count": int(train_counts.get(asin, 0))})
    items_frame = pd.DataFrame(items)
    items_frame.insert(0, "item_id", range(len(items_frame)))
    item_map = dict(zip(items_frame.parent_asin, items_frame.item_id))
    events["item_id"] = events.parent_asin.map(item_map).astype("int32")
    events = events[["user_id", "item_id", "rating", "timestamp", "positive", "split"]]
    items_frame.to_parquet(output_dir / "items.parquet", index=False)
    events.to_parquet(output_dir / "events.parquet", index=False)
    summary = {"events": len(events), "users": events.user_id.nunique(),
               "items": len(items_frame), "items_with_text": int(items_frame.text.ne("").sum()),
               **{f"{s}_events": int((events.split == s).sum()) for s in ("train", "valid", "test")}}
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
