import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parent
TEST_SEEDS = list(range(42, 52))


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with path.open("wb") as output:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                output.write(chunk)


def load_kion(folder, max_users):
    archive = folder / "raw" / "data_original.zip"
    download("https://github.com/irsafilo/KION_DATASET/raw/refs/heads/main/data_original.zip", archive)
    with zipfile.ZipFile(archive) as source:
        interactions = pd.read_csv(source.open("data_original/interactions.csv"))
        items = pd.read_csv(source.open("data_original/items.csv"))
        users = pd.read_csv(source.open("data_original/users.csv"))
    counts = interactions["user_id"].value_counts()
    selected = counts.index[np.argsort((counts.index.to_numpy(dtype="uint64") * np.uint64(11400714819323198485)) % np.uint64(18446744073709551615))[:max_users]]
    interactions = interactions[interactions.user_id.isin(selected)].copy()
    interactions["timestamp"] = pd.to_datetime(interactions["last_watch_dt"], utc=True).dt.as_unit("s").astype("int64")
    interactions["event_type"] = "watch"
    items["category"] = items["genres"].fillna("unknown").astype(str).str.split(",").str[0]
    items["content_type"] = items["content_type"].fillna("unknown")
    items["release_year"] = pd.to_numeric(items["release_year"], errors="coerce").fillna(0)
    users["user_age"] = users["age"].fillna("unknown")
    users["user_income"] = users["income"].fillna("unknown")
    users["user_sex"] = users["sex"].fillna("unknown")
    users["user_kids"] = users["kids_flg"].fillna(0)
    return interactions[["user_id", "item_id", "timestamp", "event_type"]], items[["item_id", "category", "content_type", "release_year"]], users[["user_id", "user_age", "user_income", "user_sex", "user_kids"]]


def load_yambda(folder, max_users):
    raw = folder / "raw" / "multi_event.parquet"
    mapping = folder / "raw" / "artist_item_mapping.parquet"
    download("https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/multi_event.parquet", raw)
    download("https://huggingface.co/datasets/yandex/yambda/resolve/main/artist_item_mapping.parquet", mapping)
    source = pq.ParquetFile(raw)
    counts = {}
    for batch in source.iter_batches(columns=["uid"], batch_size=1_000_000):
        values, frequencies = np.unique(batch.column(0).to_numpy(), return_counts=True)
        for user, count in zip(values, frequencies):
            counts[int(user)] = counts.get(int(user), 0) + int(count)
    users = np.array(sorted(counts), dtype="uint64")
    selected = users[np.argsort(users * np.uint64(11400714819323198485))[:max_users]]
    selected_arrow = pa.array(selected.astype("uint32"))
    frames = []
    for batch in source.iter_batches(batch_size=1_000_000):
        filtered = pa.Table.from_batches([batch]).filter(pc.is_in(batch.column(batch.schema.get_field_index("uid")), value_set=selected_arrow))
        if filtered.num_rows:
            event_index = filtered.schema.get_field_index("event_type")
            filtered = filtered.set_column(event_index, "event_type", pc.cast(filtered["event_type"], pa.string()))
            frames.append(filtered.select(["uid", "item_id", "timestamp", "event_type"]).to_pandas())
    interactions = pd.concat(frames, ignore_index=True).rename(columns={"uid": "user_id"})
    interactions["timestamp"] = interactions["timestamp"].astype("int64") * 5
    artist_source = pq.read_table(mapping)
    artist_source = artist_source.filter(pc.is_in(artist_source["item_id"], value_set=pa.array(interactions.item_id.unique().astype("uint32"))))
    artists = artist_source.to_pandas()
    artist_col = next(column for column in artists.columns if column != "item_id")
    artists = artists[["item_id", artist_col]].drop_duplicates("item_id").rename(columns={artist_col: "category"})
    artists["category"] = artists["category"].astype(str)
    artists["content_type"] = "track"
    artists["release_year"] = 0
    return interactions, artists, pd.DataFrame({"user_id": selected.astype("int64")})


def sample_candidates(queries, history, catalog, first_seen, seed, negatives):
    rng = np.random.default_rng(seed)
    histories = {user: group[["timestamp", "item_id"]].to_numpy() for user, group in history.groupby("user_id", sort=False)}
    order = np.argsort(first_seen, kind="stable")
    items = catalog[order]
    times = first_seen[order]
    rows = []
    for query in queries.itertuples():
        past = histories.get(query.user_id)
        used = set(past[past[:, 0] <= query.timestamp, 1]) if past is not None else set()
        available = items[:np.searchsorted(times, query.timestamp, side="left")]
        if len(available) <= len(used):
            continue
        sampled = []
        chosen = set(used | {query.item_id})
        attempts = 0
        while len(sampled) < negatives and attempts < negatives * 100:
            item = int(available[rng.integers(len(available))])
            attempts += 1
            if item not in chosen:
                chosen.add(item)
                sampled.append(item)
        if not sampled:
            continue
        rows.append((query.query_id, query.user_id, query.item_id, query.timestamp, 1))
        rows.extend((query.query_id, query.user_id, int(item), query.timestamp, 0) for item in sampled)
    return pd.DataFrame(rows, columns=["query_id", "user_id", "item_id", "timestamp", "label"])


def prepare(name, max_users, train_queries, negatives):
    folder = ROOT / "data" / name
    folder.mkdir(parents=True, exist_ok=True)
    if name == "KION":
        events, items, users = load_kion(folder, max_users)
    else:
        events, items, users = load_yambda(folder, max_users)
    original_metadata_items = int(items.item_id.nunique()) if name == "KION" else None
    events = events.dropna(subset=["user_id", "item_id", "timestamp"]).copy()
    original_users = pd.Index(pd.unique(events["user_id"]))
    original_items = pd.Index(pd.unique(events["item_id"]))
    user_ids = pd.Index(sorted(original_users))
    item_ids = pd.Index(sorted(original_items))
    user_lookup = dict(zip(user_ids, range(len(user_ids))))
    item_lookup = dict(zip(item_ids, range(len(item_ids))))
    events["user_id"] = events["user_id"].map(user_lookup)
    events["item_id"] = events["item_id"].map(item_lookup)
    users = users[users.user_id.isin(original_users)].copy()
    users["user_id"] = users["user_id"].map(user_lookup)
    items = items[items.item_id.isin(original_items)].copy()
    items["item_id"] = items["item_id"].map(item_lookup)
    users = pd.DataFrame({"user_id": range(len(user_ids))}).merge(users.drop_duplicates("user_id"), on="user_id", how="left")
    items = pd.DataFrame({"item_id": range(len(item_ids))}).merge(items.drop_duplicates("item_id"), on="item_id", how="left")
    items["category"] = items["category"].fillna("unknown")
    items["content_type"] = items["content_type"].fillna("track" if name == "Yambda" else "unknown")
    items["release_year"] = items["release_year"].fillna(0)
    events["rating"] = 1
    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    first = int(0.8 * len(events))
    second = int(0.9 * len(events))
    first = int(np.searchsorted(events.timestamp.to_numpy(), events.timestamp.iloc[first], side="left"))
    second = int(np.searchsorted(events.timestamp.to_numpy(), events.timestamp.iloc[second], side="left"))
    splits = {"train": events.iloc[:first].copy(), "validation": events.iloc[first:second].copy(), "test": events.iloc[second:].copy()}
    for split, frame in splits.items():
        frame.to_csv(folder / f"{split}.csv", index=False)
    items.to_csv(folder / "items.csv", index=False)
    users.to_csv(folder / "users.csv", index=False)
    train = splits["train"]
    catalog = train.item_id.drop_duplicates().to_numpy()
    first_seen = train.groupby("item_id").timestamp.min().reindex(catalog).to_numpy()
    selected_train = train.iloc[np.linspace(0, len(train) - 1, min(train_queries, len(train)), dtype=int)].copy()
    selected_train["query_id"] = np.arange(len(selected_train))
    train_pairs = sample_candidates(selected_train, train, catalog, first_seen, 42, negatives)
    train_pairs.to_parquet(folder / "train_pairs.parquet", index=False)
    eligible_users = set(train.user_id)
    eligible_items = set(catalog)
    for split in ["validation", "test"]:
        frame = splits[split]
        possible = frame[frame.user_id.isin(eligible_users) & frame.item_id.isin(eligible_items)].copy()
        if split == "validation":
            holdout = possible.groupby("user_id", sort=False).tail(1).sort_values("timestamp", kind="stable").reset_index(drop=True)
            holdout["query_id"] = np.arange(len(holdout))
            holdout.to_csv(folder / "holdout_validation_0.csv", index=False)
            holdout.to_csv(folder / "holdout_validation.csv", index=False)
        else:
            for index, seed in enumerate(TEST_SEEDS):
                holdout = possible.groupby("user_id", sort=False).sample(n=1, random_state=seed).sort_values("timestamp", kind="stable").reset_index(drop=True)
                holdout["query_id"] = np.arange(len(holdout))
                holdout.to_csv(folder / f"holdout_test_{index}.csv", index=False)
    info = {"name": name, "source": "KION data_original.zip" if name == "KION" else "Yambda flat/50m/multi_event.parquet", "seed": 42, "holdout_protocols": {"validation": "GTS Last", "test": "GTS Random"}, "holdout_counts": {"validation": 1, "test": len(TEST_SEEDS)}, "test_holdout_seeds": TEST_SEEDS, "num_users": int(events.user_id.nunique()), "num_items": int(events.item_id.nunique()), "original_metadata_items": original_metadata_items, "selected_users_observed_items": int(events.item_id.nunique()), "train_catalog_items": int(len(catalog)), "split_sizes": {key: len(value) for key, value in splits.items()}, "negative_samples_train": negatives, "evaluation_catalog": "all_train_items_unseen_before_query", "evaluation_catalog_scope": "train items observed among selected users", "selected_user_count": max_users}
    (folder / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["KION", "Yambda"], required=True)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--train-queries", type=int, default=50000)
    parser.add_argument("--negatives", type=int, default=2)
    args = parser.parse_args()
    prepare(args.dataset, args.max_users or (20000 if args.dataset == "KION" else 300), args.train_queries, args.negatives)
