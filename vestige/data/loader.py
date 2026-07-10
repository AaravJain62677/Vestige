import json
import random
from pathlib import Path

def load_slam_data(raw_dir):
    raw_path = Path(raw_dir)
    records = []
    for split in ["train", "test"]:
        filepath = raw_path / f"{split}.jsonl"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                record = {
                    "user_id":          obj.get("user_id"),
                    "lexeme_id":        obj.get("lexeme_id"),
                    "timestamp":        obj.get("timestamp", 0),
                    "history_seen":     obj.get("history_seen", 0),
                    "history_correct":  obj.get("history_correct", 0),
                    "session_seen":     obj.get("session_seen", 0),
                    "session_correct":  obj.get("session_correct", 0),
                    "format":           obj.get("format", "unknown"),
                    "token":            obj.get("token", ""),
                    "p_recall":         float(obj.get("p_recall", 0.0)),
                    "split":            split,
                }
                records.append(record)

    print(f"Loaded Records: len(records)")
    return records

def split_by_user(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    user_ids = list(set(r["user_id"] for r in records))
    random.seed(seed)
    random.shuffle(user_ids)

    n = len(user_ids)
    train_end = int(n *train_ratio)
    val_end = int(n *(train_ratio + val_ratio))

    train_users = set(user_ids[:train_end])
    val_users = set(user_ids[train_end:val_end])
    test_users = set(user_ids[val_end:])

    train_records = [r for r in records if r["user_id"] in train_users]
    val_records = [r for r in records if r["user_id"] in val_users]
    test_records = [r for r in records if r["user_id"] in test_users]

    print("Train:len(train_records) | Val: len(val_records) | Test: len(test_records)")
    return train_records, val_records, test_records

def get_unique_users(records: list[dict]) -> list[str]:
    return list(set(r["user_id"] for r in records))


def get_unique_lexemes(records: list[dict]) -> list[str]:
    return list(set(r["lexeme_id"] for r in records))