import math
from datetime import datetime
from collections import defaultdict

FORMAT_TYPES = ["listen", "reverse_tap", "type"]
# According to the dataset we are using (can be changed for other datasets)

def engineer_features(record: dict, prev_timestamp: float) -> list[float]:
    delta_seconds = max(record["timestamp"] - prev_timestamp, 0)
    delta_days = delta_seconds / 86400

    history_seen = record["history_seen"]
    history_correct = record["history_correct"]
    session_seen = record["session_seen"]
    session_correct = record["session_correct"]
    historical_accuracy = history_correct / (history_seen + 1e-8)
    session_accuracy = session_correct / (session_seen + 1e-8)

    dt = datetime.timezone.utc.localize(datetime.fromtimestamp(record["timestamp"]))
    hour_normalized = dt.hour / 24.0
    day_normalized = dt.weekday() / 7.0

    fmt = record.get("format", "unknown")
    is_listen = float(fmt == "listen")
    is_reverse_tap = float(fmt == "reverse_tap")
    is_type = float(fmt == "type")

    streak = record.get("streak", 0)
    days_since_first = record.get("days_since_first_seen", 0)
    total_study_days = record.get("total_study_days", 1)

    features = [
        math.log(delta_days + 1),               # 1  time since last review
        float(history_seen),                     # 2  total times seen
        float(history_correct),                  # 3  total times correct
        historical_accuracy,                     # 4  historical accuracy
        float(session_seen),                     # 5  seen this session
        float(session_correct),                  # 6  correct this session
        session_accuracy,                        # 7  session accuracy
        math.log(history_seen + 1),              # 8  log seen count
        is_listen,                               # 9  format one-hot
        is_reverse_tap,                          # 10 format one-hot
        is_type,                                 # 11 format one-hot
        hour_normalized,                         # 12 time of day
        day_normalized,                          # 13 day of week
        float(streak),                           # 14 consecutive correct
        math.log(streak + 1),                    # 15 log streak
        float(days_since_first) / max(float(total_study_days), 1),  # 16 review density
    ]

    assert len(features) == 16, f"Expected 16 features, got {len(features)}"
    return features

def build_user_history(
    records: list[dict],
    max_history_len: int = 20,
) -> list[tuple]:
    # grouping by user_id + lexeme_id 
    groups = defaultdict(list)
    for record in records:
        key = (record["user_id"], record["lexeme_id"])
        groups[key].append(record)

    dataset = []

    for key, group in groups.items():
        group = sorted(group, key=lambda r: r["timestamp"])
        for i, record in enumerate(group):
            # history = all previous reviews for this user+lexeme
            history = group[max(0, i - max_history_len): i]
            if not history:
                prev_timestamp = record["timestamp"]
            else:
                prev_timestamp = history[-1]["timestamp"]
            feature_matrix = []
            for j, hist_record in enumerate(history):
                prev_ts = group[max(0, i - max_history_len + j) - 1]["timestamp"] if j > 0 else record["timestamp"]
                feature_matrix.append(engineer_features(hist_record, prev_ts))
            while len(feature_matrix) < max_history_len:
                feature_matrix.insert(0, [0.0] * 16)
            delta_days = (record["timestamp"] - prev_timestamp) / 86400
            label = record["p_recall"]
            dataset.append((feature_matrix, delta_days, label))
    return dataset