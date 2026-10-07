"""Train and evaluate one weighted Random Forest baseline per client shard.

Each fleet operator trains only on its own shard.  Rows are split chronologically
within every Attack-Group, so no future observations are used for local training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# Results reported by train_baseline.py when all client shards are centralized.
CENTRALIZED_ACCURACY = 0.9244
CENTRALIZED_F1 = 0.9574

FEATURE_COLUMNS = [
    "shunt_voltage",
    "bus_voltage_V",
    "current_mA",
    "power_mW",
    "State",
]
TARGET_COLUMN = "Label"
WEIGHT_COLUMN = "class_weight"
TIME_COLUMN = "time"
GROUP_COLUMN = "Attack-Group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-dir",
        type=Path,
        default=Path("data") / "clients",
        help="Directory containing client_1.csv, client_2.csv, and client_3.csv.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser.parse_args()


def chronological_group_split(data: pd.DataFrame, test_size: float) -> tuple[pd.Index, pd.Index]:
    """Split each attack group into earliest training and latest testing rows."""
    train_indices: list[object] = []
    test_indices: list[object] = []

    for group_name, group in data.groupby(GROUP_COLUMN, sort=True, dropna=False):
        ordered_group = group.sort_values(TIME_COLUMN, kind="mergesort")
        split_at = int(len(ordered_group) * (1 - test_size))
        if split_at == 0 or split_at == len(ordered_group):
            raise ValueError(
                f"Attack-Group {group_name!r} has {len(ordered_group)} rows and cannot "
                "be split into non-empty chronological train/test sets."
            )
        train_indices.extend(ordered_group.index[:split_at])
        test_indices.extend(ordered_group.index[split_at:])

    return pd.Index(train_indices), pd.Index(test_indices)


def validate_columns(data: pd.DataFrame, client_name: str) -> None:
    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN, WEIGHT_COLUMN, TIME_COLUMN, GROUP_COLUMN]
    )
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"{client_name} is missing columns: {sorted(missing_columns)}")


def train_and_evaluate_client(
    client_name: str, data: pd.DataFrame, test_size: float, random_state: int, n_estimators: int
) -> dict[str, object]:
    """Train and evaluate a model using only one client's local data."""
    validate_columns(data, client_name)
    data = data.copy()
    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="raise")
    train_indices, test_indices = chronological_group_split(data, test_size)

    # One-hot encoding makes the categorical State values usable by the forest.
    features = pd.get_dummies(data[FEATURE_COLUMNS], columns=["State"], dtype=int)
    target = data[TARGET_COLUMN].astype(str)
    sample_weights = pd.to_numeric(data[WEIGHT_COLUMN], errors="raise")

    X_train = features.loc[train_indices]
    X_test = features.loc[test_indices]
    y_train = target.loc[train_indices]
    y_test = target.loc[test_indices]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights.loc[train_indices])
    predictions = model.predict(X_test)

    labels = sorted(target.unique())
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="attack", zero_division=0)
    recall = recall_score(y_test, predictions, pos_label="attack", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="attack", zero_division=0)
    attack_groups = ", ".join(str(group) for group in sorted(data[GROUP_COLUMN].dropna().unique()))

    print(f"\n{'=' * 72}\n{client_name} local model")
    print(f"Attack-Group composition: {attack_groups or '(missing)'}")
    print(f"Loaded {len(data)} local rows.")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; label order: " + ", ".join(labels) + "):")
    print(confusion_matrix(y_test, predictions, labels=labels))

    return {
        "client": client_name,
        "attack_groups": attack_groups or "(missing)",
        "accuracy": accuracy,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    client_files = [args.client_dir / f"client_{number}.csv" for number in range(1, 4)]
    missing_files = [str(path) for path in client_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing client shard(s): {', '.join(missing_files)}")

    results = []
    for path in client_files:
        results.append(
            train_and_evaluate_client(
                path.stem,
                pd.read_csv(path),
                args.test_size,
                args.random_state,
                args.n_estimators,
            )
        )

    print(f"\n{'=' * 72}\nLocal versus centralized baseline comparison")
    print(
        f"{'Client':<12} {'Attack-Group composition':<30} "
        f"{'Local accuracy':>14} {'Local F1':>10} "
        f"{'Central acc.':>14} {'Central F1':>12}"
    )
    for result in results:
        print(
            f"{result['client']:<12} {result['attack_groups']:<30} "
            f"{result['accuracy']:>14.4f} {result['f1']:>10.4f} "
            f"{CENTRALIZED_ACCURACY:>14.4f} {CENTRALIZED_F1:>12.4f}"
        )


if __name__ == "__main__":
    main()
