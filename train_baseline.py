"""Train and evaluate a weighted Random Forest baseline on all client shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


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
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional path for saving the fitted centralized Random Forest with joblib.",
    )
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


def main() -> None:
    args = parse_args()
    client_files = [args.client_dir / f"client_{number}.csv" for number in range(1, 4)]
    missing_files = [str(path) for path in client_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing client shard(s): {', '.join(missing_files)}")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    data = pd.concat((pd.read_csv(path) for path in client_files), ignore_index=True)
    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN, WEIGHT_COLUMN, TIME_COLUMN, GROUP_COLUMN]
    )
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Client CSVs are missing columns: {sorted(missing_columns)}")

    data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="raise")
    train_indices, test_indices = chronological_group_split(data, args.test_size)

    # One-hot encoding makes the categorical State values usable by the forest.
    features = pd.get_dummies(data[FEATURE_COLUMNS], columns=["State"], dtype=int)
    target = data[TARGET_COLUMN].astype(str)
    sample_weights = pd.to_numeric(data[WEIGHT_COLUMN], errors="raise")

    X_train = features.loc[train_indices]
    X_test = features.loc[test_indices]
    y_train = target.loc[train_indices]
    y_test = target.loc[test_indices]
    weights_train = sample_weights.loc[train_indices]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=weights_train)
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.save_model)
        print(f"Saved fitted model to: {args.save_model}")
    predictions = model.predict(X_test)

    labels = sorted(target.unique())
    print(f"Loaded {len(data)} rows from 3 client shards.")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        "Precision: "
        f"{precision_score(y_test, predictions, pos_label='attack', zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, pos_label='attack', zero_division=0):.4f}"
    )
    print(f"F1:        {f1_score(y_test, predictions, pos_label='attack', zero_division=0):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; label order: " + ", ".join(labels) + "):")
    print(confusion_matrix(y_test, predictions, labels=labels))


if __name__ == "__main__":
    main()
