"""Simulate true FedAvg training of an MLP across the three client shards.

No client training rows or labels are pooled for model fitting.  In each round,
every client starts from the current global neural-network parameters, performs
local ``partial_fit`` updates, and the server averages the resulting parameters
weighted by that client's chronological training-set size.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier


CENTRALIZED_ACCURACY = 0.9244
CENTRALIZED_F1 = 0.9574
LOCAL_BASELINES = {
    "client_1": (0.9882, 0.9932),
    "client_2": (0.8809, 0.9315),
    "client_3": (0.9963, 0.9979),
}

FEATURE_COLUMNS = ["shunt_voltage", "bus_voltage_V", "current_mA", "power_mW", "State"]
NUMERIC_COLUMNS = FEATURE_COLUMNS[:-1]
TARGET_COLUMN = "Label"
TIME_COLUMN = "time"
GROUP_COLUMN = "Attack-Group"
LABELS = np.array(["attack", "benign"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-dir", type=Path, default=Path("data") / "clients")
    parser.add_argument("--rounds", type=int, default=20, help="FedAvg communication rounds.")
    parser.add_argument("--local-epochs", type=int, default=3, help="partial_fit epochs per client per round.")
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
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
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, TIME_COLUMN, GROUP_COLUMN])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{client_name} is missing columns: {sorted(missing)}")


def make_features(data: pd.DataFrame, state_categories: list[str]) -> np.ndarray:
    """Build a fixed, shared feature schema without combining client rows."""
    numeric = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
    states = pd.get_dummies(data["State"].astype(str), prefix="State", dtype=float)
    state_columns = [f"State_{state}" for state in state_categories]
    states = states.reindex(columns=state_columns, fill_value=0.0)
    return np.hstack((numeric.to_numpy(dtype=float), states.to_numpy(dtype=float)))


def federated_standardization(training_features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-client sufficient statistics; raw rows never leave clients."""
    total_rows = sum(len(features) for features in training_features)
    feature_sum = sum((features.sum(axis=0) for features in training_features), np.zeros(training_features[0].shape[1]))
    squared_sum = sum((np.square(features).sum(axis=0) for features in training_features), np.zeros(training_features[0].shape[1]))
    mean = feature_sum / total_rows
    variance = np.maximum(squared_sum / total_rows - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return mean, scale


def standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (features - mean) / scale


def make_model(input_dim: int, hidden_units: int, learning_rate: float, random_state: int) -> MLPClassifier:
    """Create an MLP and initialize its parameter arrays through partial_fit."""
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=learning_rate,
        momentum=0.0,
        alpha=0.0001,
        batch_size=256,
        max_iter=1,
        shuffle=True,
        random_state=random_state,
    )
    # partial_fit initializes parameter arrays on first use.  The two synthetic
    # rows establish the public-API model state before any client trains; they
    # are not part of the training or evaluation data.
    model.partial_fit(np.zeros((2, input_dim)), LABELS, classes=LABELS)
    return model


def average_models(models: list[MLPClassifier], row_counts: list[int]) -> MLPClassifier:
    """FedAvg: weight every layer parameter by the number of local train rows."""
    total_rows = sum(row_counts)
    averaged = copy.deepcopy(models[0])
    averaged.coefs_ = [
        sum(model.coefs_[layer] * count for model, count in zip(models, row_counts)) / total_rows
        for layer in range(len(averaged.coefs_))
    ]
    averaged.intercepts_ = [
        sum(model.intercepts_[layer] * count for model, count in zip(models, row_counts)) / total_rows
        for layer in range(len(averaged.intercepts_))
    ]
    return averaged


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("--rounds and --local-epochs must both be positive.")

    client_files = [args.client_dir / f"client_{number}.csv" for number in range(1, 4)]
    missing = [str(path) for path in client_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing client shard(s): {', '.join(missing)}")

    clients: list[dict[str, object]] = []
    state_values: set[str] = set()
    for path in client_files:
        data = pd.read_csv(path)
        validate_columns(data, path.stem)
        data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="raise")
        train_indices, test_indices = chronological_group_split(data, args.test_size)
        train_data = data.loc[train_indices]
        test_data = data.loc[test_indices]
        state_values.update(train_data["State"].dropna().astype(str))
        clients.append({"name": path.stem, "train": train_data, "test": test_data})

    state_categories = sorted(state_values)
    if not state_categories:
        raise ValueError("No State values were found in the local training sets.")

    # Each shard is transformed locally with the shared schema; only aggregate
    # sums and sums of squares are used to obtain the common scaling constants.
    raw_train_features = [make_features(client["train"], state_categories) for client in clients]
    mean, scale = federated_standardization(raw_train_features)
    for client, raw_features in zip(clients, raw_train_features):
        train_data = client["train"]
        test_data = client["test"]
        client["X_train"] = standardize(raw_features, mean, scale)
        client["X_test"] = standardize(make_features(test_data, state_categories), mean, scale)
        client["y_train"] = train_data[TARGET_COLUMN].astype(str).to_numpy()
        client["y_test"] = test_data[TARGET_COLUMN].astype(str).to_numpy()

    global_model = make_model(len(NUMERIC_COLUMNS) + len(state_categories), args.hidden_units, args.learning_rate, args.random_state)
    row_counts = [len(client["X_train"]) for client in clients]
    print(f"FedAvg MLP: {args.rounds} rounds, {args.local_epochs} local epoch(s)/round")
    print("Client training rows: " + ", ".join(f"{client['name']}={count}" for client, count in zip(clients, row_counts)))

    for round_number in range(1, args.rounds + 1):
        local_models: list[MLPClassifier] = []
        for client in clients:
            local_model = copy.deepcopy(global_model)
            for _ in range(args.local_epochs):
                local_model.partial_fit(client["X_train"], client["y_train"])
            local_models.append(local_model)
        global_model = average_models(local_models, row_counts)
        print(f"Completed FedAvg round {round_number}/{args.rounds}.")

    # This is evaluation-only pooling: no test rows are used for fitting.
    X_test = np.vstack([client["X_test"] for client in clients])
    y_test = np.concatenate([client["y_test"] for client in clients])
    predictions = global_model.predict(X_test)
    if len(np.unique(predictions)) == 1:
        print(
            "WARNING: Predictions collapsed to a single class "
            f"({predictions[0]!r}); this is a degenerate model, not genuine convergence."
        )
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="attack", zero_division=0)
    recall = recall_score(y_test, predictions, pos_label="attack", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="attack", zero_division=0)

    print(f"\n{'=' * 72}\nFinal federated global-model evaluation (pooled chronological tests)")
    print(f"Test rows: {len(y_test)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; label order: attack, benign):")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    print(f"\n{'=' * 72}\nComparison with established baselines")
    print(f"{'Model':<24} {'Accuracy':>10} {'F1':>10}")
    print(f"{'FedAvg global MLP':<24} {accuracy:>10.4f} {f1:>10.4f}")
    print(f"{'Centralized Random Forest':<24} {CENTRALIZED_ACCURACY:>10.4f} {CENTRALIZED_F1:>10.4f}")
    for client_name, (local_accuracy, local_f1) in LOCAL_BASELINES.items():
        print(f"{client_name + ' local RF':<24} {local_accuracy:>10.4f} {local_f1:>10.4f}")


if __name__ == "__main__":
    main()
