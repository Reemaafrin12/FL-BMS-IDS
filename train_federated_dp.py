"""Run FedAvg with Gaussian noise injected into the global model each round.

This is an experimental DP-style mechanism: after FedAvg averages the client
MLP parameters, the server adds independent Gaussian noise to every coefficient
and intercept.  ``epsilon`` scales noise inversely, but this script is *not* a
certified (epsilon, delta)-DP implementation because it does not perform
per-example clipping or privacy-accountant composition across rounds.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from train_federated import (
    CENTRALIZED_ACCURACY,
    CENTRALIZED_F1,
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    LABELS,
    LOCAL_BASELINES,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    average_models,
    chronological_group_split,
    federated_standardization,
    make_features,
    make_model,
    standardize,
    validate_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-dir", type=Path, default=Path("data") / "clients")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Privacy-budget proxy; smaller values add more Gaussian noise.",
    )
    parser.add_argument(
        "--noise-multiplier",
        "--sigma",
        dest="noise_multiplier",
        type=float,
        default=0.001,
        help="Base per-parameter Gaussian standard deviation before division by epsilon.",
    )
    return parser.parse_args()


def add_gaussian_noise(
    model: MLPClassifier, epsilon: float, noise_multiplier: float, rng: np.random.Generator
) -> float:
    """Add independent N(0, (noise_multiplier / epsilon)^2) parameter noise."""
    noise_std = noise_multiplier / epsilon
    model.coefs_ = [weights + rng.normal(0.0, noise_std, weights.shape) for weights in model.coefs_]
    model.intercepts_ = [bias + rng.normal(0.0, noise_std, bias.shape) for bias in model.intercepts_]
    return noise_std


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("--rounds and --local-epochs must both be positive.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.noise_multiplier < 0:
        raise ValueError("--noise-multiplier must be non-negative.")

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
    raw_train_features = [make_features(client["train"], state_categories) for client in clients]
    mean, scale = federated_standardization(raw_train_features)
    for client, raw_features in zip(clients, raw_train_features):
        train_data = client["train"]
        test_data = client["test"]
        client["X_train"] = standardize(raw_features, mean, scale)
        client["X_test"] = standardize(make_features(test_data, state_categories), mean, scale)
        client["y_train"] = train_data[TARGET_COLUMN].astype(str).to_numpy()
        client["y_test"] = test_data[TARGET_COLUMN].astype(str).to_numpy()

    global_model = make_model(
        len(NUMERIC_COLUMNS) + len(state_categories), args.hidden_units, args.learning_rate, args.random_state
    )
    row_counts = [len(client["X_train"]) for client in clients]
    noise_rng = np.random.default_rng(args.random_state)
    noise_std = args.noise_multiplier / args.epsilon
    print(
        f"DP-style FedAvg MLP: {args.rounds} rounds, {args.local_epochs} local epoch(s)/round; "
        f"epsilon={args.epsilon:g}, noise std={noise_std:g}"
    )
    print("WARNING: This is not a certified (epsilon, delta)-DP guarantee; it is Gaussian parameter noise without clipping/accounting.")

    for round_number in range(1, args.rounds + 1):
        local_models: list[MLPClassifier] = []
        for client in clients:
            local_model = copy.deepcopy(global_model)
            for _ in range(args.local_epochs):
                local_model.partial_fit(client["X_train"], client["y_train"])
            local_models.append(local_model)
        global_model = average_models(local_models, row_counts)
        add_gaussian_noise(global_model, args.epsilon, args.noise_multiplier, noise_rng)
        print(f"Completed noisy FedAvg round {round_number}/{args.rounds}.")

    # Pooling is evaluation-only; test data never participates in fitting.
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

    print(f"\n{'=' * 72}\nFinal DP-style federated evaluation (pooled chronological tests)")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix (rows=true, columns=predicted; label order: attack, benign):")
    print(confusion_matrix(y_test, predictions, labels=LABELS))
    print(f"\n{'=' * 72}\nComparison with established baselines")
    print(f"{'Model':<28} {'Accuracy':>10} {'F1':>10}")
    print(f"{'DP-style FedAvg MLP':<28} {accuracy:>10.4f} {f1:>10.4f}")
    print(f"{'FedAvg-free centralized RF':<28} {CENTRALIZED_ACCURACY:>10.4f} {CENTRALIZED_F1:>10.4f}")
    for client_name, (local_accuracy, local_f1) in LOCAL_BASELINES.items():
        print(f"{client_name + ' local RF':<28} {local_accuracy:>10.4f} {local_f1:>10.4f}")


if __name__ == "__main__":
    main()
