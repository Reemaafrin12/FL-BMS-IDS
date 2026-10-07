"""Explain the default DP-style FedAvg MLP with sampled SHAP values."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

# Keep Matplotlib's generated font/cache files inside the project when the
# default user-profile cache directory is unavailable.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from train_federated import (
    LABELS,
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
from train_federated_dp import add_gaussian_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-dir", type=Path, default=Path("data") / "clients")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "shap")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.001)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def train_default_dp_model(
    client_dir: Path, epsilon: float, noise_multiplier: float, random_state: int
) -> tuple[MLPClassifier, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Reuse the federated data pipeline and train default 20x3 DP-style FedAvg."""
    client_files = [client_dir / f"client_{number}.csv" for number in range(1, 4)]
    missing = [str(path) for path in client_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing client shard(s): {', '.join(missing)}")
    if epsilon <= 0 or noise_multiplier < 0:
        raise ValueError("--epsilon must be positive and --noise-multiplier cannot be negative.")

    clients: list[dict[str, object]] = []
    state_values: set[str] = set()
    for path in client_files:
        data = pd.read_csv(path)
        validate_columns(data, path.stem)
        data[TIME_COLUMN] = pd.to_datetime(data[TIME_COLUMN], errors="raise")
        train_indices, test_indices = chronological_group_split(data, 0.20)
        train_data, test_data = data.loc[train_indices], data.loc[test_indices]
        state_values.update(train_data["State"].dropna().astype(str))
        clients.append({"name": path.stem, "train": train_data, "test": test_data})

    state_categories = sorted(state_values)
    raw_training = [make_features(client["train"], state_categories) for client in clients]
    mean, scale = federated_standardization(raw_training)
    for client, raw_features in zip(clients, raw_training):
        client["X_train"] = standardize(raw_features, mean, scale)
        client["X_test"] = standardize(make_features(client["test"], state_categories), mean, scale)
        client["y_train"] = client["train"][TARGET_COLUMN].astype(str).to_numpy()
        client["y_test"] = client["test"][TARGET_COLUMN].astype(str).to_numpy()

    model = make_model(len(NUMERIC_COLUMNS) + len(state_categories), 32, 0.01, random_state)
    row_counts = [len(client["X_train"]) for client in clients]
    noise_rng = np.random.default_rng(random_state)
    print("Training default DP-style FedAvg model: 20 rounds x 3 local epochs.")
    for round_number in range(1, 21):
        local_models: list[MLPClassifier] = []
        for client in clients:
            local_model = copy.deepcopy(model)
            for _ in range(3):
                local_model.partial_fit(client["X_train"], client["y_train"])
            local_models.append(local_model)
        model = average_models(local_models, row_counts)
        add_gaussian_noise(model, epsilon, noise_multiplier, noise_rng)
        print(f"  Completed round {round_number}/20.")

    X_train = np.vstack([client["X_train"] for client in clients])
    X_test = np.vstack([client["X_test"] for client in clients])
    y_test = np.concatenate([client["y_test"] for client in clients])
    feature_names = [*NUMERIC_COLUMNS, *(f"State_{state}" for state in state_categories)]
    return model, X_train, X_test, y_test, feature_names


def class_explanation(
    explanation: shap.Explanation, class_index: int, feature_names: list[str]
) -> shap.Explanation:
    """Extract a two-dimensional SHAP Explanation for one predict_proba class."""
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim != 3:
        raise ValueError(f"Expected multiclass SHAP values with 3 dimensions; received {values.shape}.")
    if values.shape[-1] != len(LABELS):
        raise ValueError(f"Unexpected SHAP output class axis: {values.shape}.")
    class_base = base_values[:, class_index] if base_values.ndim == 2 else base_values
    return shap.Explanation(
        values=values[:, :, class_index],
        base_values=class_base,
        data=explanation.data,
        feature_names=feature_names,
    )


def save_waterfall(explanation: shap.Explanation, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, X_train, X_test, y_test, feature_names = train_default_dp_model(
        args.client_dir, args.epsilon, args.noise_multiplier, args.random_state
    )
    predictions = model.predict(X_test)
    print(f"Final model test accuracy before explanation: {accuracy_score(y_test, predictions):.4f}")

    rng = np.random.default_rng(args.random_state)
    attack_candidates = np.flatnonzero((y_test == "attack") & (predictions == "attack"))
    benign_candidates = np.flatnonzero((y_test == "benign") & (predictions == "benign"))
    if not len(attack_candidates) or not len(benign_candidates):
        raise RuntimeError("Need one correctly classified attack and benign example for waterfall plots.")
    selected = [rng.choice(attack_candidates), rng.choice(benign_candidates)]
    remaining = np.setdiff1d(np.arange(len(X_test)), selected)
    additional = rng.choice(remaining, size=min(198, len(remaining)), replace=False)
    explain_indices = np.asarray([*selected, *additional])

    print("Creating a 100-point background sample from federated training features.")
    background = shap.sample(X_train, min(100, len(X_train)), random_state=args.random_state)
    print(f"Computing SHAP values for {len(explain_indices)} sampled chronological test rows.")
    # Six features make exact model-agnostic SHAP tractable with the sampled background.
    explainer = shap.Explainer(model.predict_proba, background, algorithm="exact", feature_names=feature_names)
    raw_explanation = explainer(X_test[explain_indices])
    attack_shap = class_explanation(raw_explanation, 0, feature_names)
    benign_shap = class_explanation(raw_explanation, 1, feature_names)

    importance = np.abs(attack_shap.values).mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:3]
    print("Top 3 features by mean absolute SHAP value (attack probability):")
    for rank, index in enumerate(top_indices, start=1):
        print(f"  {rank}. {feature_names[index]}: {importance[index]:.6f}")

    summary_path = args.output_dir / "federated_dp_shap_summary_bar.png"
    shap.plots.bar(attack_shap, show=False, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()
    save_waterfall(attack_shap[0], args.output_dir / "federated_dp_shap_attack_waterfall.png", "Correctly classified attack")
    save_waterfall(benign_shap[1], args.output_dir / "federated_dp_shap_benign_waterfall.png", "Correctly classified benign")
    print("Saved SHAP plots:")
    print(f"  {summary_path}")
    print(f"  {args.output_dir / 'federated_dp_shap_attack_waterfall.png'}")
    print(f"  {args.output_dir / 'federated_dp_shap_benign_waterfall.png'}")


if __name__ == "__main__":
    main()
