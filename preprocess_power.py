"""Prepare non-IID federated-learning client data from EVSE-B power readings.

Benign rows from the ``none`` attack group are split across all clients, while
each attack group is assigned to exactly one client. Consequently, every client
can train on both labels while retaining distinct attack distributions.
The output files include a ``class_weight`` column that can be passed as sample
weights (or used to construct a weighted loss) during model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "power_consumption_EVSE-B-PowerCombined.csv"
DEFAULT_OUTPUT_DIR = Path("data") / "clients"
REQUIRED_COLUMNS = {"Label", "Attack-Group"}
BENIGN_SHUFFLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-clients", type=int, default=3)
    return parser.parse_args()


def normalized(value: str | None) -> str:
    """Return a stable category name, including for blank CSV values."""
    value = (value or "").strip()
    return value if value else "<missing>"


def calculate_class_weights(label_counts: Counter[str]) -> dict[str, float]:
    """Return balanced weights: total_samples / (classes * class_samples)."""
    total = sum(label_counts.values())
    class_count = len(label_counts)
    return {
        label: total / (class_count * count)
        for label, count in sorted(label_counts.items())
    }


def assign_groups_to_clients(
    grouped_rows: dict[str, list[dict[str, str]]],
    num_clients: int,
    initial_sizes: list[int],
) -> list[list[str]]:
    """Keep attack groups intact and balance clients approximately by row count."""
    assignments: list[list[str]] = [[] for _ in range(num_clients)]
    client_sizes = initial_sizes.copy()

    # Largest-first greedy bin packing gives useful shard sizes without mixing
    # attack groups, preserving the requested non-IID partition.
    for group, rows in sorted(grouped_rows.items(), key=lambda item: (-len(item[1]), item[0])):
        client_index = min(range(num_clients), key=lambda index: (client_sizes[index], index))
        assignments[client_index].append(group)
        client_sizes[client_index] += len(rows)
    return assignments


def split_rows_evenly(
    rows: list[dict[str, str]], num_clients: int
) -> list[list[dict[str, str]]]:
    """Split rows into near-equal, disjoint chunks whose sizes differ by at most one."""
    base_size, remainder = divmod(len(rows), num_clients)
    chunks: list[list[dict[str, str]]] = []
    start = 0
    for client_index in range(num_clients):
        chunk_size = base_size + (client_index < remainder)
        chunks.append(rows[start : start + chunk_size])
        start += chunk_size
    return chunks


def main() -> None:
    args = parse_args()
    if args.num_clients != 3:
        raise ValueError("This preprocessing workflow requires exactly 3 clients.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(
                f"Input must contain {sorted(REQUIRED_COLUMNS)}; found {reader.fieldnames}"
            )
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV contains no data rows.")

    label_counts = Counter(normalized(row["Label"]) for row in rows)
    class_weights = calculate_class_weights(label_counts)

    print("Label class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print("\nBalanced class weights (use for weighted loss/sample weighting):")
    for label, weight in class_weights.items():
        print(f"  {label}: {weight:.6f}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row["class_weight"] = f"{class_weights[normalized(row['Label'])]:.12g}"
        grouped_rows[normalized(row["Attack-Group"])].append(row)

    benign_group = "none"
    benign_rows = grouped_rows.pop(benign_group, None)
    if not benign_rows:
        raise ValueError("Expected benign rows in the 'none' Attack-Group.")
    if any(normalized(row["Label"]) != "benign" for row in benign_rows):
        raise ValueError("The 'none' Attack-Group must contain only benign rows.")

    # Avoid giving each client a contiguous block of the source-file order
    # (for example, idle rows followed by charging rows).
    random.Random(BENIGN_SHUFFLE_SEED).shuffle(benign_rows)
    benign_shards = split_rows_evenly(benign_rows, args.num_clients)
    assignments = assign_groups_to_clients(
        grouped_rows,
        args.num_clients,
        initial_sizes=[len(shard) for shard in benign_shards],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_fields = [*fieldnames, "class_weight"]

    manifest: dict[str, Any] = {
        "input_file": str(args.input),
        "class_weights": class_weights,
        "clients": {},
    }
    for client_number, groups in enumerate(assignments, start=1):
        client_rows = [*benign_shards[client_number - 1]]
        client_rows.extend(row for group in groups for row in grouped_rows[group])
        output_file = args.output_dir / f"client_{client_number}.csv"
        with output_file.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(client_rows)

        client_labels = Counter(normalized(row["Label"]) for row in client_rows)
        manifest["clients"][f"client_{client_number}"] = {
            "file": output_file.name,
            "attack_groups": groups,
            "benign_rows_from_none": len(benign_shards[client_number - 1]),
            "rows": len(client_rows),
            "label_distribution": dict(sorted(client_labels.items())),
        }
        print(f"\nClient {client_number}: {len(client_rows)} rows; attack groups: {groups}")
        print(f"  Label distribution: {dict(sorted(client_labels.items()))}")

    manifest_file = args.output_dir / "client_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
    print(f"\nSaved shards and manifest to: {args.output_dir}")


if __name__ == "__main__":
    main()
